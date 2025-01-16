from pathlib import Path
import os
import pickle
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import re
import logging

logger = logging.getLogger("llama_index_api")

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
    SummaryIndex,
    load_index_from_storage
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex, ObjectRetriever
from llama_index.core.schema import QueryBundle
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.agent.openai import OpenAIAgent
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.readers.file import UnstructuredReader

class CustomObjectRetriever(ObjectRetriever):
    def __init__(
        self,
        retriever,
        object_node_mapping,
        node_postprocessors=None,
        llm=None,
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or OpenAI(model="gpt-4")
        self._node_postprocessors = node_postprocessors or []

    def retrieve(self, query_bundle):
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_str=query_bundle)

        nodes = self._retriever.retrieve(query_bundle)
        for processor in self._node_postprocessors:
            nodes = processor.postprocess_nodes(nodes, query_bundle=query_bundle)
            
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools,
            llm=self._llm
        )
        
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool",
                description="Useful for comparing multiple documents. Use for comparison queries with original query. Don't use other tools for multi-document queries."
            ),
        )

        return tools + [sub_question_tool]

class DocumentAgent:
    def __init__(self, cache_dir: str = "./data/llamaindex_docs"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize settings
        self.llm = OpenAI(model="gpt-4")
        Settings.llm = self.llm
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            embed_batch_size=256
        )
        
        self.reader = UnstructuredReader()
        self.node_parser = SentenceSplitter()
        
    def load_documents(self, directory: str, limit: int = 100) -> List[Document]:
        """Load documents from joblib file"""
        try:
            import joblib
            docs_path = Path('./app/saved_models/docs.joblib')
            
            if not docs_path.exists():
                raise FileNotFoundError(f"Pre-processed documents not found at {docs_path}")
            
            logger.info(f"Loading pre-processed documents from {docs_path}")
            docs = joblib.load(docs_path)
            logger.info(f"Successfully loaded {len(docs)} documents")
            
            return docs
            
        except Exception as e:
            logger.error(f"Error loading documents from joblib: {str(e)}", exc_info=True)
            raise

    async def build_agent_per_doc(self, nodes: List, file_base: str) -> Tuple[OpenAIAgent, str]:
        """Build an agent for a single document"""
        # Sanitize file_base for tool names - only allow alphanumeric, underscores, and hyphens
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', file_base)
        logger.debug(f"Sanitized tool name from {file_base} to {safe_name}")
        
        vi_out_path = self.cache_dir / file_base
        summary_out_path = self.cache_dir / f"{file_base}_summary.pkl"

        # Create summary_index first so it's available in all code paths
        summary_index = SummaryIndex(nodes)
        
        # Handle vector index
        if os.path.exists(vi_out_path):
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(vi_out_path))
            )
        else:
            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(persist_dir=str(vi_out_path))

        # Handle summary
        if os.path.exists(summary_out_path):
            with open(summary_out_path, 'rb') as f:
                summary = pickle.load(f)
        else:
            summary_query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize",
                llm=self.llm
            )
            summary = str(
                await summary_query_engine.aquery(
                    "Extract a concise 1-2 line summary of this document"
                )
            )
            with open(summary_out_path, 'wb') as f:
                pickle.dump(summary, f)

        # Create query engines and tools
        vector_query_engine = vector_index.as_query_engine(llm=self.llm)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            llm=self.llm
        )

        # Create tool names that strictly follow the pattern ^[a-zA-Z0-9_-]+$
        vector_tool_name = f"vector_{safe_name}"
        summary_tool_name = f"summary_{safe_name}"

        tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name=vector_tool_name,
                    description=f"Useful for specific fact questions about {file_base}"
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name=summary_tool_name,
                    description=f"Useful for summarization questions about {file_base}"
                ),
            ),
        ]

        agent = OpenAIAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=True,
            system_prompt=f"You are a specialized agent for '{file_base}'. Always use provided tools; don't rely on prior knowledge."
        )

        return agent, summary

    async def build_agents(self, docs: List[Document]) -> Tuple[Dict, Dict]:
        """Build agents for all documents"""
        agents_dict = {}
        extra_info_dict = {}

        for doc in tqdm(docs):
            nodes = self.node_parser.get_nodes_from_documents([doc])
            file_path = Path(doc.metadata["path"])
            file_base = f"{file_path.parent.stem}_{file_path.stem}"
            
            agent, summary = await self.build_agent_per_doc(nodes, file_base)
            
            agents_dict[file_base] = agent
            extra_info_dict[file_base] = {
                "summary": summary,
                "nodes": nodes
            }

        return agents_dict, extra_info_dict

    def create_top_agent(
        self,
        agents_dict: Dict,
        extra_info_dict: Dict,
        similarity_top_k: int = 10,
        rerank_top_n: int = 5
    ) -> OpenAIAgent:
        """Create the top-level agent"""
        tools = []
        for file_base, agent in agents_dict.items():
            summary = extra_info_dict[file_base]["summary"]
            doc_tool = QueryEngineTool(
                query_engine=agent,
                metadata=ToolMetadata(
                    name=f"tool_{file_base}",
                    description=summary,
                ),
            )
            tools.append(doc_tool)

        obj_index = ObjectIndex.from_objects(
            tools,
            index_cls=VectorStoreIndex,
        )
        
        vector_retriever = obj_index.as_node_retriever(
            similarity_top_k=similarity_top_k
        )

        custom_retriever = CustomObjectRetriever(
            vector_retriever,
            obj_index.object_node_mapping,
            node_postprocessors=[CohereRerank(top_n=rerank_top_n)],
            llm=self.llm
        )

        return OpenAIAgent.from_tools(
            tool_retriever=custom_retriever,
            system_prompt="You are a documentation query agent. Always use provided tools; don't rely on prior knowledge. You should always respond in markdown format.",
            llm=self.llm,
            verbose=True
        )

async def initialize_agent(
    docs_dir: str,
    cache_dir: str = "./data/llamaindex_docs",
    doc_limit: int = 100
) -> OpenAIAgent:
    """Initialize the complete agent system"""
    doc_agent = DocumentAgent(cache_dir=cache_dir)
    
    # Load documents
    docs = doc_agent.load_documents(docs_dir, limit=doc_limit)
    
    # Build individual agents
    agents_dict, extra_info_dict = await doc_agent.build_agents(docs)
    
    # Create and return top agent
    return doc_agent.create_top_agent(agents_dict, extra_info_dict)


