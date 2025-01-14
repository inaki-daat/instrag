from pathlib import Path
from llama_index.readers.file import UnstructuredReader
from llama_index.core import Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
import joblib
import os
from functools import lru_cache
from .utils.cache import Cache

class CustomObjectRetriever:
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
            query_engine_tools=tools, llm=self._llm
        )
        sub_question_description = """\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]

def initialize_retriever():
    # Load your documents
    documents = SimpleDirectoryReader('../files').load_data()
    
    # Create the index
    index = VectorStoreIndex.from_documents(documents)
    
    # Get the retriever
    retriever = index.as_retriever()
    
    # Store in cache
    Cache.get_instance().set('retriever', retriever)

def get_retriever():
    cache = Cache.get_instance()
    retriever = cache.get('retriever')
    if retriever is None:
        raise RuntimeError("Retriever not initialized")
    return retriever

