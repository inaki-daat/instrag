from fastapi import APIRouter, Depends, HTTPException
from ..models.query import QueryRequest, QueryResponse
from src.initialize import get_retriever
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_agent(query: QueryRequest):
    try:
        retriever = get_retriever()
        llm = OpenAI(model="gpt-4o")
        
        agent = OpenAIAgent.from_tools(
            tool_retriever=retriever,
            system_prompt="""\
You are an agent designed to answer queries about the documentation.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
""",
            llm=llm,
            verbose=True,
        )
        
        response = await agent.aquery(query.question)
        return QueryResponse(response=str(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))