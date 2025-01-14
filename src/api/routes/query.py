from fastapi import APIRouter, HTTPException, Request
from ..models.query import QueryRequest, QueryResponse
from typing import Dict
from src.core.agent import DocumentAgent
from llama_index.agent.openai import OpenAIAgent
import logging

logger = logging.getLogger("llama_index_api")
router = APIRouter()

# Store agents by user_id
agents_cache: Dict[str, OpenAIAgent] = {}

@router.post("/query", response_model=QueryResponse)
async def query_agent(request: Request, query_request: QueryRequest):
    logger.info(f"Processing query for user {query_request.user_id}")
    try:
        # Get or initialize agent for this user
        if query_request.user_id not in agents_cache:
            logger.info(f"Initializing new agent for user {query_request.user_id}")
            # Get cached documents from app state
            agents_dict = request.app.state.docs_cache["agents_dict"]
            extra_info_dict = request.app.state.docs_cache["extra_info_dict"]
            
            # Create document agent just for creating top agent
            doc_agent = DocumentAgent(cache_dir=f"./data/user_{query_request.user_id}")
            
            # Create top agent using cached documents
            agent = doc_agent.create_top_agent(agents_dict, extra_info_dict)
            agents_cache[query_request.user_id] = agent
            logger.info(f"Agent created and cached for user {query_request.user_id}")
        
        # Get existing agent
        agent = agents_cache[query_request.user_id]
        logger.debug(f"Retrieved cached agent for user {query_request.user_id}")
        
        # Query the agent
        logger.debug(f"Querying agent with question: {query_request.question}")
        response = await agent.aquery(query_request.question)
        logger.info(f"Query completed for user {query_request.user_id}")
        
        return QueryResponse(response=str(response))

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup endpoint
@router.delete("/cleanup/{user_id}")
async def cleanup_agent(user_id: str):
    if user_id in agents_cache:
        del agents_cache[user_id]
        return {"message": f"Agent for user {user_id} cleaned up"}
    raise HTTPException(status_code=404, detail="User agent not found")