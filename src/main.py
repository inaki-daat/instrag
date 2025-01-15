from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import query
from .config.settings import get_settings
from .config.logging_config import setup_logging
from src.core.agent import DocumentAgent
import os
from fastapi.responses import JSONResponse
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logger = setup_logging()

# Get settings and set environment variables
settings = get_settings()
os.environ["OPENAI_API_KEY"] = settings.openai_api_key
os.environ["COHERE_API_KEY"] = settings.cohere_api_key

logger.info("Starting FastAPI application")
app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize empty state
app.state.docs_cache = {}
app.state.is_ready = False
app.state.is_loading = False
app.state.loading_error = None

@app.get("/health")
async def health_check():
    """Health check that returns detailed status"""
    status_code = status.HTTP_200_OK
    
    response_data = {
        "status": "starting",
        "is_loading": app.state.is_loading,
        "is_ready": app.state.is_ready
    }
    
    if app.state.loading_error:
        response_data["error"] = str(app.state.loading_error)
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    elif app.state.is_ready:
        response_data["status"] = "healthy"
    
    return JSONResponse(
        content=response_data,
        status_code=status_code
    )

async def run_in_executor(func, *args):
    """Run a blocking function in an executor"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(ThreadPoolExecutor(), func, *args)

async def load_documents():
    """Background task to load documents"""
    if app.state.is_loading:
        return
        
    app.state.is_loading = True
    app.state.loading_error = None
    
    try:
        # Initialize DocumentAgent
        doc_agent = DocumentAgent(cache_dir="./data/llamaindex_docs")
        logger.info("DocumentAgent initialized")
        
        # Load documents in a separate thread
        docs = await run_in_executor(
            doc_agent.load_documents,
            "./files",
            100  # limit
        )
        logger.info(f"Loaded {len(docs)} documents")
        
        # Build agents dictionary and cache
        agents_dict, extra_info_dict = await doc_agent.build_agents(docs)
        logger.info("Built agents dictionary and cache")
        
        # Store in app state
        app.state.docs_cache["agents_dict"] = agents_dict
        app.state.docs_cache["extra_info_dict"] = extra_info_dict
        app.state.is_ready = True
        logger.info("Stored agents in app state")
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}", exc_info=True)
        app.state.loading_error = e
        raise
    finally:
        app.state.is_loading = False

@app.on_event("startup")
async def startup_event():
    logger.info("Running startup event")
    # Start document loading in the background
    asyncio.create_task(load_documents())

app.include_router(query.router, prefix="/api/v1")