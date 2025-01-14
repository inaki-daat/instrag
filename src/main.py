from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import query
from .config.settings import get_settings
from .config.logging_config import setup_logging
from src.core.agent import DocumentAgent
import os

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
    allow_origins=["*"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize empty state
app.state.docs_cache = {}

app.include_router(query.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Running startup event")
    try:
        # Initialize DocumentAgent
        doc_agent = DocumentAgent(cache_dir="./data/llamaindex_docs")
        logger.info("DocumentAgent initialized")
        
        # Load documents
        docs = doc_agent.load_documents("./files", limit=100)
        logger.info(f"Loaded {len(docs)} documents")
        
        # Build agents dictionary and cache
        agents_dict, extra_info_dict = await doc_agent.build_agents(docs)
        logger.info("Built agents dictionary and cache")
        
        # Store in app state
        app.state.docs_cache["agents_dict"] = agents_dict
        app.state.docs_cache["extra_info_dict"] = extra_info_dict
        logger.info("Stored agents in app state")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint called")
    return {"status": "healthy"}