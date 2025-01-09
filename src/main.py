from fastapi import FastAPI
from .api.routes import query
from .config.settings import get_settings
import os

# Get settings and set environment variables
settings = get_settings()
os.environ["OPENAI_API_KEY"] = settings.openai_api_key
os.environ["COHERE_API_KEY"] = settings.cohere_api_key

app = FastAPI(title=settings.app_name)

app.include_router(query.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}