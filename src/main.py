from fastapi import FastAPI
from .api.routes import query
from .config.settings import get_settings

app = FastAPI(title=get_settings().app_name)

app.include_router(query.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}