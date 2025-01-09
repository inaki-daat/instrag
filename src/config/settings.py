from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "LlamaIndex API"
    openai_api_key: str
    environment: str = "development"
    model_path: str = "./saved_models"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()