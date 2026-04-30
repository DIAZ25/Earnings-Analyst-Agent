from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"

    # LangFuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # SEC EDGAR
    edgar_user_agent: str = "EarningsAnalystAgent contact@example.com"

    # Companies House
    companies_house_api_key: str = ""

    # Storage
    vector_store_dir: str = "data/vector_stores"
    guidance_cache_dir: str = "data/guidance_cache"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
