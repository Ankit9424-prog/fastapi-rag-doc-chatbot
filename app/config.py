"""Application configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Authentication
    api_key: str = "changeme"

    # Groq API
    groq_api_key: str

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "llama3-8b-8192"
    embedding_dimension: int = 384

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "palmmind_documents"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    chat_history_ttl: int = 3600  # seconds

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://palmmind:palmmind_dev@localhost:5433/palmmind"

    # Chunking defaults
    chunk_size: int = 512
    chunk_overlap: int = 50

    # RAG
    retrieval_top_k: int = 5

    # CORS
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
