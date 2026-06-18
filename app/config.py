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

    # Google Gemini
    gemini_api_key: str

    # Models
    embedding_model: str = "gemini-embedding-2"
    llm_model: str = "gemini-2.0-flash"
    embedding_dimension: int = 768

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


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
