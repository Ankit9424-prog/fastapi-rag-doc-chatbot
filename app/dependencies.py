"""FastAPI dependency injection providers."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import async_session_factory
from app.services.booking_service import BookingService
from app.services.chat_memory import ChatMemoryService
from app.services.embedding import EmbeddingService
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStoreService


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_qdrant_client() -> AsyncGenerator[AsyncQdrantClient, None]:
    """Provide an async Qdrant client."""
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        yield client
    finally:
        await client.close()


async def get_redis_client() -> AsyncGenerator[aioredis.Redis, None]:
    """Provide an async Redis client."""
    settings = get_settings()
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


async def get_embedding_service() -> EmbeddingService:
    """Provide the embedding service."""
    settings = get_settings()
    return EmbeddingService(settings=settings)


async def get_vector_store(
    client: AsyncQdrantClient = Depends(get_qdrant_client),
) -> VectorStoreService:
    """Provide the vector store service."""
    settings = get_settings()
    return VectorStoreService(client=client, settings=settings)


async def get_chat_memory(
    redis_client: aioredis.Redis = Depends(get_redis_client),
) -> ChatMemoryService:
    """Provide the chat memory service."""
    settings = get_settings()
    return ChatMemoryService(client=redis_client, ttl=settings.chat_history_ttl)


async def get_booking_service(
    db: AsyncSession = Depends(get_db),
) -> BookingService:
    """Provide the booking service."""
    settings = get_settings()
    return BookingService(db=db, settings=settings)


async def get_rag_pipeline(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    chat_memory: ChatMemoryService = Depends(get_chat_memory),
    booking_service: BookingService = Depends(get_booking_service),
) -> RAGPipeline:
    """Provide the RAG pipeline with all dependencies."""
    settings = get_settings()
    return RAGPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chat_memory=chat_memory,
        booking_service=booking_service,
        settings=settings,
    )
