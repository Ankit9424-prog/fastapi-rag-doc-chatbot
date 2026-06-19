"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from app.auth import verify_api_key
from app.config import get_settings
from app.db.session import Base, engine
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    # Startup
    settings = get_settings()

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")

    # Initialize Qdrant collection
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    vector_store = VectorStoreService(client=qdrant_client, settings=settings)
    await vector_store.ensure_collection()
    await qdrant_client.close()
    logger.info("Qdrant collection initialized")

    yield

    # Shutdown
    await engine.dispose()
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="PalmMind RAG Backend",
        description=(
            "A production-grade RAG backend with document ingestion and "
            "conversational AI capabilities. Built with FastAPI, Qdrant, "
            "Redis, and Groq."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers with API key authentication
    from app.api.conversation import router as conversation_router
    from app.api.ingestion import router as ingestion_router

    app.include_router(
        ingestion_router,
        prefix="/api/v1",
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        conversation_router,
        prefix="/api/v1",
        dependencies=[Depends(verify_api_key)],
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

    # Health check (no auth required)
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "palmmind-rag-backend"}

    return app


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = create_app()
