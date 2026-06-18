"""Document embedding service using FastEmbed locally."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastembed import TextEmbedding

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using local FastEmbed model."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.embedding_model
        logger.info("Initializing FastEmbed with model: %s", self._model)
        self._client = TextEmbedding(model_name=self._model)

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []

        # FastEmbed returns a generator of numpy arrays. We convert them to lists of floats.
        embeddings_generator = self._client.embed(texts)
        result = [emb.tolist() for emb in embeddings_generator]
        return result

    async def generate_single_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: String to embed.

        Returns:
            Embedding vector.
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]
