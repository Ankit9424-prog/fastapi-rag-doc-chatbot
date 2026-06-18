"""Embedding generation using Google Gemini."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from google import genai

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate text embeddings using Google Gemini API."""

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.embedding_model
        self._dimension = settings.embedding_dimension

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []

        # Process in batches of 100 (Gemini API limit)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self._client.models.embed_content(
                model=self._model,
                contents=batch,
            )
            for embedding in result.embeddings:
                embeddings.append(list(embedding.values))

            logger.info(
                "Generated embeddings for batch %d/%d (%d texts)",
                i // batch_size + 1,
                (len(texts) + batch_size - 1) // batch_size,
                len(batch),
            )

        return embeddings

    async def generate_single_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        result = self._client.models.embed_content(
            model=self._model,
            contents=[text],
        )
        return list(result.embeddings[0].values)
