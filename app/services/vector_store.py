"""Vector store operations using Qdrant."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    vector_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


class VectorStoreService:
    """Manage vector storage and retrieval in Qdrant."""

    def __init__(self, client: AsyncQdrantClient, settings: Settings) -> None:
        self._client = client
        self._collection = settings.qdrant_collection
        self._dimension = settings.embedding_dimension

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]

        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self._collection)
        else:
            logger.info("Qdrant collection already exists: %s", self._collection)

    async def upsert_vectors(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> list[str]:
        """Upsert vectors with metadata payloads.

        Args:
            vectors: List of embedding vectors.
            payloads: List of metadata dicts (must include document_id, filename, chunk_index, content).

        Returns:
            List of generated point IDs.
        """
        point_ids: list[str] = []
        points: list[PointStruct] = []

        for vector, payload in zip(vectors, payloads):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert (Qdrant handles large batches efficiently)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )

        logger.info("Upserted %d vectors to collection '%s'", len(points), self._collection)
        return point_ids

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.
            document_id: Optional filter by document ID.

        Returns:
            List of SearchResult objects.
        """
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        results = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            SearchResult(
                vector_id=str(point.id),
                document_id=point.payload.get("document_id", ""),
                filename=point.payload.get("filename", ""),
                chunk_index=point.payload.get("chunk_index", 0),
                content=point.payload.get("content", ""),
                score=point.score,
            )
            for point in results.points
        ]

    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all vectors associated with a document.

        Args:
            document_id: The document ID whose vectors should be removed.
        """
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info("Deleted vectors for document: %s", document_id)
