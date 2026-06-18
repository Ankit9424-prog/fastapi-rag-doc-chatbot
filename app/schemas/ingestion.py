"""Schemas for document ingestion API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"


class DocumentResponse(BaseModel):
    """Response after document upload."""

    document_id: str = Field(..., alias="id", description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="File MIME type")
    file_size: int = Field(..., description="File size in bytes")
    chunking_strategy: ChunkingStrategy = Field(..., description="Strategy used for chunking")
    total_chunks: int = Field(..., description="Number of chunks created")
    created_at: datetime = Field(..., description="Upload timestamp")

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    documents: list[DocumentResponse] = Field(default_factory=list)
    total: int = Field(..., description="Total number of documents")


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""

    document_id: str
    message: str = "Document deleted successfully"
    chunks_removed: int = Field(..., description="Number of chunks removed")
