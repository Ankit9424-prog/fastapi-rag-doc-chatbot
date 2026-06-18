"""Document ingestion API endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.dependencies import get_db, get_embedding_service, get_vector_store
from app.models.document import Document, DocumentChunk
from app.schemas.ingestion import (
    ChunkingStrategy,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.chunking import get_chunker
from app.services.embedding import EmbeddingService
from app.services.text_extractor import UnsupportedFileTypeError, extract_text
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description="Upload a PDF or TXT file, extract text, chunk it using the selected strategy, "
    "generate embeddings, and store vectors in Qdrant with metadata in PostgreSQL.",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF or TXT file to upload")],
    chunking_strategy: Annotated[
        ChunkingStrategy,
        Form(description="Chunking strategy: 'fixed_size' or 'semantic'"),
    ] = ChunkingStrategy.FIXED_SIZE,
    db: AsyncSession = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentResponse:
    """Upload, process, and ingest a document into the RAG system."""
    settings = get_settings()

    # Step 1: Validate and extract text
    try:
        text = await extract_text(file)
    except UnsupportedFileTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # Step 2: Chunk text
    chunker = get_chunker(
        strategy=chunking_strategy,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk(text)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No chunks could be generated from the document.",
        )

    # Step 3: Create document record
    document = Document(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        file_size=file.size or 0,
        chunking_strategy=chunking_strategy.value,
        total_chunks=len(chunks),
    )
    db.add(document)
    await db.flush()  # Get the document ID

    # Step 4: Generate embeddings
    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = await embedding_service.generate_embeddings(chunk_texts)

    # Step 5: Store vectors in Qdrant
    payloads = [
        {
            "document_id": document.id,
            "filename": document.filename,
            "chunk_index": chunk.index,
            "content": chunk.content,
        }
        for chunk in chunks
    ]
    vector_ids = await vector_store.upsert_vectors(embeddings, payloads)

    # Step 6: Save chunk metadata to PostgreSQL
    for chunk, vector_id in zip(chunks, vector_ids):
        db_chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=chunk.index,
            content=chunk.content,
            vector_id=vector_id,
            token_count=chunk.token_count,
        )
        db.add(db_chunk)

    await db.flush()

    logger.info(
        "Document '%s' ingested: %d chunks, strategy=%s",
        document.filename,
        len(chunks),
        chunking_strategy.value,
    )

    return DocumentResponse.model_validate(document)


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Retrieve all ingested documents with metadata."""
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    documents = list(result.scalars().all())
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=len(documents),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Retrieve a specific document by ID."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentDeleteResponse:
    """Delete a document and its associated chunks and vectors."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    chunks_count = document.total_chunks

    # Delete vectors from Qdrant
    await vector_store.delete_by_document_id(document_id)

    # Delete from database (cascades to chunks)
    await db.delete(document)
    await db.flush()

    logger.info("Document '%s' deleted with %d chunks", document.filename, chunks_count)

    return DocumentDeleteResponse(
        document_id=document_id,
        chunks_removed=chunks_count,
    )
