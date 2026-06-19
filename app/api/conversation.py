"""Conversational RAG API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_booking_service, get_chat_memory, get_db, get_rag_pipeline
from app.schemas.booking import BookingDetail, BookingListResponse
from app.schemas.conversation import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from app.services.booking_service import BookingService
from app.services.chat_memory import ChatMemoryService
from app.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversational RAG"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the RAG chatbot",
    description="Send a message and receive a contextual response based on ingested documents. "
    "Supports multi-turn conversations and interview booking.",
)
async def chat(
    request: ChatRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ChatResponse:
    """Process a chat message through the RAG pipeline."""
    try:
        response = await rag_pipeline.process(
            message=request.message,
            session_id=request.session_id,
        )
        return response
    except Exception:
        logger.exception("Error processing chat message")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message. Please try again.",
        )


@router.get(
    "/chat/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get chat history for a session",
)
async def get_chat_history(
    session_id: str,
    chat_memory: ChatMemoryService = Depends(get_chat_memory),
) -> ChatHistoryResponse:
    """Retrieve the conversation history for a specific session."""
    messages = await chat_memory.get_history(session_id)
    total = await chat_memory.get_message_count(session_id)

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
            )
            for msg in messages
        ],
        total_messages=total,
    )


@router.delete(
    "/chat/{session_id}",
    summary="Clear chat history",
)
async def clear_chat_history(
    session_id: str,
    chat_memory: ChatMemoryService = Depends(get_chat_memory),
) -> dict[str, str]:
    """Clear the conversation history for a specific session."""
    existed = await chat_memory.clear_history(session_id)
    if not existed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chat history found for session '{session_id}'.",
        )
    return {"message": f"Chat history for session '{session_id}' cleared successfully."}


@router.get(
    "/bookings",
    response_model=BookingListResponse,
    summary="List all interview bookings",
)
async def list_bookings(
    booking_service: BookingService = Depends(get_booking_service),
) -> BookingListResponse:
    """Retrieve all interview bookings."""
    bookings = await booking_service.get_all_bookings()
    return BookingListResponse(
        bookings=[BookingDetail.model_validate(b) for b in bookings],
        total=len(bookings),
    )


@router.get(
    "/bookings/{booking_id}",
    response_model=BookingDetail,
    summary="Get booking details",
)
async def get_booking(
    booking_id: str,
    booking_service: BookingService = Depends(get_booking_service),
) -> BookingDetail:
    """Retrieve a specific booking by ID."""
    booking = await booking_service.get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Booking with ID '{booking_id}' not found.",
        )
    return BookingDetail.model_validate(booking)
