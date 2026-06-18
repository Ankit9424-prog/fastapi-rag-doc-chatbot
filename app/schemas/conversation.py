"""Schemas for conversational RAG API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity. Auto-generated if not provided."
    )
    message: str = Field(
        ..., min_length=1, max_length=4096, description="User's message/query"
    )


class SourceChunk(BaseModel):
    """A retrieved source chunk used for the response."""

    document_id: str
    filename: str
    chunk_index: int
    content: str = Field(..., description="Chunk text content")
    relevance_score: float = Field(..., description="Similarity score")


class BookingInfo(BaseModel):
    """Extracted booking information."""

    booking_id: str
    candidate_name: str
    candidate_email: str
    interview_date: str
    interview_time: str
    status: str


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    session_id: str = Field(..., description="Session identifier")
    response: str = Field(..., description="Assistant's response")
    sources: list[SourceChunk] = Field(
        default_factory=list, description="Retrieved source chunks"
    )
    booking: Optional[BookingInfo] = Field(
        None, description="Booking info if an interview was scheduled"
    )


class ChatMessage(BaseModel):
    """A single message in chat history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO format timestamp")


class ChatHistoryResponse(BaseModel):
    """Response for chat history endpoint."""

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    total_messages: int
