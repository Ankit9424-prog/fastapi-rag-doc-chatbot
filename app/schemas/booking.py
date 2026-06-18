"""Schemas for interview booking."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BookingDetail(BaseModel):
    """Detailed booking information."""

    id: str
    session_id: str
    candidate_name: str
    candidate_email: str
    interview_date: str
    interview_time: str
    status: str
    notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class BookingListResponse(BaseModel):
    """Response for listing all bookings."""

    bookings: list[BookingDetail] = Field(default_factory=list)
    total: int
