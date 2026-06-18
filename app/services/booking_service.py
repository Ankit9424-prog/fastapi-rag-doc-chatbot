"""Interview booking service with LLM-based extraction."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from google import genai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.booking import InterviewBooking

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class BookingExtractionResult:
    """Result of booking intent detection and extraction."""

    is_booking_intent: bool
    candidate_name: str | None = None
    candidate_email: str | None = None
    interview_date: str | None = None
    interview_time: str | None = None
    missing_fields: list[str] | None = None
    follow_up_message: str | None = None

    @property
    def is_complete(self) -> bool:
        """Check if all booking fields are present."""
        return all([
            self.candidate_name,
            self.candidate_email,
            self.interview_date,
            self.interview_time,
        ])


BOOKING_EXTRACTION_PROMPT = """You are an AI assistant that detects interview booking intent and extracts booking details from conversations.

Analyze the following conversation and determine:
1. Does the user want to book/schedule an interview?
2. If yes, extract the following fields (if provided):
   - candidate_name: The candidate's full name
   - candidate_email: The candidate's email address
   - interview_date: The interview date (in YYYY-MM-DD format)
   - interview_time: The interview time (in HH:MM format, 24-hour)

Respond with ONLY a valid JSON object in this exact format:
{{
    "is_booking_intent": true/false,
    "candidate_name": "name or null",
    "candidate_email": "email or null",
    "interview_date": "YYYY-MM-DD or null",
    "interview_time": "HH:MM or null",
    "missing_fields": ["list of missing field names"],
    "follow_up_message": "A polite message asking for missing info, or confirming the booking if complete. null if not a booking intent."
}}

Conversation context:
{conversation_context}

Latest user message:
{user_message}"""


class BookingService:
    """Handle interview booking with LLM-based extraction."""

    def __init__(self, db: AsyncSession, settings: Settings) -> None:
        self._db = db
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.llm_model

    async def detect_and_extract(
        self,
        user_message: str,
        conversation_context: str,
        session_id: str,
    ) -> BookingExtractionResult:
        """Detect booking intent and extract details using LLM.

        Args:
            user_message: The latest user message.
            conversation_context: Formatted conversation history.
            session_id: Current session ID.

        Returns:
            BookingExtractionResult with extraction details.
        """
        prompt = BOOKING_EXTRACTION_PROMPT.format(
            conversation_context=conversation_context,
            user_message=user_message,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        try:
            result_data = json.loads(response.text)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse booking extraction response")
            return BookingExtractionResult(is_booking_intent=False)

        extraction = BookingExtractionResult(
            is_booking_intent=result_data.get("is_booking_intent", False),
            candidate_name=result_data.get("candidate_name"),
            candidate_email=result_data.get("candidate_email"),
            interview_date=result_data.get("interview_date"),
            interview_time=result_data.get("interview_time"),
            missing_fields=result_data.get("missing_fields"),
            follow_up_message=result_data.get("follow_up_message"),
        )

        # If all fields are present, save the booking
        if extraction.is_booking_intent and extraction.is_complete:
            await self._save_booking(extraction, session_id)

        return extraction

    async def _save_booking(
        self,
        extraction: BookingExtractionResult,
        session_id: str,
    ) -> InterviewBooking:
        """Save a complete booking to the database."""
        booking = InterviewBooking(
            id=str(uuid.uuid4()),
            session_id=session_id,
            candidate_name=extraction.candidate_name,  # type: ignore[arg-type]
            candidate_email=extraction.candidate_email,  # type: ignore[arg-type]
            interview_date=extraction.interview_date,  # type: ignore[arg-type]
            interview_time=extraction.interview_time,  # type: ignore[arg-type]
            status="confirmed",
        )
        self._db.add(booking)
        await self._db.flush()
        logger.info(
            "Booking saved: %s for %s on %s at %s",
            booking.id,
            booking.candidate_name,
            booking.interview_date,
            booking.interview_time,
        )
        return booking

    async def get_all_bookings(self) -> list[InterviewBooking]:
        """Retrieve all interview bookings."""
        result = await self._db.execute(
            select(InterviewBooking).order_by(InterviewBooking.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_booking_by_id(self, booking_id: str) -> InterviewBooking | None:
        """Retrieve a specific booking by ID."""
        result = await self._db.execute(
            select(InterviewBooking).where(InterviewBooking.id == booking_id)
        )
        return result.scalar_one_or_none()
