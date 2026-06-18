"""Custom RAG pipeline — no RetrievalQAChain.

This module implements a fully manual Retrieval-Augmented Generation pipeline:
1. Load chat history from Redis
2. Rewrite query using LLM (for multi-turn context)
3. Generate query embedding
4. Search Qdrant for relevant chunks
5. Build augmented prompt with context
6. Generate response using LLM
7. Detect booking intent
8. Store messages in Redis
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from google import genai

from app.schemas.conversation import BookingInfo, ChatResponse, SourceChunk
from app.services.booking_service import BookingService
from app.services.chat_memory import ChatMemoryService
from app.services.embedding import EmbeddingService
from app.services.vector_store import SearchResult, VectorStoreService

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


QUERY_REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that captures the full context.

If the follow-up question is already standalone or the conversation history is empty, return it as-is.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


RAG_SYSTEM_PROMPT = """You are an intelligent assistant for Palm Mind AI. Answer questions based on the provided context documents.

Rules:
- Answer ONLY based on the provided context. If the context doesn't contain relevant information, say so.
- Be concise, professional, and helpful.
- If the user wants to book/schedule an interview, collect their name, email, preferred date (YYYY-MM-DD), and time (HH:MM). Guide them through providing each piece of information.
- Cite which document the information comes from when relevant.
- For multi-turn conversations, maintain coherence with previous messages."""


RAG_USER_PROMPT = """Context from documents:
---
{context}
---

Conversation history:
{history}

User question: {question}

Provide a helpful response based on the context above."""


@dataclass
class PipelineResult:
    """Internal result from the RAG pipeline."""

    response_text: str
    sources: list[SearchResult] = field(default_factory=list)
    booking_info: BookingInfo | None = None
    session_id: str = ""


class RAGPipeline:
    """Custom Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        chat_memory: ChatMemoryService,
        booking_service: BookingService,
        settings: Settings,
    ) -> None:
        self._embedding = embedding_service
        self._vector_store = vector_store
        self._memory = chat_memory
        self._booking = booking_service
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.llm_model
        self._top_k = settings.retrieval_top_k

    async def process(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ChatResponse:
        """Execute the full RAG pipeline.

        Args:
            message: User's input message.
            session_id: Optional session ID for conversation continuity.

        Returns:
            ChatResponse with the generated answer and sources.
        """
        # Step 0: Ensure session ID
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info("Processing message for session %s", session_id)

        # Step 1: Load chat history from Redis
        history = await self._memory.get_history(session_id, last_n=10)
        history_text = self._format_history(history)

        # Step 2: Rewrite query for multi-turn context
        standalone_query = await self._rewrite_query(message, history_text)
        logger.debug("Rewritten query: %s", standalone_query)

        # Step 3: Generate query embedding
        query_embedding = await self._embedding.generate_single_embedding(
            standalone_query
        )

        # Step 4: Search Qdrant for relevant chunks
        search_results = await self._vector_store.search(
            query_vector=query_embedding,
            top_k=self._top_k,
        )

        # Step 5: Build augmented prompt
        context_text = self._format_context(search_results)

        # Step 6: Generate response using LLM
        response_text = await self._generate_response(
            question=message,
            context=context_text,
            history=history_text,
        )

        # Step 7: Detect booking intent
        booking_info = await self._handle_booking(
            message=message,
            history_text=history_text,
            session_id=session_id,
            response_text=response_text,
        )

        # If booking produced a follow-up message, append it to the response
        if booking_info and isinstance(booking_info, dict) and booking_info.get("follow_up"):
            response_text = booking_info["follow_up"]
            booking_info_schema = booking_info.get("booking")
        else:
            booking_info_schema = booking_info

        # Step 8: Store messages in Redis
        await self._memory.add_message(session_id, "user", message)
        await self._memory.add_message(session_id, "assistant", response_text)

        # Build response
        sources = [
            SourceChunk(
                document_id=r.document_id,
                filename=r.filename,
                chunk_index=r.chunk_index,
                content=r.content[:200] + "..." if len(r.content) > 200 else r.content,
                relevance_score=round(r.score, 4),
            )
            for r in search_results
        ]

        return ChatResponse(
            session_id=session_id,
            response=response_text,
            sources=sources,
            booking=booking_info_schema if isinstance(booking_info_schema, BookingInfo) else None,
        )

    async def _rewrite_query(self, question: str, history: str) -> str:
        """Rewrite question to be standalone using conversation context."""
        if not history:
            return question

        prompt = QUERY_REWRITE_PROMPT.format(
            history=history,
            question=question,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
            ),
        )
        return response.text.strip() if response.text else question

    async def _generate_response(
        self,
        question: str,
        context: str,
        history: str,
    ) -> str:
        """Generate a response using the LLM with retrieved context."""
        user_prompt = RAG_USER_PROMPT.format(
            context=context if context else "No relevant documents found.",
            history=history if history else "No previous conversation.",
            question=question,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                genai.types.Content(
                    role="user",
                    parts=[genai.types.Part(text=RAG_SYSTEM_PROMPT + "\n\n" + user_prompt)],
                ),
            ],
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        return response.text.strip() if response.text else "I couldn't generate a response. Please try again."

    async def _handle_booking(
        self,
        message: str,
        history_text: str,
        session_id: str,
        response_text: str,
    ) -> BookingInfo | dict | None:
        """Detect booking intent and handle it."""
        extraction = await self._booking.detect_and_extract(
            user_message=message,
            conversation_context=history_text,
            session_id=session_id,
        )

        if not extraction.is_booking_intent:
            return None

        if extraction.is_complete:
            return BookingInfo(
                booking_id=str(uuid.uuid4()),
                candidate_name=extraction.candidate_name or "",
                candidate_email=extraction.candidate_email or "",
                interview_date=extraction.interview_date or "",
                interview_time=extraction.interview_time or "",
                status="confirmed",
            )

        # Incomplete booking — return follow-up message
        if extraction.follow_up_message:
            return {"follow_up": extraction.follow_up_message, "booking": None}

        return None

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """Format chat history for prompt inclusion."""
        if not history:
            return ""
        lines: list[str] = []
        for msg in history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """Format search results as context for the prompt."""
        if not results:
            return ""
        parts: list[str] = []
        for i, result in enumerate(results, 1):
            parts.append(
                f"[Source {i} - {result.filename} (chunk {result.chunk_index})]:\n"
                f"{result.content}\n"
            )
        return "\n".join(parts)
