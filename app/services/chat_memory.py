"""Redis-backed chat memory for multi-turn conversations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class ChatMemoryService:
    """Manage conversation history using Redis."""

    KEY_PREFIX = "chat"

    def __init__(self, client: aioredis.Redis, ttl: int = 3600) -> None:
        """Initialize chat memory service.

        Args:
            client: Async Redis client.
            ttl: Time-to-live for chat history in seconds.
        """
        self._client = client
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"{self.KEY_PREFIX}:{session_id}:messages"

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append a message to the conversation history.

        Args:
            session_id: Unique session identifier.
            role: Message role ('user' or 'assistant').
            content: Message content.
        """
        key = self._key(session_id)
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        await self._client.rpush(key, message)
        await self._client.expire(key, self._ttl)
        logger.debug("Added %s message to session %s", role, session_id)

    async def get_history(
        self,
        session_id: str,
        last_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve conversation history.

        Args:
            session_id: Unique session identifier.
            last_n: Number of most recent messages to retrieve.
                    If None, returns all messages.

        Returns:
            List of message dicts with role, content, and timestamp.
        """
        key = self._key(session_id)

        if last_n is not None:
            raw_messages = await self._client.lrange(key, -last_n, -1)
        else:
            raw_messages = await self._client.lrange(key, 0, -1)

        return [json.loads(msg) for msg in raw_messages]

    async def clear_history(self, session_id: str) -> bool:
        """Clear conversation history for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            True if the history existed and was deleted.
        """
        key = self._key(session_id)
        result = await self._client.delete(key)
        logger.info("Cleared history for session %s", session_id)
        return result > 0

    async def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        key = self._key(session_id)
        return await self._client.llen(key)
