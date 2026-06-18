"""Tests for conversation services (mocked LLM and Redis)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app.services.chat_memory import ChatMemoryService


class TestChatMemory:
    """Tests for Redis-backed chat memory using mocked Redis."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.rpush = AsyncMock()
        redis.expire = AsyncMock()
        redis.lrange = AsyncMock(return_value=[])
        redis.llen = AsyncMock(return_value=0)
        redis.delete = AsyncMock(return_value=1)
        return redis

    @pytest.fixture
    def memory(self, mock_redis: AsyncMock) -> ChatMemoryService:
        """Create a ChatMemoryService with mocked Redis."""
        return ChatMemoryService(client=mock_redis, ttl=3600)

    @pytest.mark.asyncio
    async def test_add_message(self, memory: ChatMemoryService, mock_redis: AsyncMock) -> None:
        """Test adding a message to chat history."""
        await memory.add_message("session-1", "user", "Hello")

        mock_redis.rpush.assert_called_once()
        call_args = mock_redis.rpush.call_args
        assert call_args[0][0] == "chat:session-1:messages"

        message_data = json.loads(call_args[0][1])
        assert message_data["role"] == "user"
        assert message_data["content"] == "Hello"
        assert "timestamp" in message_data

    @pytest.mark.asyncio
    async def test_get_history(self, memory: ChatMemoryService, mock_redis: AsyncMock) -> None:
        """Test retrieving chat history."""
        mock_messages = [
            json.dumps({"role": "user", "content": "Hi", "timestamp": "2024-01-01T00:00:00"}),
            json.dumps({"role": "assistant", "content": "Hello!", "timestamp": "2024-01-01T00:00:01"}),
        ]
        mock_redis.lrange.return_value = mock_messages

        history = await memory.get_history("session-1")

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_history_with_limit(
        self, memory: ChatMemoryService, mock_redis: AsyncMock
    ) -> None:
        """Test retrieving limited chat history."""
        await memory.get_history("session-1", last_n=5)
        mock_redis.lrange.assert_called_once_with("chat:session-1:messages", -5, -1)

    @pytest.mark.asyncio
    async def test_clear_history(self, memory: ChatMemoryService, mock_redis: AsyncMock) -> None:
        """Test clearing chat history."""
        result = await memory.clear_history("session-1")

        assert result is True
        mock_redis.delete.assert_called_once_with("chat:session-1:messages")

    @pytest.mark.asyncio
    async def test_get_message_count(self, memory: ChatMemoryService, mock_redis: AsyncMock) -> None:
        """Test getting message count."""
        mock_redis.llen.return_value = 5

        count = await memory.get_message_count("session-1")

        assert count == 5
        mock_redis.llen.assert_called_once_with("chat:session-1:messages")

    @pytest.mark.asyncio
    async def test_ttl_set_on_add(
        self, memory: ChatMemoryService, mock_redis: AsyncMock
    ) -> None:
        """Test that TTL is set when adding a message."""
        await memory.add_message("session-1", "user", "test")
        mock_redis.expire.assert_called_once_with("chat:session-1:messages", 3600)
