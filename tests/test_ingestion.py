"""Integration tests for the ingestion API (mocked external services)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from io import BytesIO

from fastapi import UploadFile

from app.services.text_extractor import extract_text, UnsupportedFileTypeError


class TestTextExtractor:
    """Tests for text extraction service."""

    @pytest.mark.asyncio
    async def test_txt_extraction(self) -> None:
        """Test extracting text from a TXT file."""
        content = b"Hello, this is a test document."
        file = UploadFile(
            filename="test.txt",
            file=BytesIO(content),
            headers={"content-type": "text/plain"},
        )
        text = await extract_text(file)
        assert text == "Hello, this is a test document."

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self) -> None:
        """Test that unsupported file types raise an error."""
        file = UploadFile(
            filename="test.docx",
            file=BytesIO(b"content"),
            headers={"content-type": "application/vnd.openxmlformats"},
        )
        with pytest.raises(UnsupportedFileTypeError):
            await extract_text(file)

    @pytest.mark.asyncio
    async def test_txt_extension_fallback(self) -> None:
        """Test that .txt extension is detected even without proper content-type."""
        content = b"Text content"
        file = UploadFile(
            filename="test.txt",
            file=BytesIO(content),
            headers={"content-type": "application/octet-stream"},
        )
        text = await extract_text(file)
        assert text == "Text content"

    @pytest.mark.asyncio
    async def test_empty_file_raises_error(self) -> None:
        """Test that empty files raise ValueError."""
        file = UploadFile(
            filename="empty.txt",
            file=BytesIO(b""),
            headers={"content-type": "text/plain"},
        )
        with pytest.raises(ValueError, match="No text content"):
            await extract_text(file)
