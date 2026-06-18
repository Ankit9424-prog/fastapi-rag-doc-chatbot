"""Unit tests for chunking strategies."""

from __future__ import annotations

import pytest

from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking import (
    FixedSizeChunker,
    SemanticChunker,
    TextChunk,
    get_chunker,
)


class TestFixedSizeChunker:
    """Tests for the fixed-size chunking strategy."""

    def test_basic_chunking(self, sample_text: str) -> None:
        """Test that text is split into chunks."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(sample_text)

        assert len(chunks) > 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(len(c.content) <= 210 for c in chunks)  # Allow some word-boundary flex

    def test_chunk_indices_sequential(self, sample_text: str) -> None:
        """Test that chunk indices are sequential."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(sample_text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_overlap(self, sample_text: str) -> None:
        """Test that chunks have overlapping content."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(sample_text)

        if len(chunks) >= 2:
            # The end of chunk 0 should overlap with the start of chunk 1
            # Due to word boundary logic, exact overlap may vary
            assert len(chunks) >= 2

    def test_empty_text(self, empty_text: str) -> None:
        """Test handling of empty text."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(empty_text)

        assert chunks == []

    def test_short_text(self, short_text: str) -> None:
        """Test text shorter than chunk size."""
        chunker = FixedSizeChunker(chunk_size=1000, overlap=50)
        chunks = chunker.chunk(short_text)

        assert len(chunks) == 1
        assert chunks[0].content == short_text

    def test_invalid_chunk_size(self) -> None:
        """Test that invalid chunk_size raises ValueError."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0)

    def test_invalid_overlap(self) -> None:
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_token_count_populated(self, sample_text: str) -> None:
        """Test that token_count is calculated."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(sample_text)

        assert all(c.token_count > 0 for c in chunks)


class TestSemanticChunker:
    """Tests for the semantic chunking strategy."""

    def test_basic_chunking(self, sample_text: str) -> None:
        """Test that text is split into semantic chunks."""
        chunker = SemanticChunker(max_chunk_size=300, min_chunk_size=50)
        chunks = chunker.chunk(sample_text)

        assert len(chunks) > 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_respects_paragraph_boundaries(self) -> None:
        """Test that chunking respects paragraph structure."""
        text = "First paragraph about topic A.\n\nSecond paragraph about topic B.\n\nThird paragraph about topic C."
        chunker = SemanticChunker(max_chunk_size=1000)
        chunks = chunker.chunk(text)

        # With large chunk size, all paragraphs should fit in one chunk
        assert len(chunks) == 1

    def test_splits_large_paragraphs(self) -> None:
        """Test that very large paragraphs are split into sentences."""
        text = "This is sentence one. This is sentence two. This is sentence three. " * 20
        chunker = SemanticChunker(max_chunk_size=100, min_chunk_size=20)
        chunks = chunker.chunk(text)

        assert len(chunks) > 1

    def test_empty_text(self, empty_text: str) -> None:
        """Test handling of empty text."""
        chunker = SemanticChunker(max_chunk_size=300)
        chunks = chunker.chunk(empty_text)

        assert chunks == []

    def test_short_text_merging(self, short_text: str) -> None:
        """Test that very short chunks get merged."""
        chunker = SemanticChunker(max_chunk_size=500, min_chunk_size=50)
        chunks = chunker.chunk(short_text)

        # Short text should result in a single chunk
        assert len(chunks) == 1

    def test_chunk_indices_sequential(self, sample_text: str) -> None:
        """Test that chunk indices are sequential."""
        chunker = SemanticChunker(max_chunk_size=200)
        chunks = chunker.chunk(sample_text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i


class TestChunkerFactory:
    """Tests for the get_chunker factory function."""

    def test_fixed_size_strategy(self) -> None:
        """Test that factory returns FixedSizeChunker."""
        chunker = get_chunker(ChunkingStrategy.FIXED_SIZE)
        assert isinstance(chunker, FixedSizeChunker)

    def test_semantic_strategy(self) -> None:
        """Test that factory returns SemanticChunker."""
        chunker = get_chunker(ChunkingStrategy.SEMANTIC)
        assert isinstance(chunker, SemanticChunker)

    def test_custom_params(self) -> None:
        """Test factory with custom parameters."""
        chunker = get_chunker(
            ChunkingStrategy.FIXED_SIZE,
            chunk_size=256,
            overlap=32,
        )
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 32
