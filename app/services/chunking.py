"""Text chunking strategies for document processing."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.schemas.ingestion import ChunkingStrategy


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    content: str
    index: int
    start_char: int
    end_char: int
    token_count: int = 0

    def __post_init__(self) -> None:
        self.token_count = len(self.content.split())


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into chunks.

        Args:
            text: The input text to chunk.

        Returns:
            List of TextChunk objects.
        """
        ...


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size character chunks with overlap.

    This strategy divides text into chunks of a specified character count,
    with configurable overlap between consecutive chunks to preserve context.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into fixed-size chunks with overlap."""
        if not text.strip():
            return []

        chunks: list[TextChunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at word boundary if not at the end
            if end < len(text):
                last_space = chunk_text.rfind(" ")
                if last_space > self.chunk_size * 0.5:  # Only if space is in latter half
                    end = start + last_space
                    chunk_text = text[start:end]

            chunks.append(
                TextChunk(
                    content=chunk_text.strip(),
                    index=index,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
            start = end - self.overlap

        return [c for c in chunks if c.content]  # Filter empty chunks


class SemanticChunker(BaseChunker):
    """Split text by semantic boundaries (paragraphs and sentences).

    This strategy respects natural language structure by splitting on
    paragraph boundaries first, then grouping sentences together up to
    a maximum chunk size. This produces more semantically coherent chunks.
    """

    SENTENCE_PATTERN: re.Pattern[str] = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|\n\n+'
    )

    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 100) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into semantically meaningful chunks."""
        if not text.strip():
            return []

        # Step 1: Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Step 2: Split large paragraphs into sentences
        sentences: list[str] = []
        for para in paragraphs:
            if len(para) <= self.max_chunk_size:
                sentences.append(para)
            else:
                # Split paragraph into sentences
                para_sentences = self.SENTENCE_PATTERN.split(para)
                sentences.extend(s.strip() for s in para_sentences if s.strip())

        # Step 3: Group sentences into chunks respecting size limits
        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        current_length = 0
        char_offset = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.max_chunk_size and current_sentences:
                # Flush current buffer
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        index=len(chunks),
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                    )
                )
                char_offset += len(chunk_text) + 1
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_len + 1  # +1 for space/join

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            # Merge with last chunk if too small
            if len(chunk_text) < self.min_chunk_size and chunks:
                last = chunks[-1]
                merged_content = last.content + " " + chunk_text
                chunks[-1] = TextChunk(
                    content=merged_content,
                    index=last.index,
                    start_char=last.start_char,
                    end_char=last.start_char + len(merged_content),
                )
            else:
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        index=len(chunks),
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                    )
                )

        return chunks


def get_chunker(strategy: ChunkingStrategy, chunk_size: int = 512, overlap: int = 50) -> BaseChunker:
    """Factory function to create a chunker based on strategy selection.

    Args:
        strategy: The chunking strategy to use.
        chunk_size: Maximum chunk size in characters.
        overlap: Overlap between chunks (used by fixed_size strategy).

    Returns:
        An instance of the appropriate chunker.

    Raises:
        ValueError: If an unknown strategy is provided.
    """
    match strategy:
        case ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        case ChunkingStrategy.SEMANTIC:
            return SemanticChunker(max_chunk_size=chunk_size)
        case _:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
