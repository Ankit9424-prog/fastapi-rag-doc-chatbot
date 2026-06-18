"""Text extraction from PDF and TXT files."""

from __future__ import annotations

import chardet
import fitz  # PyMuPDF
from fastapi import UploadFile


class UnsupportedFileTypeError(Exception):
    """Raised when an unsupported file type is uploaded."""

    def __init__(self, content_type: str) -> None:
        self.content_type = content_type
        super().__init__(f"Unsupported file type: {content_type}")


SUPPORTED_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
}


async def extract_text(file: UploadFile) -> str:
    """Extract text content from an uploaded file.

    Args:
        file: FastAPI UploadFile (PDF or TXT).

    Returns:
        Extracted text as a string.

    Raises:
        UnsupportedFileTypeError: If the file type is not PDF or TXT.
        ValueError: If no text could be extracted.
    """
    content_type = file.content_type or ""

    # Also check file extension as fallback
    filename = file.filename or ""
    if content_type not in SUPPORTED_TYPES:
        if filename.lower().endswith(".pdf"):
            content_type = "application/pdf"
        elif filename.lower().endswith(".txt"):
            content_type = "text/plain"
        else:
            raise UnsupportedFileTypeError(content_type)

    file_bytes = await file.read()

    if content_type == "application/pdf":
        text = _extract_from_pdf(file_bytes)
    else:
        text = _extract_from_txt(file_bytes)

    if not text.strip():
        raise ValueError(f"No text content could be extracted from '{filename}'")

    return text


def _extract_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    pages: list[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            if page_text.strip():
                pages.append(page_text)
    return "\n\n".join(pages)


def _extract_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT bytes with encoding detection."""
    detected = chardet.detect(file_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    return file_bytes.decode(encoding)
