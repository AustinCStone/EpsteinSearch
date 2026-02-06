"""Text chunking for RAG."""

from typing import Generator
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def create_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter with configured settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_text(
    text: str,
    source_id: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Split text into chunks with metadata.

    Args:
        text: The text to chunk
        source_id: Identifier for the source document
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of dicts with 'text', 'source', 'chunk_index' keys
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "source": source_id,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]


def chunk_documents(
    documents: list[tuple[str, str, dict]]
) -> Generator[dict, None, None]:
    """
    Chunk multiple documents.

    Args:
        documents: List of (doc_id, text, metadata) tuples

    Yields:
        Chunk dicts with text, source, chunk_index, and metadata
    """
    for doc_id, text, metadata in documents:
        chunks = chunk_text(text, doc_id)

        for chunk in chunks:
            chunk["metadata"] = metadata
            yield chunk


def estimate_chunk_count(text: str, chunk_size: int = CHUNK_SIZE) -> int:
    """Estimate number of chunks for a text."""
    # Rough estimate accounting for overlap
    return max(1, len(text) // (chunk_size - CHUNK_OVERLAP))
