"""Embedding function + text chunking for the local vector index."""
import logging

from config.settings import CHUNK_OVERLAP, CHUNK_TOKENS

log = logging.getLogger(__name__)

# Rough approximation: 1 token ~= 4 characters
_CHUNK_CHARS = CHUNK_TOKENS * 4
_OVERLAP_CHARS = CHUNK_OVERLAP * 4


def get_embedding_function():
    """all-MiniLM-L6-v2 via ChromaDB's bundled ONNX runner (fully local).

    Uses the DefaultEmbeddingFunction: it pulls the model from ChromaDB's
    S3 bucket, which works even when HuggingFace's CDN is unreachable.
    """
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction()
    except Exception as e:  # noqa: BLE001
        log.warning("Default ONNX EF unavailable (%s); using Chroma fallback.", e)
        return None


def chunk_text(text: str, chunk_size: int = _CHUNK_CHARS, overlap: int = _OVERLAP_CHARS) -> list[str]:
    """Sliding-window character chunker (~384-token windows, ~50-token overlap)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks
