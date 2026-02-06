"""Embeddings using Gemini API (replacing local GPU)."""

import time
import numpy as np
from typing import Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

from .config import GEMINI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Embedding settings
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_BATCH_SIZE = 100   # texts per API call (max 250)
PARALLEL_REQUESTS = 200  # concurrent API calls (Tier 3: 4000+ RPM = ~67/sec)
EMBED_DIM = 768          # output dimension


def normalize(vectors: list[list[float]]) -> list[list[float]]:
    """Normalize vectors to unit length for cosine similarity."""
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = arr / norms
    return normalized.tolist()


def batch_embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Embed a batch of texts with retry logic."""
    for attempt in range(retries):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=texts,
                task_type="retrieval_document",
                output_dimensionality=EMBED_DIM,
            )
            return result["embedding"]
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Embed error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def batch_embed_query(text: str) -> list[float]:
    """Embed a single query text."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
        output_dimensionality=EMBED_DIM,
    )
    return result["embedding"]


class GeminiEmbeddingModel:
    """Wrapper for Gemini Embedding API (same interface as old EmbeddingModel)."""

    def __init__(
        self,
        batch_size: int = EMBED_BATCH_SIZE,
        parallel_requests: int = PARALLEL_REQUESTS,
        dimension: int = EMBED_DIM
    ):
        self.batch_size = batch_size
        self.parallel_requests = parallel_requests
        self.dimension = dimension
        logger.info(f"Using Gemini Embedding API: {EMBED_MODEL}")
        logger.info(f"Batch size: {batch_size}, Parallel requests: {parallel_requests}, Dimension: {dimension}")

    def embed(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """
        Embed a list of texts with parallelization.

        Args:
            texts: List of text strings to embed
            show_progress: Show progress (logged)

        Returns:
            List of normalized embedding vectors
        """
        if not texts:
            return []

        # Create batches
        batches = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_embeddings = [None] * len(batches)

        logger.info(f"Embedding {len(texts)} texts in {len(batches)} batches...")

        # Process with parallelization
        with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
            future_to_idx = {executor.submit(batch_embed, b): i for i, b in enumerate(batches)}

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    all_embeddings[idx] = future.result()
                    completed += 1
                    if show_progress and completed % 10 == 0:
                        logger.info(f"  Progress: {completed}/{len(batches)} batches")
                except Exception as e:
                    logger.error(f"Batch {idx} failed: {e}")
                    raise

        # Flatten and normalize
        flat_embeddings = [e for batch in all_embeddings for e in batch]
        normalized = normalize(flat_embeddings)

        logger.info(f"Embedded {len(normalized)} texts")
        return normalized

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (uses retrieval_query task type)."""
        embedding = batch_embed_query(query)
        # Normalize single vector
        normalized = normalize([embedding])[0]
        return normalized


# Global model instance (lazy loaded)
_model: Optional[GeminiEmbeddingModel] = None


def get_embedding_model() -> GeminiEmbeddingModel:
    """Get or create the global embedding model instance."""
    global _model
    if _model is None:
        _model = GeminiEmbeddingModel()
    return _model


def embed_texts(texts: list[str], show_progress: bool = True) -> list[list[float]]:
    """Convenience function to embed texts using the global model."""
    model = get_embedding_model()
    return model.embed(texts, show_progress)


def embed_query(query: str) -> list[float]:
    """Convenience function to embed a query using the global model."""
    model = get_embedding_model()
    return model.embed_query(query)


def get_embedding_dimension() -> int:
    """Get the dimension of embeddings."""
    return EMBED_DIM
