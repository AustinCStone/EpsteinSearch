"""FAISS vector store for document retrieval."""

import json
import numpy as np
from pathlib import Path
from typing import Optional
import logging

import faiss

from .config import VECTOR_STORE_DIR, PROCESSED_DATA_DIR, TOP_K_RESULTS
from .embeddings import get_embedding_dimension, embed_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"


class VectorStore:
    """FAISS-based vector store for document chunks."""

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension or get_embedding_dimension()
        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []

    def create_index(self, use_gpu: bool = True):
        """Create a new FAISS index."""
        # Use inner product (equivalent to cosine for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)

        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            logger.info("Using CPU FAISS index")

        self.metadata = []

    def add(self, embeddings: list[list[float]], metadata: list[dict]):
        """
        Add embeddings and metadata to the index.

        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dicts (must match embeddings length)
        """
        if self.index is None:
            self.create_index()

        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_np)
        self.metadata.extend(metadata)

        logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_RESULTS
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of dicts with 'text', 'source', 'score', and 'metadata'
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        query_np = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_np, top_k)

        results = []
        seen_sources = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue
            result = self.metadata[idx].copy()
            result["score"] = float(score)

            # Deduplicate by source document
            source = result.get("source", "")
            if source in seen_sources:
                continue
            seen_sources.add(source)

            # Load full document text
            full_text = self._load_full_text(source)
            if full_text is not None:
                result["full_text"] = full_text

            results.append(result)

        return results

    def _load_full_text(self, source: str) -> Optional[str]:
        """Load full document text from processed directory by source stem."""
        # Build lookup cache on first call
        if not hasattr(self, '_source_to_path'):
            self._source_to_path = {}
            if PROCESSED_DATA_DIR.exists():
                for ds_dir in PROCESSED_DATA_DIR.iterdir():
                    if ds_dir.is_dir():
                        txt_path = ds_dir / f"{source}.txt"
                        if txt_path.exists():
                            self._source_to_path[source] = txt_path

        # Check cache, or search if not found yet
        if source not in self._source_to_path:
            for ds_dir in PROCESSED_DATA_DIR.iterdir():
                if ds_dir.is_dir():
                    txt_path = ds_dir / f"{source}.txt"
                    if txt_path.exists():
                        self._source_to_path[source] = txt_path
                        break

        path = self._source_to_path.get(source)
        if path and path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def search_text(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """Search using a text query (embeds automatically)."""
        query_embedding = embed_query(query)
        return self.search(query_embedding, top_k)

    def save(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        """Save index and metadata to disk."""
        index_path = index_path or INDEX_FILE
        metadata_path = metadata_path or METADATA_FILE

        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Move to CPU for saving if on GPU
        cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'getDevice') else self.index
        faiss.write_index(cpu_index, str(index_path))

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

        logger.info(f"Saved index ({self.index.ntotal} vectors) to {index_path}")

    def load(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None, use_gpu: bool = True):
        """Load index and metadata from disk."""
        index_path = index_path or INDEX_FILE
        metadata_path = metadata_path or METADATA_FILE

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(str(index_path))

        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving loaded index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors")

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal if self.index else 0


# Global store instance
_store: Optional[VectorStore] = None


def get_vector_store(load_existing: bool = True) -> VectorStore:
    """Get or create the global vector store instance."""
    global _store
    if _store is None:
        _store = VectorStore()
        if load_existing and INDEX_FILE.exists():
            _store.load()
    return _store


def search(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Convenience function to search the global store."""
    store = get_vector_store()
    return store.search_text(query, top_k)
