#!/usr/bin/env python3
"""Build FAISS vector index from processed chunks."""

import argparse
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import get_embedding_model, embed_texts
from src.vectorstore import VectorStore
from src.config import PROCESSED_DATA_DIR, VECTOR_STORE_DIR, EMBEDDING_BATCH_SIZE


def load_all_chunks(processed_dir: Path) -> list[dict]:
    """Load all chunk files from processed directory (including subdirectories)."""
    chunks = []
    # Search recursively for chunk files in all subdirectories
    chunk_files = sorted(processed_dir.glob("**/*_chunks.json"))

    print(f"Found {len(chunk_files)} chunk files")

    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        with open(chunk_file, "r") as f:
            file_chunks = json.load(f)
            chunks.extend(file_chunks)

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Input directory containing chunk files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=VECTOR_STORE_DIR,
        help="Output directory for index"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help="Embedding batch size"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )

    args = parser.parse_args()

    # Load chunks
    print("Loading chunks...")
    chunks = load_all_chunks(args.input)
    print(f"Total chunks to embed: {len(chunks)}")

    if not chunks:
        print("No chunks found. Run process_batch.py first.")
        sys.exit(1)

    # Initialize embedding model
    print("\nInitializing embedding model...")
    model = get_embedding_model()
    print(f"Embedding dimension: {model.dimension}")

    # Create vector store
    print("\nCreating vector store...")
    store = VectorStore(dimension=model.dimension)
    store.create_index(use_gpu=not args.no_gpu)

    # Embed in batches
    print(f"\nEmbedding {len(chunks)} chunks in batches of {args.batch_size}...")

    texts = [c["text"] for c in chunks]
    metadata = [{"text": c["text"], "source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    # Process in batches to show progress
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i + args.batch_size]
        batch_metadata = metadata[i:i + args.batch_size]

        embeddings = model.embed(batch_texts, show_progress=False)
        store.add(embeddings, batch_metadata)

    # Save index
    print(f"\nSaving index to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    store.save()

    print(f"\nDone! Index contains {store.size} vectors")
    print(f"Index saved to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
