#!/usr/bin/env python3
"""Benchmark Gemini Embedding API throughput and verify correctness."""
import time
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Config
EMBED_MODEL = "models/gemini-embedding-001"
CHUNKS_FILE = Path("/storage/epstein_llm/data/processed/final_epstein_documents_2024_chunks.json")


def batch_embed(texts: list[str], dim: int = 768) -> list[list[float]]:
    """Embed a batch of texts."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=texts,
        task_type="retrieval_document",
        output_dimensionality=dim,
    )
    return result["embedding"]


def benchmark(chunks: list[dict], batch_size: int, workers: int, max_chunks: int = 500):
    """Run benchmark with given parameters."""
    texts = [c["text"] for c in chunks[:max_chunks]]
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    embeddings = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(batch_embed, batch): i for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            try:
                result = future.result()
                embeddings.extend(result)
            except Exception as e:
                print(f"  Error: {e}")
                return None, None, None

    elapsed = time.time() - start
    rate = len(texts) / elapsed

    return len(texts), elapsed, rate, embeddings


def verify_embeddings(embeddings: list[list[float]], chunks: list[dict]):
    """Verify embeddings are valid and semantically meaningful."""
    import numpy as np

    print("\n=== Verification ===")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Check dimensions
    dims = set(len(e) for e in embeddings)
    print(f"All same dimension: {len(dims) == 1}")

    # Check values are normalized (should be ~1.0 for cosine similarity)
    norms = [np.linalg.norm(e) for e in embeddings[:10]]
    print(f"Sample norms (should be ~1.0): {[f'{n:.3f}' for n in norms]}")

    # Test semantic similarity
    print("\n=== Semantic Test ===")
    # Find two similar chunks (from same document, adjacent)
    if len(chunks) >= 2:
        e1, e2 = np.array(embeddings[0]), np.array(embeddings[1])
        sim_adjacent = np.dot(e1, e2)
        print(f"Similarity between adjacent chunks: {sim_adjacent:.3f}")

    # Compare first and last chunk (should be less similar)
    if len(chunks) >= 100:
        e_first, e_last = np.array(embeddings[0]), np.array(embeddings[99])
        sim_distant = np.dot(e_first, e_last)
        print(f"Similarity between distant chunks: {sim_distant:.3f}")

    # Test query embedding
    print("\n=== Query Test ===")
    query = "flight logs and passengers"
    query_embed = batch_embed([query])[0]
    query_np = np.array(query_embed)

    # Find most similar chunk
    similarities = [np.dot(query_np, np.array(e)) for e in embeddings[:100]]
    best_idx = np.argmax(similarities)
    print(f"Query: '{query}'")
    print(f"Best match (idx {best_idx}, sim {similarities[best_idx]:.3f}):")
    print(f"  {chunks[best_idx]['text'][:200]}...")


def main():
    print("Loading chunks...")
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")
    print(f"Sample chunk length: {len(chunks[0]['text'])} chars")

    # Test configurations
    configs = [
        (50, 1),   # Baseline: small batch, single thread
        (100, 1),  # Larger batch
        (100, 3),  # Some parallelism
        (100, 5),  # ExperienceEmbed default
    ]

    print("\n=== Benchmark Results ===")
    print(f"{'Batch':<8} {'Workers':<8} {'Chunks':<8} {'Time':<10} {'Rate':<15}")
    print("-" * 55)

    best_embeddings = None
    best_rate = 0

    for batch_size, workers in configs:
        result = benchmark(chunks, batch_size, workers, max_chunks=200)
        if result[0] is not None:
            n, t, rate, embeddings = result
            print(f"{batch_size:<8} {workers:<8} {n:<8} {t:.2f}s{'':<5} {rate:.1f} chunks/sec")
            if rate > best_rate:
                best_rate = rate
                best_embeddings = embeddings
        else:
            print(f"{batch_size:<8} {workers:<8} FAILED")

        time.sleep(1)  # Small delay between tests

    # Verify with best config
    if best_embeddings:
        verify_embeddings(best_embeddings, chunks)

    # Cost estimate for full dataset
    print("\n=== Cost Estimate ===")
    total_chars = sum(len(c['text']) for c in chunks)
    avg_tokens = total_chars / 4 / len(chunks)  # ~4 chars per token
    print(f"This file: {len(chunks)} chunks, ~{total_chars:,} chars, ~{int(avg_tokens)} tokens/chunk")

    # Project to full dataset
    estimated_chunks = 5_000_000
    estimated_tokens = estimated_chunks * avg_tokens
    cost_paid = estimated_tokens * 0.15 / 1_000_000
    cost_batch = estimated_tokens * 0.075 / 1_000_000

    print(f"\nFor {estimated_chunks:,} chunks (full dataset estimate):")
    print(f"  Estimated tokens: {estimated_tokens/1e9:.2f}B")
    print(f"  Paid tier ($0.15/1M): ${cost_paid:.2f}")
    print(f"  Batch API ($0.075/1M): ${cost_batch:.2f}")

    # Time estimate
    if best_rate > 0:
        time_hours = estimated_chunks / best_rate / 3600
        print(f"  Time at {best_rate:.0f} chunks/sec: {time_hours:.1f} hours")


if __name__ == "__main__":
    main()
