#!/usr/bin/env python3
"""Search the vector store without calling Gemini - for debugging RAG retrieval."""

import sys
sys.path.insert(0, "/storage/epstein_llm")

from src.vectorstore import search

def main():
    if len(sys.argv) < 2:
        print("Usage: python search_only.py 'your query here' [num_results]")
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Query: {query}")
    print(f"Top {top_k} results:\n")

    results = search(query, top_k=top_k)

    for i, r in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"CHUNK {i} | Score: {r['score']:.3f} | Source: {r['source']}")
        print(f"{'='*60}")
        print(r['text'])
        print()

if __name__ == "__main__":
    main()
