"""Command-line interface for the Epstein Documents RAG system."""

import sys
import argparse
import logging

from .rag import ask
from .vectorstore import get_vector_store
from .config import TOP_K_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("  Epstein Documents RAG System")
    print("  Query the DOJ court document release")
    print("=" * 60)
    print("\nCommands:")
    print("  Type your question and press Enter")
    print("  'sources' - Toggle showing source excerpts")
    print("  'stats'   - Show index statistics")
    print("  'quit'    - Exit the program")
    print()


def interactive_mode(show_sources: bool = False):
    """Run interactive query mode."""
    print_banner()

    # Load vector store
    print("Loading vector store...")
    try:
        store = get_vector_store()
        print(f"Loaded {store.size} document chunks\n")
    except FileNotFoundError:
        print("ERROR: Vector store not found. Run 'python -m scripts.build_index' first.\n")
        return

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == 'quit':
            print("Goodbye!")
            break

        if query.lower() == 'sources':
            show_sources = not show_sources
            print(f"Source display: {'ON' if show_sources else 'OFF'}\n")
            continue

        if query.lower() == 'stats':
            print(f"\nIndex statistics:")
            print(f"  Document chunks: {store.size}")
            print()
            continue

        # Process query
        print("\nSearching documents...")
        try:
            result = ask(query, show_sources=show_sources)

            print("\n" + "-" * 60)
            print("ANSWER:")
            print("-" * 60)
            print(result["answer"])

            if show_sources and result.get("sources"):
                print("\n" + "-" * 60)
                print("SOURCES:")
                print("-" * 60)
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n[{i}] {source['source']} (score: {source['score']:.2f})")
                    print(f"    {source['text'][:200]}...")

            print()

        except Exception as e:
            logger.error(f"Query failed: {e}")
            print(f"\nError: {e}\n")


def single_query(question: str, show_sources: bool = False):
    """Run a single query and exit."""
    try:
        store = get_vector_store()
    except FileNotFoundError:
        print("ERROR: Vector store not found. Run 'python -m scripts.build_index' first.")
        sys.exit(1)

    result = ask(question, show_sources=show_sources)
    print(result["answer"])

    if show_sources and result.get("sources"):
        print("\n--- Sources ---")
        for i, source in enumerate(result["sources"], 1):
            print(f"[{i}] {source['source']}: {source['text'][:100]}...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Query Epstein court documents")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--sources", "-s", action="store_true", help="Show source excerpts")
    parser.add_argument("--top-k", "-k", type=int, default=TOP_K_RESULTS, help="Number of chunks to retrieve")

    args = parser.parse_args()

    if args.question:
        single_query(args.question, show_sources=args.sources)
    else:
        interactive_mode(show_sources=args.sources)


if __name__ == "__main__":
    main()
