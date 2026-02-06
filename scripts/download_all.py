#!/usr/bin/env python3
"""Download all Epstein documents from DOJ."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.download import download_all_documents, download_2024_release
from src.config import RAW_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Download Epstein documents")
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Maximum number of files to download"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=RAW_DATA_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--test-2024",
        action="store_true",
        help="Download only the 2024 court release (943 pages) for testing"
    )

    args = parser.parse_args()

    if args.test_2024:
        print("Downloading 2024 court release (for testing)...")
        result = download_2024_release(args.output)
        if result:
            print(f"Downloaded: {result}")
        else:
            print("Download failed")
            sys.exit(1)
    else:
        print("Downloading DOJ documents...")
        print("NOTE: Document URL list needs to be configured in src/download.py")
        print("      Once DOJ release structure is known, update get_document_urls()")
        downloaded = download_all_documents(args.output, args.limit)
        print(f"Downloaded {len(downloaded)} files")


if __name__ == "__main__":
    main()
