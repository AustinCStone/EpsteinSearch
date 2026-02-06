#!/usr/bin/env python3
"""Fix orphaned files that have .txt but no _chunks.json."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunk import chunk_text

PROCESSED_DIR = Path("/storage/epstein_llm/data/processed/dataset_11")

def main():
    # Find all .txt files without corresponding _chunks.json
    txt_stems = {f.stem for f in PROCESSED_DIR.glob("*.txt")}
    chunk_stems = {f.stem.replace("_chunks", "") for f in PROCESSED_DIR.glob("*_chunks.json")}

    orphaned = txt_stems - chunk_stems
    print(f"Found {len(orphaned)} orphaned files")

    fixed = 0
    for stem in sorted(orphaned):
        txt_path = PROCESSED_DIR / f"{stem}.txt"
        chunks_path = PROCESSED_DIR / f"{stem}_chunks.json"

        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, source_id=stem)

        with open(chunks_path, "w") as f:
            json.dump(chunks, f)

        fixed += 1
        print(f"  [{fixed}] {stem}: {len(chunks)} chunks")

    print(f"\nFixed {fixed} orphaned files")

if __name__ == "__main__":
    main()
