#!/usr/bin/env python3
"""Phase 1: Extract text and create chunks with full multiprocessing.

Uses ProcessPoolExecutor to bypass the GIL entirely.
For files needing OCR, saves _ocr_pages.json for Phase 2 (process_ocr.py).
Skip detection uses _chunks.json existence (no DB needed during processing).
"""

import argparse
import sys
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    OCR_MIN_TEXT_THRESHOLD, DATASET_OCR_MODE,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Image extensions to support
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'}


def get_dataset_name(file_path: Path) -> str:
    """Extract dataset name from file path."""
    for part in file_path.parts:
        if part.startswith('dataset_'):
            return part
    if 'doj_2026' in str(file_path) and 'dataset' not in str(file_path):
        return 'doj_2026_other'
    return 'other'


def process_one_file(args: tuple) -> tuple[str, int, bool, str, int]:
    """
    Extract text, chunk, save, and flag pages for OCR.
    Runs in a separate process - no shared state, no locks.

    Args:
        args: (pdf_path, ocr_mode) tuple
            ocr_mode: "none", "all", or "heuristic"

    Returns:
        (filename, num_chunks, success, error_msg, num_ocr_pages)
    """
    import fitz
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import re

    pdf_path, ocr_mode = args

    try:
        # Extract text
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        pages_text = []
        ocr_pages = []

        for page_idx, page in enumerate(doc):
            page_text = page.get_text().strip()
            pages_text.append(page_text)

            # Determine if this page needs OCR
            if ocr_mode == "all":
                ocr_pages.append(page_idx)
            elif ocr_mode == "heuristic":
                if len(page_text) < OCR_MIN_TEXT_THRESHOLD:
                    ocr_pages.append(page_idx)
            # "none": no pages flagged

        doc.close()

        text = '\n\n'.join(pages_text)

        # Minimal cleaning
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunk_texts = splitter.split_text(text)
        chunks = [
            {"text": c, "source": pdf_path.stem, "chunk_index": i}
            for i, c in enumerate(chunk_texts)
        ]

        # Save outputs
        dataset_name = get_dataset_name(pdf_path)
        chunks_dir = PROCESSED_DATA_DIR / dataset_name
        chunks_dir.mkdir(parents=True, exist_ok=True)

        chunks_path = chunks_dir / f"{pdf_path.stem}_chunks.json"
        with open(chunks_path, "w") as f:
            json.dump(chunks, f)

        text_path = chunks_dir / f"{pdf_path.stem}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Save OCR manifest for Phase 2
        if ocr_pages:
            ocr_manifest = {
                "pages": ocr_pages,
                "total_pages": page_count,
                "source_pdf": str(pdf_path),
            }
            ocr_path = chunks_dir / f"{pdf_path.stem}_ocr_pages.json"
            with open(ocr_path, "w") as f:
                json.dump(ocr_manifest, f)

        return pdf_path.name, len(chunks), True, "", len(ocr_pages)
    except Exception as e:
        return pdf_path.name, 0, False, str(e), 0


def get_all_files(input_dir: Path, datasets: list[str] | None = None) -> list[Path]:
    """Get all document files from input directory, optionally filtered by dataset."""
    if datasets:
        files = []
        for ds in datasets:
            # Search for dataset dir directly or nested (e.g. raw/doj_2026/dataset_12)
            ds_dirs = [input_dir / ds] + list(input_dir.glob(f"**/{ds}"))
            for ds_dir in ds_dirs:
                if ds_dir.is_dir():
                    files.extend(ds_dir.glob("**/*.pdf"))
                    for ext in IMAGE_EXTENSIONS:
                        files.extend(ds_dir.glob(f"**/*{ext}"))
        return sorted(files)

    files = list(input_dir.glob("**/*.pdf"))
    for ext in IMAGE_EXTENSIONS:
        files.extend(input_dir.glob(f"**/*{ext}"))
    return sorted(files)


def get_processed_stems(dataset_name: str) -> set[str]:
    """Get stems of already-processed files from _chunks.json existence."""
    chunks_dir = PROCESSED_DATA_DIR / dataset_name
    if not chunks_dir.exists():
        return set()
    return {f.stem.replace("_chunks", "") for f in chunks_dir.glob("*_chunks.json")}


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Extract text and flag pages for OCR")
    parser.add_argument("--input", "-i", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--limit", "-l", type=int)
    parser.add_argument("--workers", "-w", type=int, default=28,
                        help="Number of parallel processes (default: 28, leave headroom for OS)")
    parser.add_argument("--batch-size", "-b", type=int, default=5000)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--dataset", "-d", nargs="+",
                        help="Process specific dataset(s), e.g. --dataset dataset_1 dataset_12")
    parser.add_argument("--force", action="store_true",
                        help="Re-process files even if _chunks.json already exists")
    args = parser.parse_args()

    # Get all files (optionally filtered by dataset)
    logger.info(f"Scanning {args.input} for documents...")
    all_files = get_all_files(args.input, args.dataset)
    logger.info(f"Found {len(all_files)} total files")

    if args.force:
        doc_files = all_files
        logger.info("--force: re-processing all files")
    else:
        # Filter already processed using _chunks.json existence
        logger.info("Loading processed files cache...")
        processed_stems = set()
        if PROCESSED_DATA_DIR.exists():
            for subdir in PROCESSED_DATA_DIR.iterdir():
                if subdir.is_dir():
                    for chunk_file in subdir.glob("*_chunks.json"):
                        processed_stems.add(chunk_file.stem.replace("_chunks", ""))
            for chunk_file in PROCESSED_DATA_DIR.glob("*_chunks.json"):
                processed_stems.add(chunk_file.stem.replace("_chunks", ""))

        logger.info(f"Already processed: {len(processed_stems)}")
        doc_files = [f for f in all_files if f.stem not in processed_stems]

    if args.limit:
        doc_files = doc_files[:args.limit]

    # Build work items: (pdf_path, ocr_mode) tuples
    work_items = []
    for f in doc_files:
        ds = get_dataset_name(f)
        ocr_mode = DATASET_OCR_MODE.get(ds, "none")
        work_items.append((f, ocr_mode))

    if args.stats_only:
        from collections import Counter
        mode_counts = Counter(mode for _, mode in work_items)
        print(f"Total files: {len(all_files)}")
        print(f"To process: {len(work_items)}")
        print(f"OCR modes: {dict(mode_counts)}")
        return

    total_files = len(work_items)
    logger.info(f"Processing {total_files} files with {args.workers} processes (batch size {args.batch_size})")

    processed = 0
    total_chunks = 0
    total_ocr_pages = 0
    failed = 0
    start_time = time.time()

    for batch_start in range(0, total_files, args.batch_size):
        batch = work_items[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (total_files + args.batch_size - 1) // args.batch_size
        logger.info(f"=== Batch {batch_num}/{total_batches}: {len(batch)} files ===")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_file, item): item for item in batch}

            for future in as_completed(futures):
                name, n_chunks, success, error, n_ocr = future.result()
                if success:
                    processed += 1
                    total_chunks += n_chunks
                    total_ocr_pages += n_ocr
                else:
                    failed += 1
                    if failed <= 20:
                        logger.error(f"Failed: {name}: {error}")

                done = processed + failed
                if done % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total_files - done) / rate / 60 if rate > 0 else 0
                    logger.info(
                        f"Progress: {done}/{total_files} ({done*100/total_files:.1f}%) | "
                        f"{rate:.1f} files/sec | "
                        f"~{remaining:.0f} min remaining | "
                        f"{total_chunks} chunks | {total_ocr_pages} OCR pages | {failed} failed"
                    )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed/60:.1f} minutes")
    print(f"Processed: {processed} | Failed: {failed}")
    print(f"Total chunks: {total_chunks}")
    print(f"OCR pages flagged: {total_ocr_pages}")
    print(f"Rate: {processed/elapsed:.1f} files/sec")
    if total_ocr_pages > 0:
        print(f"\nRun `python scripts/process_ocr.py` to process {total_ocr_pages} flagged OCR pages.")


if __name__ == "__main__":
    main()
