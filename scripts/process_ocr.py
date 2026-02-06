#!/usr/bin/env python3
"""Phase 2: OCR flagged pages using Gemini Flash vision API.

Finds all _ocr_pages.json manifests left by Phase 1 (process_batch.py),
renders flagged pages to images, sends to Gemini for OCR, merges text
with pymupdf extraction, re-chunks, and overwrites output files.

Deletes _ocr_pages.json on success → built-in resume support.

Uses ThreadPoolExecutor since OCR is I/O-bound (API calls).
"""

import argparse
import sys
import json
import time
import re
import io
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PROCESSED_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, OCR_DPI,
)
from src.ocr import ocr_image_single

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def find_ocr_manifests(base_dir: Path, datasets: list[str] | None = None) -> list[Path]:
    """Find all _ocr_pages.json files under processed data directory."""
    if datasets:
        manifests = []
        for ds in datasets:
            ds_dir = base_dir / ds
            if ds_dir.exists():
                manifests.extend(sorted(ds_dir.glob("*_ocr_pages.json")))
        return manifests
    return sorted(base_dir.glob("**/*_ocr_pages.json"))


def process_one_manifest(manifest_path: Path) -> tuple[str, int, bool, str]:
    """
    Process a single OCR manifest: render pages, OCR, merge, re-chunk, save.

    Returns:
        (filename, num_ocr_pages, success, error_msg)
    """
    import fitz
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from PIL import Image

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        ocr_page_indices = manifest["pages"]
        total_pages = manifest["total_pages"]
        source_pdf = Path(manifest["source_pdf"])
        stem = manifest_path.stem.replace("_ocr_pages", "")
        parent_dir = manifest_path.parent

        if not source_pdf.exists():
            return stem, 0, False, f"Source PDF not found: {source_pdf}"

        # Open PDF and extract text per page + OCR flagged pages
        doc = fitz.open(str(source_pdf))
        ocr_set = set(ocr_page_indices)
        pages_text = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pymupdf_text = page.get_text().strip()

            if page_idx in ocr_set:
                # Render to image and OCR
                pix = page.get_pixmap(dpi=OCR_DPI)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = ocr_image_single(img)

                # Use OCR text if it's longer/better than pymupdf text
                if len(ocr_text) > len(pymupdf_text):
                    pages_text.append(ocr_text)
                else:
                    pages_text.append(pymupdf_text)
            else:
                pages_text.append(pymupdf_text)

        doc.close()

        # Reassemble full text
        text = '\n\n'.join(pages_text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        # Re-chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunk_texts = splitter.split_text(text)
        chunks = [
            {"text": c, "source": stem, "chunk_index": i}
            for i, c in enumerate(chunk_texts)
        ]

        # Overwrite .txt and _chunks.json
        text_path = parent_dir / f"{stem}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        chunks_path = parent_dir / f"{stem}_chunks.json"
        with open(chunks_path, "w") as f:
            json.dump(chunks, f)

        # Delete manifest (resume marker)
        manifest_path.unlink()

        return stem, len(ocr_page_indices), True, ""

    except Exception as e:
        return manifest_path.stem, 0, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: OCR flagged pages via Gemini")
    parser.add_argument("--workers", "-w", type=int, default=200,
                        help="Concurrent OCR threads (default: 200, Gemini Tier 3 = 4000 RPM)")
    parser.add_argument("--dataset", "-d", nargs="+",
                        help="Process specific dataset(s)")
    parser.add_argument("--limit", "-l", type=int,
                        help="Limit number of manifests to process")
    parser.add_argument("--stats-only", action="store_true",
                        help="Show stats without processing")
    args = parser.parse_args()

    # Find all OCR manifests
    logger.info("Scanning for _ocr_pages.json manifests...")
    manifests = find_ocr_manifests(PROCESSED_DATA_DIR, args.dataset)
    logger.info(f"Found {len(manifests)} files needing OCR")

    if not manifests:
        print("No _ocr_pages.json files found. Nothing to do.")
        return

    # Count total pages
    total_pages = 0
    for m in manifests:
        with open(m) as f:
            data = json.load(f)
            total_pages += len(data["pages"])

    if args.stats_only:
        from collections import Counter
        ds_counts = Counter()
        ds_pages = Counter()
        for m in manifests:
            ds = m.parent.name
            with open(m) as f:
                data = json.load(f)
                ds_counts[ds] += 1
                ds_pages[ds] += len(data["pages"])
        print(f"Total manifests: {len(manifests)}")
        print(f"Total OCR pages: {total_pages}")
        print(f"\nPer dataset:")
        for ds in sorted(ds_counts):
            print(f"  {ds}: {ds_counts[ds]} files, {ds_pages[ds]} pages")
        return

    if args.limit:
        manifests = manifests[:args.limit]

    # Process manifests. Each manifest processes its pages sequentially (one PDF),
    # but multiple manifests run concurrently via ThreadPoolExecutor.
    # The concurrency here is at the manifest level. Within each manifest,
    # pages are OCR'd sequentially to keep memory bounded.
    # With 200 workers and most PDFs being 1-2 pages, effective parallelism ≈ 200 API calls.
    total_manifests = len(manifests)
    logger.info(f"Processing {total_manifests} manifests ({total_pages} OCR pages) with {args.workers} threads")

    processed = 0
    total_ocr_done = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_manifest, m): m for m in manifests}

        for future in as_completed(futures):
            stem, n_pages, success, error = future.result()
            if success:
                processed += 1
                total_ocr_done += n_pages
            else:
                failed += 1
                if failed <= 50:
                    logger.error(f"Failed: {stem}: {error}")

            done = processed + failed
            if done % 100 == 0 or done == total_manifests:
                elapsed = time.time() - start_time
                pages_rate = total_ocr_done / elapsed if elapsed > 0 else 0
                remaining_pages = total_pages - total_ocr_done
                remaining_min = remaining_pages / pages_rate / 60 if pages_rate > 0 else 0
                logger.info(
                    f"Progress: {done}/{total_manifests} files | "
                    f"{total_ocr_done}/{total_pages} pages | "
                    f"{pages_rate:.1f} pages/sec | "
                    f"~{remaining_min:.0f} min remaining | "
                    f"{failed} failed"
                )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"OCR completed in {elapsed/60:.1f} minutes")
    print(f"Processed: {processed} files | Failed: {failed}")
    print(f"OCR pages: {total_ocr_done}")
    if elapsed > 0:
        print(f"Rate: {total_ocr_done/elapsed:.1f} pages/sec")

    # Verify no manifests remain
    remaining = find_ocr_manifests(PROCESSED_DATA_DIR, args.dataset)
    if remaining:
        print(f"\nWARNING: {len(remaining)} manifests still remaining (failed files)")
    else:
        print("\nAll OCR manifests processed successfully.")


if __name__ == "__main__":
    main()
