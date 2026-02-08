#!/usr/bin/env python3
"""Phase 2: OCR flagged pages using Gemini Flash vision API.

Finds all _ocr_pages.json manifests left by Phase 1 (process_batch.py),
renders flagged pages to images, sends to Gemini for OCR, merges text
with pymupdf extraction, re-chunks, and overwrites output files.

Deletes _ocr_pages.json on success → built-in resume support.

Architecture: Process manifests in small batches. For each batch:
  1. Render pages and submit OCR tasks (streaming, not all at once)
  2. Collect results
  3. Reassemble text, re-chunk, save, delete manifests
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


def ocr_page_from_pdf(pdf_path: str, page_idx: int) -> str:
    """Render a single PDF page and OCR it. Opens/closes PDF each time to avoid holding memory."""
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()

    return ocr_image_single(img)


def process_batch_of_manifests(manifest_batch: list[Path], max_workers: int) -> tuple[int, int, int]:
    """
    Process a batch of manifests with page-level OCR parallelism.

    Returns: (processed_count, ocr_pages_count, failed_count)
    """
    import fitz
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Parse all manifests
    items = []
    for manifest_path in manifest_batch:
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            source_pdf = Path(manifest["source_pdf"])
            if not source_pdf.exists():
                logger.error(f"Source PDF not found: {source_pdf}")
                items.append(None)
                continue
            items.append({
                "manifest_path": manifest_path,
                "stem": manifest_path.stem.replace("_ocr_pages", ""),
                "parent_dir": manifest_path.parent,
                "source_pdf": str(source_pdf),
                "ocr_page_indices": manifest["pages"],
                "total_pages": manifest["total_pages"],
                "ocr_results": {},
            })
        except Exception as e:
            logger.error(f"Failed to read manifest {manifest_path}: {e}")
            items.append(None)

    # Submit all page OCR tasks to thread pool
    # Each task opens its own PDF handle, renders one page, OCRs it, closes — no memory accumulation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        for item_idx, item in enumerate(items):
            if item is None:
                continue
            for page_idx in item["ocr_page_indices"]:
                future = executor.submit(ocr_page_from_pdf, item["source_pdf"], page_idx)
                future_to_info[future] = (item_idx, page_idx)

        for future in as_completed(future_to_info):
            item_idx, page_idx = future_to_info[future]
            try:
                ocr_text = future.result()
                items[item_idx]["ocr_results"][page_idx] = ocr_text
            except Exception as e:
                logger.error(f"OCR failed for item {item_idx} page {page_idx}: {e}")

    # Reassemble text, re-chunk, save
    processed = 0
    total_ocr = 0
    failed = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for item in items:
        if item is None:
            failed += 1
            continue
        try:
            doc = fitz.open(item["source_pdf"])
            ocr_set = set(item["ocr_page_indices"])
            pages_text = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pymupdf_text = page.get_text().strip()

                if page_idx in ocr_set and page_idx in item["ocr_results"]:
                    ocr_text = item["ocr_results"][page_idx]
                    if len(ocr_text) > len(pymupdf_text):
                        pages_text.append(ocr_text)
                    else:
                        pages_text.append(pymupdf_text)
                else:
                    pages_text.append(pymupdf_text)

            doc.close()

            text = '\n\n'.join(pages_text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            text = text.strip()

            chunk_texts = splitter.split_text(text)
            chunks = [
                {"text": c, "source": item["stem"], "chunk_index": i}
                for i, c in enumerate(chunk_texts)
            ]

            text_path = item["parent_dir"] / f"{item['stem']}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            chunks_path = item["parent_dir"] / f"{item['stem']}_chunks.json"
            with open(chunks_path, "w") as f:
                json.dump(chunks, f)

            item["manifest_path"].unlink()
            processed += 1
            total_ocr += len(item["ocr_page_indices"])

        except Exception as e:
            logger.error(f"Failed to save {item['stem']}: {e}")
            failed += 1

    return processed, total_ocr, failed


def main():
    parser = argparse.ArgumentParser(description="Phase 2: OCR flagged pages via Gemini")
    parser.add_argument("--workers", "-w", type=int, default=200,
                        help="Concurrent OCR threads (default: 200, Gemini Tier 3 = 4000 RPM)")
    parser.add_argument("--dataset", "-d", nargs="+",
                        help="Process specific dataset(s)")
    parser.add_argument("--limit", "-l", type=int,
                        help="Limit number of manifests to process")
    parser.add_argument("--batch-size", "-b", type=int, default=200,
                        help="Manifests per batch (default: 200)")
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
        total_pages = 0
        for m in manifests:
            with open(m) as f:
                data = json.load(f)
                total_pages += len(data["pages"])

    total_manifests = len(manifests)
    logger.info(f"Processing {total_manifests} manifests ({total_pages} OCR pages) "
                f"with {args.workers} threads, batch size {args.batch_size}")

    processed = 0
    total_ocr_done = 0
    failed = 0
    start_time = time.time()

    for batch_start in range(0, total_manifests, args.batch_size):
        batch = manifests[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (total_manifests + args.batch_size - 1) // args.batch_size
        logger.info(f"=== Batch {batch_num}/{total_batches}: {len(batch)} manifests ===")

        bp, bo, bf = process_batch_of_manifests(batch, args.workers)
        processed += bp
        total_ocr_done += bo
        failed += bf

        elapsed = time.time() - start_time
        pages_rate = total_ocr_done / elapsed if elapsed > 0 else 0
        remaining_pages = total_pages - total_ocr_done
        remaining_min = remaining_pages / pages_rate / 60 if pages_rate > 0 else 0
        logger.info(
            f"Progress: {processed + failed}/{total_manifests} files | "
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
