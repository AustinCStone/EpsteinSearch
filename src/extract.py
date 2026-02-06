"""Extract text from PDF documents with parallel OCR support."""

import re
import sqlite3
import json
from pathlib import Path
from typing import Optional, Generator
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

import pymupdf4llm
import fitz  # pymupdf

from .config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROGRESS_DB,
    BATCH_SIZE_EXTRACT,
    OCR_ENABLED
)
from .ocr import needs_ocr, ocr_pdf_pages_batch, ocr_file, ocr_files_batch

# Supported image extensions for standalone image processing
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe database lock
_db_lock = threading.Lock()


def init_progress_db():
    """Initialize SQLite database for tracking extraction progress."""
    PROGRESS_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(PROGRESS_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extraction_progress (
            file_path TEXT PRIMARY KEY,
            status TEXT,
            pages INTEGER,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def mark_file_processed(file_path: Path, status: str, pages: int = 0, error: str = None):
    """Mark a file as processed in the progress database (thread-safe)."""
    with _db_lock:
        conn = sqlite3.connect(PROGRESS_DB)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO extraction_progress (file_path, status, pages, error)
            VALUES (?, ?, ?, ?)
        """, (str(file_path), status, pages, error))
        conn.commit()
        conn.close()

    # Also update in-memory cache
    if status == "completed" and _processed_cache is not None:
        with _processed_cache_lock:
            _processed_cache.add(file_path.stem)


# Cache of processed file stems for fast lookup
_processed_cache: set = None
_processed_cache_lock = threading.Lock()


def _load_processed_cache() -> set:
    """Load all processed file stems into memory for fast lookup."""
    global _processed_cache
    with _processed_cache_lock:
        if _processed_cache is None:
            logger.info("Loading processed files cache...")
            _processed_cache = set()
            # Scan all chunk files recursively
            for chunk_file in PROCESSED_DATA_DIR.glob("**/*_chunks.json"):
                # Extract original file stem (remove _chunks suffix)
                stem = chunk_file.stem.replace("_chunks", "")
                _processed_cache.add(stem)
            logger.info(f"Loaded {len(_processed_cache)} processed files into cache")
        return _processed_cache


def is_file_processed(file_path: Path) -> bool:
    """Check if a file has already been processed (using in-memory cache)."""
    cache = _load_processed_cache()
    return file_path.stem in cache


def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove common page headers/footers
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'HIGHLY CONFIDENTIAL', '', text)

    # Clean up any remaining artifacts
    text = text.strip()

    return text


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int, dict]:
    """
    Extract text from a PDF file, using parallel OCR for scanned pages.

    Returns:
        Tuple of (extracted_text, page_count, ocr_metadata)
    """
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        all_text = [""] * page_count
        pages_needing_ocr = []

        # First pass: extract text and identify OCR pages
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()

            if OCR_ENABLED and needs_ocr(page):
                pages_needing_ocr.append((page, page_num))
                all_text[page_num] = ""  # Will be filled by OCR
            else:
                all_text[page_num] = text

        # Batch OCR all pages that need it
        ocr_pages = []
        if pages_needing_ocr:
            logger.info(f"  Running OCR on {len(pages_needing_ocr)} pages in parallel...")
            ocr_results = ocr_pdf_pages_batch(pages_needing_ocr)

            for page_num, ocr_text in ocr_results.items():
                if ocr_text:
                    all_text[page_num] = ocr_text
                    ocr_pages.append(page_num + 1)

        doc.close()

        # Combine and clean
        combined_text = '\n\n'.join(all_text)
        cleaned = clean_text(combined_text)

        ocr_metadata = {
            "ocr_pages": ocr_pages,
            "ocr_page_count": len(ocr_pages),
            "ocr_ratio": len(ocr_pages) / page_count if page_count > 0 else 0
        }

        return cleaned, page_count, ocr_metadata

    except Exception as e:
        logger.error(f"Error extracting {pdf_path}: {e}")
        raise


def extract_text_from_image(image_path: Path) -> tuple[str, dict]:
    """
    Extract text from a standalone image file using OCR.

    Returns:
        Tuple of (extracted_text, metadata)
    """
    try:
        logger.info(f"Running OCR on image: {image_path}")
        text = ocr_file(image_path)
        cleaned = clean_text(text)

        metadata = {
            "type": "image",
            "ocr": True,
            "source_file": str(image_path)
        }

        return cleaned, metadata

    except Exception as e:
        logger.error(f"Error extracting {image_path}: {e}")
        raise


def get_dataset_name(file_path: Path) -> str:
    """Extract dataset name from file path for organizing output."""
    path_str = str(file_path)

    # Look for dataset_XX pattern
    for part in file_path.parts:
        if part.startswith('dataset_'):
            return part

    # Look for doj_2026 pattern
    if 'doj_2026' in path_str and 'dataset' not in path_str:
        return 'doj_2026_other'

    # Default for other files (like 2024 release)
    return 'other'


def save_extracted_text(pdf_path: Path, text: str, metadata: dict):
    """Save extracted text and metadata to processed directory, organized by dataset."""
    # Organize by dataset
    dataset_name = get_dataset_name(pdf_path)
    output_dir = PROCESSED_DATA_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename based on input PDF
    base_name = pdf_path.stem
    text_path = output_dir / f"{base_name}.txt"
    meta_path = output_dir / f"{base_name}.json"

    # Save text
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return text_path


def get_pdf_files(input_dir: Optional[Path] = None) -> list[Path]:
    """Get all PDF files from input directory."""
    input_dir = input_dir or RAW_DATA_DIR
    return sorted(input_dir.glob("**/*.pdf"))


def get_document_files(input_dir: Optional[Path] = None, include_images: bool = True) -> list[Path]:
    """
    Get all document files (PDFs and optionally images) from input directory.

    Args:
        input_dir: Directory to search
        include_images: Whether to include image files

    Returns:
        Sorted list of file paths
    """
    input_dir = input_dir or RAW_DATA_DIR
    files = list(input_dir.glob("**/*.pdf"))

    if include_images:
        for ext in IMAGE_EXTENSIONS:
            files.extend(input_dir.glob(f"**/*{ext}"))

    return sorted(files)


def process_single_file(file_path: Path) -> tuple[Path, str, dict, int]:
    """
    Process a single file (PDF or image).

    Returns:
        Tuple of (file_path, text, metadata, page_count)
    """
    if file_path.suffix.lower() == '.pdf':
        text, page_count, ocr_metadata = extract_text_from_pdf(file_path)
        metadata = {
            "source_file": str(file_path),
            "type": "pdf",
            "pages": page_count,
            "characters": len(text),
            **ocr_metadata
        }
    elif file_path.suffix.lower() in IMAGE_EXTENSIONS:
        text, metadata = extract_text_from_image(file_path)
        metadata["characters"] = len(text)
        page_count = 1
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return file_path, text, metadata, page_count


def extract_batch(
    input_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    skip_processed: bool = True,
    include_images: bool = True
) -> Generator[tuple[Path, str], None, None]:
    """
    Extract text from PDFs and images in batches.

    Args:
        input_dir: Directory containing documents
        limit: Maximum number of files to process
        skip_processed: Skip already processed files
        include_images: Include image files for OCR processing

    Yields:
        Tuples of (file_path, extracted_text)
    """
    init_progress_db()

    doc_files = get_document_files(input_dir, include_images=include_images)
    if limit:
        doc_files = doc_files[:limit]

    for file_path in doc_files:
        if skip_processed and is_file_processed(file_path):
            logger.info(f"Skipping already processed: {file_path}")
            continue

        try:
            logger.info(f"Extracting: {file_path}")
            file_path, text, metadata, page_count = process_single_file(file_path)

            save_extracted_text(file_path, text, metadata)
            mark_file_processed(file_path, "completed", page_count)

            yield file_path, text

        except Exception as e:
            logger.error(f"Failed to extract {file_path}: {e}")
            mark_file_processed(file_path, "failed", error=str(e))
            continue


def extract_batch_parallel(
    input_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    skip_processed: bool = True,
    include_images: bool = True,
    max_workers: int = 4
) -> Generator[tuple[Path, str], None, None]:
    """
    Extract text from PDFs and images with file-level parallelization.

    Uses ThreadPoolExecutor for parallel file processing.
    Each file's OCR is already parallelized internally.

    Args:
        input_dir: Directory containing documents
        limit: Maximum number of files to process
        skip_processed: Skip already processed files
        include_images: Include image files for OCR processing
        max_workers: Number of parallel file processors

    Yields:
        Tuples of (file_path, extracted_text)
    """
    init_progress_db()

    doc_files = get_document_files(input_dir, include_images=include_images)

    # Filter already processed
    if skip_processed:
        doc_files = [f for f in doc_files if not is_file_processed(f)]

    if limit:
        doc_files = doc_files[:limit]

    logger.info(f"Processing {len(doc_files)} files with {max_workers} workers...")

    # Process in batches to avoid creating too many futures at once
    # (300k+ futures causes memory exhaustion)
    BATCH_SIZE = 1000

    def process_and_save(file_path: Path) -> tuple[Path, str, bool]:
        """Process file and save results."""
        try:
            logger.info(f"Processing: {file_path}")
            file_path, text, metadata, page_count = process_single_file(file_path)
            save_extracted_text(file_path, text, metadata)
            mark_file_processed(file_path, "completed", page_count)
            return file_path, text, True
        except Exception as e:
            logger.error(f"Failed: {file_path}: {e}")
            mark_file_processed(file_path, "failed", error=str(e))
            return file_path, "", False

    for batch_start in range(0, len(doc_files), BATCH_SIZE):
        batch = doc_files[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(doc_files) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Batch {batch_num}/{total_batches}: submitting {len(batch)} files...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_and_save, f): f for f in batch}

            for future in as_completed(futures):
                file_path, text, success = future.result()
                if success:
                    yield file_path, text


def get_all_processed_texts() -> Generator[tuple[Path, str, dict], None, None]:
    """
    Load all processed texts.

    Yields:
        Tuples of (text_path, text_content, metadata)
    """
    for text_path in PROCESSED_DATA_DIR.glob("*.txt"):
        meta_path = text_path.with_suffix(".json")

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        yield text_path, text, metadata
