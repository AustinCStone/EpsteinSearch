"""OCR processing using Gemini 3 Flash vision capabilities with heavy parallelization."""

import google.generativeai as genai
from PIL import Image
from pathlib import Path
import fitz  # pymupdf
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import threading

from .config import GEMINI_API_KEY, OCR_ENABLED, OCR_MIN_TEXT_THRESHOLD, OCR_DPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Model for OCR
OCR_MODEL = "gemini-3-flash-preview"

# Parallelization settings for Tier 3 (4000+ RPM = ~67/sec)
OCR_PARALLEL_REQUESTS = 200  # Concurrent OCR requests
OCR_RETRY_ATTEMPTS = 3
OCR_RETRY_DELAY = 1.0

# Thread-local storage for models (one per thread)
_thread_local = threading.local()


def get_thread_model():
    """Get or create a Gemini model for the current thread."""
    if not hasattr(_thread_local, 'model'):
        _thread_local.model = genai.GenerativeModel(OCR_MODEL)
    return _thread_local.model


def needs_ocr(page: fitz.Page) -> bool:
    """
    Determine if a PDF page needs OCR processing.

    Returns True if:
    - Page has very little extracted text but contains images
    - Page has almost no text at all (likely scanned)
    """
    if not OCR_ENABLED:
        return False

    text = page.get_text().strip()
    images = page.get_images()
    text_len = len(text)

    # Page has images but minimal text - likely scanned
    if len(images) > 0 and text_len < OCR_MIN_TEXT_THRESHOLD:
        return True

    # Almost no text extracted at all
    if text_len < 50:
        return True

    return False


OCR_PROMPT = """Extract ALL text from this image. This may be a scanned document, handwritten notes, or typed text.

Instructions:
- Transcribe every word visible in the image
- Preserve the original layout and line breaks where possible
- For handwritten text, do your best to interpret cursive or messy writing
- If text is partially obscured or unclear, make your best guess and indicate uncertainty with [?]
- Do not add any commentary or descriptions - only output the extracted text
- If there is no text in the image, respond with: [NO TEXT FOUND]

Extracted text:"""


def ocr_image_single(image: Image.Image, retries: int = OCR_RETRY_ATTEMPTS) -> str:
    """
    Run OCR on a single PIL Image using thread-local Gemini model.

    Args:
        image: PIL Image
        retries: Number of retry attempts

    Returns:
        Extracted text as string
    """
    model = get_thread_model()

    for attempt in range(retries):
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            response = model.generate_content([OCR_PROMPT, image])
            text = response.text.strip()

            # Handle empty result
            if text == "[NO TEXT FOUND]":
                return ""

            return text
        except Exception as e:
            if attempt < retries - 1:
                wait_time = OCR_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"OCR attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"OCR failed after {retries} attempts: {e}")
                return ""


def ocr_images_batch(images: list[Image.Image], max_workers: int = OCR_PARALLEL_REQUESTS) -> list[str]:
    """
    Run OCR on multiple images in parallel.

    Args:
        images: List of PIL Images
        max_workers: Maximum concurrent requests

    Returns:
        List of extracted texts (same order as input)
    """
    if not images:
        return []

    results = [None] * len(images)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(ocr_image_single, img): i for i, img in enumerate(images)}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"OCR failed for image {idx}: {e}")
                results[idx] = ""

    return results


def ocr_pdf_pages_batch(pages: list[tuple[fitz.Page, int]], dpi: int = None) -> dict[int, str]:
    """
    Run OCR on multiple PDF pages in parallel.

    Args:
        pages: List of (page, page_number) tuples
        dpi: Resolution for rendering

    Returns:
        Dict mapping page_number to extracted text
    """
    dpi = dpi or OCR_DPI

    # Convert pages to images
    images = []
    page_nums = []

    for page, page_num in pages:
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
            page_nums.append(page_num)
        except Exception as e:
            logger.error(f"Failed to render page {page_num}: {e}")

    # Run OCR in parallel
    texts = ocr_images_batch(images)

    # Map results back to page numbers
    return {page_num: text for page_num, text in zip(page_nums, texts)}


def ocr_pdf_page(page: fitz.Page, dpi: int = None) -> str:
    """
    Convert PDF page to image and run OCR (single page, for backwards compatibility).

    Args:
        page: PyMuPDF page object
        dpi: Resolution for rendering (default from config)

    Returns:
        Extracted text from OCR
    """
    dpi = dpi or OCR_DPI

    try:
        # Render page to image
        pix = page.get_pixmap(dpi=dpi)

        # Convert to PIL Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Run OCR
        return ocr_image_single(img)
    except Exception as e:
        logger.error(f"Failed to OCR PDF page: {e}")
        return ""


def ocr_file(file_path: Path) -> str:
    """
    Run OCR on an image file (jpg, png, etc.).

    Args:
        file_path: Path to image file

    Returns:
        Extracted text from OCR
    """
    try:
        img = Image.open(file_path)
        return ocr_image_single(img)
    except Exception as e:
        logger.error(f"Failed to OCR file {file_path}: {e}")
        return ""


def ocr_files_batch(file_paths: list[Path], max_workers: int = OCR_PARALLEL_REQUESTS) -> list[str]:
    """
    Run OCR on multiple image files in parallel.

    Args:
        file_paths: List of paths to image files
        max_workers: Maximum concurrent requests

    Returns:
        List of extracted texts (same order as input)
    """
    images = []
    for path in file_paths:
        try:
            images.append(Image.open(path))
        except Exception as e:
            logger.error(f"Failed to open {path}: {e}")
            images.append(None)

    # Filter out None images and track indices
    valid_images = [(i, img) for i, img in enumerate(images) if img is not None]

    if not valid_images:
        return [""] * len(file_paths)

    # OCR valid images
    indices, imgs = zip(*valid_images)
    texts = ocr_images_batch(list(imgs), max_workers)

    # Map back to original order
    results = [""] * len(file_paths)
    for idx, text in zip(indices, texts):
        results[idx] = text

    return results


def get_page_ocr_stats(page: fitz.Page) -> dict:
    """
    Get statistics about a page to help decide if OCR is needed.

    Returns dict with:
        - text_chars: number of characters from text extraction
        - image_count: number of images on page
        - needs_ocr: boolean recommendation
    """
    text = page.get_text().strip()
    images = page.get_images()

    return {
        "text_chars": len(text),
        "image_count": len(images),
        "needs_ocr": needs_ocr(page)
    }
