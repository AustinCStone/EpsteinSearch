"""Download Epstein documents from DOJ and other sources."""

import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import logging

from .config import RAW_DATA_DIR, DOJ_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar and resume support."""
    try:
        # Check if file partially exists for resume
        existing_size = output_path.stat().st_size if output_path.exists() else 0

        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            logger.info(f"Resuming download from byte {existing_size}")

        response = requests.get(url, headers=headers, stream=True, timeout=30)

        # Handle resume response
        if response.status_code == 206:  # Partial content
            mode = "ab"
            total_size = existing_size + int(response.headers.get("content-length", 0))
        elif response.status_code == 200:
            mode = "wb"
            existing_size = 0
            total_size = int(response.headers.get("content-length", 0))
        else:
            logger.error(f"Failed to download {url}: {response.status_code}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, mode) as f:
            with tqdm(total=total_size, initial=existing_size, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def get_document_urls() -> list[dict]:
    """
    Get list of document URLs from DOJ website.

    NOTE: This needs to be implemented once the actual DOJ download
    structure is known. The January 2026 release format may be:
    - Multiple ZIP files
    - Individual PDF downloads
    - A bulk download archive

    Returns list of dicts with 'url' and 'filename' keys.
    """
    # TODO: Scrape or manually compile the list of document URLs
    # For now, return a placeholder
    logger.warning("Document URL list not yet implemented - need to investigate DOJ site structure")

    # Example structure:
    # return [
    #     {"url": "https://...", "filename": "batch_001.zip"},
    #     {"url": "https://...", "filename": "batch_002.zip"},
    # ]

    return []


def download_all_documents(
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None
) -> list[Path]:
    """
    Download all Epstein documents.

    Args:
        output_dir: Directory to save files (default: RAW_DATA_DIR)
        limit: Maximum number of files to download (for testing)

    Returns:
        List of paths to downloaded files
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = get_document_urls()
    if limit:
        urls = urls[:limit]

    downloaded = []
    for doc in tqdm(urls, desc="Downloading documents"):
        output_path = output_dir / doc["filename"]
        if output_path.exists():
            logger.info(f"Skipping existing file: {output_path}")
            downloaded.append(output_path)
            continue

        if download_file(doc["url"], output_path):
            downloaded.append(output_path)

    return downloaded


# Alternative: Download the 2024 court release (smaller, for testing)
GUARDIAN_PDF_URL = "https://uploads.guim.co.uk/2024/01/04/Final_Epstein_documents.pdf"


def download_2024_release(output_dir: Optional[Path] = None) -> Optional[Path]:
    """Download the 2024 court document release (943 pages) for testing."""
    output_dir = output_dir or RAW_DATA_DIR
    output_path = output_dir / "final_epstein_documents_2024.pdf"

    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        return output_path

    if download_file(GUARDIAN_PDF_URL, output_path):
        return output_path
    return None
