"""Download DOJ 2026 Epstein document release."""

import requests
import sqlite3
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import logging

from .config import RAW_DATA_DIR, METADATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DOJ URL structure
DOJ_BASE = "https://www.justice.gov/epstein/files"

# Data set ranges (approximate, based on web scraping)
# Format: (dataset_number, start_file, end_file)
DATA_SETS = [
    (1, 1, 3158),
    (2, 3159, 6000),
    (3, 6001, 50000),
    (4, 50001, 100000),
    (5, 100001, 200000),
    (6, 200001, 400000),
    (7, 400001, 600000),
    (8, 600001, 900000),
    (9, 900001, 1200000),
    (10, 1200001, 1800000),
    (11, 1800001, 2400000),
    (12, 2400001, 2731488),
]

# Progress database
PROGRESS_DB = METADATA_DIR / "doj_download_progress.db"


def init_progress_db():
    """Initialize SQLite database for tracking download progress."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PROGRESS_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS downloads (
            file_id INTEGER PRIMARY KEY,
            dataset INTEGER,
            status TEXT,
            size INTEGER,
            error TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()


def get_pending_files(start: int = 1, end: int = 2731488) -> list[int]:
    """Get list of file IDs that haven't been downloaded yet."""
    conn = sqlite3.connect(PROGRESS_DB)
    c = conn.cursor()
    c.execute('SELECT file_id FROM downloads WHERE status = "success"')
    downloaded = set(row[0] for row in c.fetchall())
    conn.close()

    pending = [i for i in range(start, end + 1) if i not in downloaded]
    return pending


def mark_file_status(file_id: int, dataset: int, status: str, size: int = 0, error: str = ""):
    """Record file download status."""
    conn = sqlite3.connect(PROGRESS_DB)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO downloads (file_id, dataset, status, size, error, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (file_id, dataset, status, size, error, time.time()))
    conn.commit()
    conn.close()


def get_dataset_for_file(file_id: int) -> int:
    """Determine which dataset a file belongs to."""
    for ds_num, start, end in DATA_SETS:
        if start <= file_id <= end:
            return ds_num
    return 12  # Default to last dataset


def build_url(file_id: int) -> str:
    """Build the DOJ URL for a file."""
    dataset = get_dataset_for_file(file_id)
    filename = f"EFTA{file_id:08d}.pdf"
    return f"{DOJ_BASE}/DataSet%20{dataset}/{filename}"


def download_single_file(file_id: int, output_dir: Path, session: requests.Session) -> tuple[int, bool, str]:
    """Download a single file. Returns (file_id, success, error_message)."""
    dataset = get_dataset_for_file(file_id)
    filename = f"EFTA{file_id:08d}.pdf"
    url = build_url(file_id)
    output_path = output_dir / f"dataset_{dataset}" / filename

    # Skip if already exists
    if output_path.exists() and output_path.stat().st_size > 0:
        mark_file_status(file_id, dataset, "success", output_path.stat().st_size)
        return (file_id, True, "exists")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        response = session.get(url, timeout=30)

        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            size = len(response.content)
            mark_file_status(file_id, dataset, "success", size)
            return (file_id, True, "")
        elif response.status_code == 404:
            mark_file_status(file_id, dataset, "not_found", 0, "404")
            return (file_id, False, "404")
        else:
            mark_file_status(file_id, dataset, "error", 0, f"HTTP {response.status_code}")
            return (file_id, False, f"HTTP {response.status_code}")

    except Exception as e:
        mark_file_status(file_id, dataset, "error", 0, str(e))
        return (file_id, False, str(e))


def download_doj_files(
    output_dir: Optional[Path] = None,
    start: int = 1,
    end: int = 2731488,
    workers: int = 10,
    limit: Optional[int] = None
):
    """
    Download DOJ Epstein files with parallel workers.

    Args:
        output_dir: Where to save files (default: RAW_DATA_DIR/doj_2026)
        start: First file ID to download
        end: Last file ID to download
        workers: Number of parallel download workers
        limit: Max files to download (for testing)
    """
    output_dir = output_dir or (RAW_DATA_DIR / "doj_2026")
    output_dir.mkdir(parents=True, exist_ok=True)

    init_progress_db()

    # Get pending files
    pending = get_pending_files(start, end)
    if limit:
        pending = pending[:limit]

    logger.info(f"Files to download: {len(pending)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using {workers} parallel workers")

    if not pending:
        logger.info("No pending files to download!")
        return

    # Create session for connection pooling
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    success_count = 0
    error_count = 0
    not_found_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_single_file, fid, output_dir, session): fid
            for fid in pending
        }

        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            file_id, success, error = future.result()
            if success:
                success_count += 1
            elif error == "404":
                not_found_count += 1
            else:
                error_count += 1
                if error_count <= 10:
                    logger.warning(f"File {file_id} failed: {error}")

    logger.info(f"Download complete: {success_count} success, {not_found_count} not found, {error_count} errors")


def get_download_stats() -> dict:
    """Get download statistics from the progress database."""
    if not PROGRESS_DB.exists():
        return {"total": 0, "success": 0, "not_found": 0, "error": 0}

    conn = sqlite3.connect(PROGRESS_DB)
    c = conn.cursor()

    stats = {}
    c.execute('SELECT COUNT(*) FROM downloads')
    stats['total'] = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM downloads WHERE status = "success"')
    stats['success'] = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM downloads WHERE status = "not_found"')
    stats['not_found'] = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM downloads WHERE status = "error"')
    stats['error'] = c.fetchone()[0]

    c.execute('SELECT SUM(size) FROM downloads WHERE status = "success"')
    total_size = c.fetchone()[0] or 0
    stats['total_size_gb'] = total_size / (1024**3)

    conn.close()
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download DOJ Epstein files")
    parser.add_argument("--start", type=int, default=1, help="Start file ID")
    parser.add_argument("--end", type=int, default=2731488, help="End file ID")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--limit", type=int, help="Max files to download")
    parser.add_argument("--stats", action="store_true", help="Show download stats only")

    args = parser.parse_args()

    if args.stats:
        stats = get_download_stats()
        print(f"Download Statistics:")
        print(f"  Total tracked: {stats['total']:,}")
        print(f"  Successful: {stats['success']:,}")
        print(f"  Not found (404): {stats['not_found']:,}")
        print(f"  Errors: {stats['error']:,}")
        print(f"  Total size: {stats['total_size_gb']:.2f} GB")
    else:
        download_doj_files(
            start=args.start,
            end=args.end,
            workers=args.workers,
            limit=args.limit
        )
