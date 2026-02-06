"""Configuration settings for the Epstein Documents RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths - adjust BASE_DIR for atlas deployment
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Database for tracking progress
PROGRESS_DB = METADATA_DIR / "progress.db"

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Document sources
DOJ_BASE_URL = "https://www.justice.gov/epstein"  # Placeholder - actual URLs TBD

# PDF Processing
BATCH_SIZE_EXTRACT = 500  # PDFs per batch for extraction

# Chunking settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Embedding settings
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_BATCH_SIZE = 1024
EMBEDDING_DEVICE = "cuda"  # or "cpu"

# FAISS settings
FAISS_INDEX_TYPE = "flat"  # or "ivf" for larger datasets

# Retrieval settings
TOP_K_RESULTS = 10

# Gemini settings
GEMINI_MODEL = "gemini-3-flash-preview"

# OCR Settings
OCR_ENABLED = True
OCR_MIN_TEXT_THRESHOLD = 100  # chars per page to trigger OCR
OCR_DPI = 200  # Resolution for PDF-to-image conversion
OCR_CONFIDENCE_THRESHOLD = 0.5  # Min confidence for OCR results

# Per-dataset OCR mode:
#   "all"       - OCR every page (photos, scans, handwritten)
#   "none"      - no OCR (digital text, emails, transcripts)
#   "heuristic" - OCR pages with < OCR_MIN_TEXT_THRESHOLD chars
DATASET_OCR_MODE = {
    "dataset_1": "all",            # Photos, evidence cards
    "dataset_2": "all",            # Photos
    "dataset_3": "all",            # Scanned folder covers
    "dataset_4": "all",            # FBI docs + scans
    "dataset_5": "all",            # Evidence box photos
    "dataset_6": "none",           # Grand jury transcripts
    "dataset_7": "none",           # Deposition transcripts
    "dataset_8": "none",           # Emails, itineraries
    "dataset_9": "heuristic",      # Emails (mostly), some scans
    "dataset_9_extract": "heuristic",
    "dataset_10": "heuristic",     # Bank docs, emails, scans
    "dataset_10_extract": "heuristic",
    "dataset_11": "none",          # Emails
    "dataset_12": "all",           # Memos + notebooks
}
