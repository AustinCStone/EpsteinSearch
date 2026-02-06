# Epstein Documents RAG System

A Retrieval-Augmented Generation (RAG) system for querying the DOJ Epstein document release.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PDFs/Images                                                           │
│       │                                                                 │
│       ▼                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │   PyMuPDF   │───▶│  OCR Check  │───▶│   Gemini    │                │
│   │   Extract   │    │  needs_ocr()│    │   3 Flash   │                │
│   └─────────────┘    └─────────────┘    │   (Vision)  │                │
│                             │           └─────────────┘                │
│                             ▼                   │                       │
│                      ┌─────────────┐           │                       │
│                      │  Raw Text   │◀──────────┘                       │
│                      └─────────────┘                                   │
│                             │                                          │
│                             ▼                                          │
│                      ┌─────────────┐                                   │
│                      │   Chunker   │  1200 chars, 200 overlap          │
│                      └─────────────┘                                   │
│                             │                                          │
│                             ▼                                          │
│                      ┌─────────────┐                                   │
│                      │   Gemini    │  768-dim embeddings               │
│                      │  Embedding  │  100 texts/batch                  │
│                      │     API     │  3 parallel requests              │
│                      └─────────────┘                                   │
│                             │                                          │
│                             ▼                                          │
│                      ┌─────────────┐                                   │
│                      │    FAISS    │  IndexFlatIP                      │
│                      │   Index     │  (cosine similarity)              │
│                      └─────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           QUERY PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Query                                                            │
│       │                                                                 │
│       ▼                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │   Gemini    │───▶│    FAISS    │───▶│   Top-K     │                │
│   │  Embedding  │    │   Search    │    │   Chunks    │                │
│   └─────────────┘    └─────────────┘    └─────────────┘                │
│                                                │                        │
│                                                ▼                        │
│                                         ┌─────────────┐                │
│                                         │   GPT-5     │                │
│                                         │  (Azure)    │                │
│                                         └─────────────┘                │
│                                                │                        │
│                                                ▼                        │
│                                         ┌─────────────┐                │
│                                         │   Answer    │                │
│                                         │  + Sources  │                │
│                                         └─────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Preprocessing Pipeline

### 1. PDF Text Extraction (`src/extract.py`)

Uses PyMuPDF (`fitz`) to extract text from PDFs:

```python
doc = fitz.open(pdf_path)
for page in doc:
    text = page.get_text()
```

### 2. OCR Detection (`src/ocr.py`)

Automatically detects pages that need OCR:

```python
def needs_ocr(page) -> bool:
    text = page.get_text().strip()
    images = page.get_images()

    # Trigger OCR if:
    # 1. Has images but minimal text (< 100 chars) - likely scanned
    # 2. Almost no text at all (< 50 chars)
    if len(images) > 0 and len(text) < 100:
        return True
    if len(text) < 50:
        return True
    return False
```

### 3. OCR Processing (Gemini 3 Flash Vision)

For scanned pages and handwritten content:

```python
# Render PDF page to image
pix = page.get_pixmap(dpi=200)
img = Image.open(io.BytesIO(pix.tobytes("png")))

# Send to Gemini 3 Flash for OCR
response = model.generate_content([OCR_PROMPT, img])
text = response.text
```

**Why Gemini 3 Flash?**
- Handles cursive handwriting (traditional OCR cannot)
- Works on scanned documents
- Supports multiple languages
- $0.15/1M output tokens (cheap)

### 4. Text Chunking (`src/chunk.py`)

Uses LangChain's RecursiveCharacterTextSplitter:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)
```

### 5. Embedding (`src/embeddings.py`)

Uses Gemini Embedding API with parallelization:

```python
# Batch embedding (100 texts per API call)
result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=texts,
    task_type="retrieval_document",
    output_dimensionality=768
)

# Parallel processing (3 concurrent requests)
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(batch_embed, batch) for batch in batches]
```

**Throughput:** ~265 chunks/second

### 6. Vector Storage (`src/vectorstore.py`)

FAISS IndexFlatIP for cosine similarity search:

```python
index = faiss.IndexFlatIP(768)  # 768-dimensional embeddings
index.add(normalized_embeddings)
```

## Query Pipeline

### 1. Query Embedding

Same Gemini model, different task type:

```python
result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=query,
    task_type="retrieval_query"  # Different from retrieval_document
)
```

### 2. Vector Search

```python
scores, indices = index.search(query_embedding, top_k=10)
```

### 3. Answer Generation (GPT-5)

```python
# Context from retrieved chunks
context = format_context(top_k_chunks)

# Generate answer with Azure OpenAI GPT-5
response = azure_client.chat.completions.create(
    model="gpt-5-chat",
    messages=[{"role": "user", "content": prompt}]
)
```

**Why GPT-5?**
- Gemini has content filters that block sensitive document content
- GPT-5 handles the legal/court document context appropriately

## Usage

### Setup

```bash
cd /storage/epstein_llm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Add: GEMINI_API_KEY=your_key
```

### Process Documents

```bash
# Extract and process PDFs
python scripts/process_batch.py --input data/raw/doj_2026/

# Build vector index
python scripts/build_index.py
```

### Query

```bash
# Interactive CLI
python -m src.cli

# Search only (no LLM)
python scripts/search_only.py "flight logs passengers" 10
```

## File Structure

```
/storage/epstein_llm/
├── data/
│   ├── raw/                    # Original PDFs
│   │   └── doj_2026/           # DOJ 2026 release
│   ├── processed/              # Extracted text + chunks
│   └── metadata/               # Progress tracking DB
├── vector_store/               # FAISS index + metadata
├── src/
│   ├── config.py               # Settings
│   ├── extract.py              # PDF text extraction
│   ├── ocr.py                  # Gemini OCR
│   ├── chunk.py                # Text chunking
│   ├── embeddings.py           # Gemini Embedding API
│   ├── vectorstore.py          # FAISS operations
│   └── rag.py                  # Query + GPT-5
└── scripts/
    ├── process_batch.py        # Batch processing
    ├── build_index.py          # Build FAISS index
    └── search_only.py          # Debug search
```

## Cost Estimates

| Component | Model | Cost |
|-----------|-------|------|
| OCR | Gemini 3 Flash | ~$0.15/1M output tokens |
| Embeddings | Gemini Embedding | ~$0.15/1M tokens |
| RAG Answers | GPT-5 | ~$5/1M tokens |

For 5M chunks (~1B tokens):
- Embedding: ~$150-175
- OCR: ~$10-15 (only for scanned pages)
- Queries: Variable, ~$0.01 per query

## Performance

| Step | Rate |
|------|------|
| Text extraction | ~100 pages/sec |
| OCR (Gemini) | ~3 pages/sec |
| Chunking | ~10K chunks/sec |
| Embedding | ~265 chunks/sec |
| Search | <100ms |
