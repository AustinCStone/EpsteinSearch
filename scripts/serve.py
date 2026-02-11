#!/usr/bin/env python3
"""Web UI for Epstein Documents RAG system."""

import sys
sys.path.insert(0, "/storage/epstein_llm")

import asyncio
import json
import logging
import queue
import re
import secrets
import sqlite3
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import uvicorn

from src.vectorstore import search as vector_search, get_vector_store
from src.rag import ask, ask_streaming
from src.config import QUERY_LOG_DB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Epstein Documents RAG")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

STATS_KEY = "epstein-stats-2026"


RAW_DIR = Path("/storage/epstein_llm/data/raw/doj_2026")

# Full index: EFTA source stem -> Path on disk (built once at startup)
_pdf_index: dict[str, Path] = {}
_pdf_index_built = False


def _build_pdf_index():
    """Scan all PDFs under RAW_DIR once and index by stem."""
    global _pdf_index_built
    if _pdf_index_built:
        return
    logger.info("Building PDF path index...")
    t0 = time.time()
    count = 0
    for pdf in RAW_DIR.rglob("EFTA*.pdf"):
        stem = pdf.stem  # e.g. EFTA00081180
        if stem not in _pdf_index:
            _pdf_index[stem] = pdf
            count += 1
    _pdf_index_built = True
    logger.info(f"PDF index ready — {count:,} files in {time.time()-t0:.1f}s")


def _find_pdf(source: str) -> Path | None:
    """Find the on-disk PDF for a source like EFTA01262782."""
    if not _pdf_index_built:
        _build_pdf_index()
    return _pdf_index.get(source)


def _pdf_url(source: str) -> str | None:
    """Return the local API URL for a source's PDF, or None."""
    if not source.startswith("EFTA"):
        return None
    return f"/api/pdf/{source}"


class QueryRequest(BaseModel):
    query: str = Field(max_length=2000)
    top_k: int = Field(default=100, le=200)


class ShareRequest(BaseModel):
    query: str = Field(max_length=2000)
    answer: str = Field(max_length=50000)
    sources: list[dict] = Field(max_length=50)
    queries_generated: list[str] = Field(default=[], max_length=20)


# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------
def _init_query_log():
    """Create query log table if it doesn't exist."""
    QUERY_LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(QUERY_LOG_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            mode TEXT,
            query TEXT,
            num_queries_generated INTEGER,
            results_count INTEGER,
            response_time_ms REAL,
            ip_address TEXT,
            user_agent TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shared_results (
            id TEXT PRIMARY KEY,
            created_at REAL,
            query TEXT,
            answer TEXT,
            sources TEXT,
            queries_generated TEXT,
            view_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Query log ready at {QUERY_LOG_DB}")


def _log_query(
    mode: str, query: str, results_count: int, response_time_ms: float,
    ip_address: str = "", user_agent: str = "", num_queries_generated: int = 0,
):
    """Log a query to the database."""
    try:
        conn = sqlite3.connect(QUERY_LOG_DB)
        conn.execute(
            "INSERT INTO queries (timestamp, mode, query, num_queries_generated, results_count, response_time_ms, ip_address, user_agent) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), mode, query, num_queries_generated, results_count, response_time_ms, ip_address, user_agent),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")


# ---------------------------------------------------------------------------
# Load vector store on startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    logger.info("Loading vector store...")
    t0 = time.time()
    store = get_vector_store()
    logger.info(f"Vector store ready — {store.size:,} vectors loaded in {time.time()-t0:.1f}s")
    _build_pdf_index()
    _init_query_log()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/api/search")
@limiter.limit("60/minute")
async def api_search(req: QueryRequest, request: Request):
    """Raw vector search — no LLM, just retrieval."""
    t0 = time.time()
    results = await asyncio.to_thread(vector_search, req.query, req.top_k)
    _log_query(
        mode="search", query=req.query, results_count=len(results),
        response_time_ms=(time.time() - t0) * 1000,
        ip_address=request.client.host if request.client else "",
        user_agent=request.headers.get("user-agent", ""),
    )
    return {
        "query": req.query,
        "results": [
            {
                "source": r.get("source", "Unknown"),
                "score": round(r.get("score", 0), 4),
                "text": r.get("text", ""),
                "full_text": r.get("full_text", ""),
                "pdf_url": _pdf_url(r.get("source", "")),
            }
            for r in results
        ],
    }


@app.post("/api/ask")
@limiter.limit("10/minute")
async def api_ask(req: QueryRequest, request: Request):
    """Full RAG pipeline with SSE streaming — retrieval + LLM answer."""
    client_ip = request.client.host if request.client else ""
    user_agent = request.headers.get("user-agent", "")

    q: queue.Queue = queue.Queue()

    def run_pipeline():
        t0 = time.time()
        num_queries = 0
        results_count = 0
        try:
            for event in ask_streaming(req.query, top_k=req.top_k):
                if event.get("type") == "queries":
                    num_queries = len(event.get("queries", []))
                if event.get("type") == "sources":
                    results_count = len(event.get("sources", []))
                    for s in event.get("sources", []):
                        s["pdf_url"] = _pdf_url(s.get("source", ""))
                q.put(event)
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            _log_query(
                mode="ask", query=req.query, results_count=results_count,
                response_time_ms=(time.time() - t0) * 1000,
                ip_address=client_ip, user_agent=user_agent,
                num_queries_generated=num_queries,
            )
            q.put(None)  # sentinel

    asyncio.get_event_loop().run_in_executor(None, run_pipeline)

    async def event_stream():
        while True:
            event = await asyncio.to_thread(q.get)
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/pdf/{source}")
async def api_pdf(source: str):
    """Serve a raw PDF from disk by source ID (e.g. EFTA01262782)."""
    path = _find_pdf(source)
    if path is None:
        raise HTTPException(status_code=404, detail=f"PDF not found for {source}")
    return FileResponse(path, media_type="application/pdf", filename=f"{source}.pdf")


@app.get("/api/stats")
async def api_stats(key: str = ""):
    """Return query log statistics (requires ?key= param)."""
    if key != STATS_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    conn = sqlite3.connect(QUERY_LOG_DB)
    c = conn.cursor()
    today = time.time() - 86400
    stats = {}
    c.execute("SELECT COUNT(*) FROM queries")
    stats["total_queries"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM queries WHERE timestamp > ?", (today,))
    stats["queries_last_24h"] = c.fetchone()[0]
    c.execute("SELECT mode, COUNT(*) FROM queries GROUP BY mode")
    stats["by_mode"] = dict(c.fetchall())
    c.execute("SELECT AVG(response_time_ms) FROM queries")
    avg = c.fetchone()[0]
    stats["avg_response_ms"] = round(avg, 1) if avg else 0
    c.execute("SELECT query, COUNT(*) as cnt FROM queries GROUP BY query ORDER BY cnt DESC LIMIT 20")
    stats["top_queries"] = [{"query": r[0], "count": r[1]} for r in c.fetchall()]
    c.execute("SELECT query, mode, results_count, response_time_ms, ip_address, timestamp FROM queries ORDER BY id DESC LIMIT 20")
    stats["recent"] = [
        {"query": r[0], "mode": r[1], "results": r[2], "time_ms": round(r[3], 1), "ip": r[4], "timestamp": r[5]}
        for r in c.fetchall()
    ]
    conn.close()
    return stats


# ---------------------------------------------------------------------------
# Share endpoints
# ---------------------------------------------------------------------------
def _gen_share_id() -> str:
    return secrets.token_urlsafe(12)


@app.post("/api/share")
@limiter.limit("20/minute")
async def api_share(req: ShareRequest, request: Request):
    """Create a shareable link for a query result."""
    share_id = _gen_share_id()
    # Keep top 20 sources and strip full_text to keep payload small
    trimmed = [
        {k: v for k, v in s.items() if k != "full_text"}
        for s in req.sources[:20]
    ]
    conn = sqlite3.connect(QUERY_LOG_DB)
    conn.execute(
        "INSERT INTO shared_results (id, created_at, query, answer, sources, queries_generated, view_count) VALUES (?, ?, ?, ?, ?, ?, 0)",
        (share_id, time.time(), req.query, req.answer, json.dumps(trimmed), json.dumps(req.queries_generated)),
    )
    conn.commit()
    conn.close()
    return {"share_id": share_id, "share_url": f"/s/{share_id}"}


@app.get("/s/{share_id}", response_class=HTMLResponse)
async def share_page(share_id: str):
    """Render a shared result page."""
    conn = sqlite3.connect(QUERY_LOG_DB)
    conn.execute("UPDATE shared_results SET view_count = view_count + 1 WHERE id = ?", (share_id,))
    conn.commit()
    row = conn.execute(
        "SELECT query, answer, sources, queries_generated, created_at, view_count FROM shared_results WHERE id = ?",
        (share_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Shared result not found")
    query, answer, sources_json, queries_json, created_at, view_count = row
    sources = json.loads(sources_json)
    queries_generated = json.loads(queries_json)
    # Inject data into the share page template
    return SHARE_PAGE.replace("__SHARE_DATA__", json.dumps({
        "query": query,
        "answer": answer,
        "sources": sources,
        "queries_generated": queries_generated,
        "created_at": created_at,
        "view_count": view_count,
    }))


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
SHARE_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Epstein Documents RAG — Shared Result</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
<script data-goatcounter="https://partlyshady.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0a0a0a; --surface: #141414; --border: #262626; --border-hover: #404040;
  --text: #e0e0e0; --text-dim: #888; --accent: #c9a84c; --accent-dim: #8a6d2b;
}
body { background: var(--bg); color: var(--text); font-family: "SF Mono", "Cascadia Code", "Fira Code", "JetBrains Mono", monospace; font-size: 14px; line-height: 1.6; min-height: 100vh; }
.container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
header { text-align: center; margin-bottom: 40px; }
header h1 { font-size: 18px; font-weight: 600; letter-spacing: 0.05em; color: var(--accent); text-transform: uppercase; }
header p { color: var(--text-dim); font-size: 12px; margin-top: 6px; }
.query-display { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 16px 20px; margin-bottom: 16px; font-size: 15px; color: var(--text); }
.query-display .label { font-size: 11px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.queries-box { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 12px 16px; margin-bottom: 16px; }
.queries-box .label { font-size: 11px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.query-chip { display: inline-block; background: rgba(201,168,76,0.1); border: 1px solid var(--accent-dim); border-radius: 3px; padding: 4px 10px; margin: 3px 4px; font-size: 12px; color: var(--text); }
.answer-box { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 4px; padding: 20px 24px; margin-bottom: 24px; line-height: 1.7; }
.answer-box .label { font-size: 11px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }
.answer-content { white-space: normal; }
.answer-content h1, .answer-content h2, .answer-content h3 { color: var(--accent); margin: 16px 0 8px; font-size: 15px; }
.answer-content h1 { font-size: 17px; }
.answer-content p { margin-bottom: 10px; }
.answer-content ul, .answer-content ol { margin: 8px 0 8px 20px; }
.answer-content li { margin-bottom: 4px; }
.answer-content code { background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 3px; font-size: 13px; }
.answer-content blockquote { border-left: 3px solid var(--accent-dim); padding-left: 12px; color: var(--text-dim); margin: 10px 0; }
.answer-content strong { color: #fff; }
.answer-content a { color: var(--accent); }
.results-header { font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }
.source-card { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; margin-bottom: 10px; transition: border-color 0.2s; }
.source-card:hover { border-color: var(--border-hover); }
.source-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; cursor: pointer; user-select: none; }
.source-header .meta { display: flex; gap: 16px; align-items: center; }
.source-header .doc-id { font-weight: 600; font-size: 13px; color: var(--text); }
.source-header .score { font-size: 11px; color: var(--accent); background: rgba(201,168,76,0.1); padding: 2px 8px; border-radius: 3px; }
.pdf-link { font-size: 11px; color: var(--accent); text-decoration: none; padding: 2px 8px; border: 1px solid var(--accent-dim); border-radius: 3px; transition: all 0.2s; }
.pdf-link:hover { background: var(--accent-dim); color: #fff; }
.source-header .toggle { color: var(--text-dim); font-size: 12px; transition: transform 0.2s; }
.source-header .toggle.open { transform: rotate(90deg); }
.source-snippet { padding: 0 16px 12px; color: var(--text-dim); font-size: 12px; line-height: 1.5; }
.source-full { display: none; padding: 0 16px 16px; border-top: 1px solid var(--border); margin: 0 16px 16px; padding-top: 16px; font-size: 12px; line-height: 1.7; color: var(--text); white-space: pre-wrap; max-height: 500px; overflow-y: auto; }
.source-full.visible { display: block; }
.cta-link { display: inline-block; margin-top: 24px; color: var(--accent); text-decoration: none; font-size: 13px; border: 1px solid var(--accent-dim); padding: 8px 16px; border-radius: 4px; transition: all 0.2s; }
.cta-link:hover { background: var(--accent-dim); color: #fff; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Epstein Documents RAG</h1>
    <p>Shared result</p>
  </header>
  <div id="content"></div>
  <a class="cta-link" href="/">Ask your own question</a>
  <footer style="text-align:center; margin-top:48px; padding:16px 0; border-top:1px solid var(--border); font-size:11px; color:var(--text-dim);">
    Contact: <a href="mailto:findhiddensecrets@gmail.com" style="color:var(--accent);">findhiddensecrets@gmail.com</a>
  </footer>
</div>
<script>
const data = __SHARE_DATA__;
const container = document.getElementById("content");

let html = '<div class="query-display"><div class="label">Question</div>' + escapeHtml(data.query) + '</div>';

if (data.queries_generated && data.queries_generated.length > 0) {
  let chips = "";
  for (const q of data.queries_generated) chips += '<div class="query-chip">' + escapeHtml(q) + '</div>';
  html += '<div class="queries-box"><div class="label">Generated Search Queries</div>' + chips + '</div>';
}

html += '<div class="answer-box"><div class="label">LLM Answer</div><div class="answer-content">'
  + (typeof marked !== "undefined" ? DOMPurify.sanitize(marked.parse(data.answer || "")) : escapeHtml(data.answer || ""))
  + '</div></div>';

if (data.sources && data.sources.length > 0) {
  html += '<div class="results-header">' + data.sources.length + ' source documents</div>';
  data.sources.forEach(function(s, i) {
    const snippet = (s.text || "").substring(0, 300).replace(/\\n/g, " ");
    const fullText = s.full_text || s.text || "";
    const pdfLink = s.pdf_url
      ? '<a class="pdf-link" href="' + escapeHtml(s.pdf_url) + '" target="_blank" rel="noopener" onclick="event.stopPropagation()">View PDF</a>'
      : "";
    html += '<div class="source-card">'
      + '<div class="source-header" onclick="toggleCard(' + i + ')">'
      + '<div class="meta">'
      + '<span class="doc-id">' + escapeHtml(s.source || "Unknown") + '</span>'
      + '<span class="score">' + (s.score || 0).toFixed(4) + '</span>'
      + pdfLink
      + '</div>'
      + '<span class="toggle" id="toggle-' + i + '">&#9654;</span>'
      + '</div>'
      + '<div class="source-snippet">' + escapeHtml(snippet) + (snippet.length >= 300 ? "..." : "") + '</div>'
      + '<div class="source-full" id="full-' + i + '">' + escapeHtml(fullText) + '</div>'
      + '</div>';
  });
}

container.innerHTML = html;

function toggleCard(i) {
  document.getElementById("full-" + i).classList.toggle("visible");
  document.getElementById("toggle-" + i).classList.toggle("open");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
</script>
</body>
</html>
"""

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Epstein Documents RAG</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
<script data-goatcounter="https://partlyshady.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0a0a0a;
  --surface: #141414;
  --border: #262626;
  --border-hover: #404040;
  --text: #e0e0e0;
  --text-dim: #888;
  --accent: #c9a84c;
  --accent-dim: #8a6d2b;
  --red: #c0392b;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: "SF Mono", "Cascadia Code", "Fira Code", "JetBrains Mono", monospace;
  font-size: 14px;
  line-height: 1.6;
  min-height: 100vh;
}

.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 40px 20px;
}

header {
  text-align: center;
  margin-bottom: 40px;
}

header h1 {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--accent);
  text-transform: uppercase;
}

header p {
  color: var(--text-dim);
  font-size: 12px;
  margin-top: 6px;
}

/* Search area */
.search-box {
  display: flex;
  gap: 0;
  margin-bottom: 12px;
}

.search-box input {
  flex: 1;
  background: var(--surface);
  border: 1px solid var(--border);
  border-right: none;
  border-radius: 4px 0 0 4px;
  padding: 12px 16px;
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.search-box input:focus {
  border-color: var(--accent-dim);
}

.search-box input::placeholder {
  color: var(--text-dim);
}

.search-box button {
  background: var(--accent-dim);
  color: #fff;
  border: 1px solid var(--accent-dim);
  border-radius: 0 4px 4px 0;
  padding: 12px 24px;
  font-family: inherit;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
  white-space: nowrap;
}

.search-box button:hover { background: var(--accent); border-color: var(--accent); }
.search-box button:disabled { opacity: 0.5; cursor: wait; }

/* Mode toggle */
.mode-toggle {
  display: flex;
  gap: 8px;
  margin-bottom: 32px;
  font-size: 12px;
}

.mode-toggle label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  transition: all 0.2s;
  color: var(--text-dim);
}

.mode-toggle input { display: none; }
.mode-toggle label.active {
  border-color: var(--accent-dim);
  color: var(--accent);
}

/* Status */
.status {
  text-align: center;
  color: var(--text-dim);
  font-size: 12px;
  margin-bottom: 24px;
  min-height: 18px;
}

.status.error { color: var(--red); }

/* Generated queries display */
.queries-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 12px 16px;
  margin-bottom: 16px;
}

.queries-box .label {
  font-size: 11px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.query-chip {
  display: inline-block;
  background: rgba(201, 168, 76, 0.1);
  border: 1px solid var(--accent-dim);
  border-radius: 3px;
  padding: 4px 10px;
  margin: 3px 4px;
  font-size: 12px;
  color: var(--text);
}

/* Answer box */
.answer-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 4px;
  padding: 20px 24px;
  margin-bottom: 24px;
  line-height: 1.7;
}

.answer-box .label {
  font-size: 11px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 12px;
}

/* Markdown content styling */
.answer-content { white-space: normal; }
.answer-content h1, .answer-content h2, .answer-content h3 {
  color: var(--accent);
  margin: 16px 0 8px;
  font-size: 15px;
}
.answer-content h1 { font-size: 17px; }
.answer-content p { margin-bottom: 10px; }
.answer-content ul, .answer-content ol {
  margin: 8px 0 8px 20px;
}
.answer-content li { margin-bottom: 4px; }
.answer-content code {
  background: rgba(255,255,255,0.06);
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 13px;
}
.answer-content blockquote {
  border-left: 3px solid var(--accent-dim);
  padding-left: 12px;
  color: var(--text-dim);
  margin: 10px 0;
}
.answer-content strong { color: #fff; }
.answer-content a { color: var(--accent); }

/* Results */
.results-header {
  font-size: 12px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}

.source-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  margin-bottom: 10px;
  transition: border-color 0.2s;
}

.source-card:hover { border-color: var(--border-hover); }

.source-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
}

.source-header .meta {
  display: flex;
  gap: 16px;
  align-items: center;
}

.source-header .doc-id {
  font-weight: 600;
  font-size: 13px;
  color: var(--text);
}

.source-header .score {
  font-size: 11px;
  color: var(--accent);
  background: rgba(201, 168, 76, 0.1);
  padding: 2px 8px;
  border-radius: 3px;
}

.pdf-link {
  font-size: 11px;
  color: var(--accent);
  text-decoration: none;
  padding: 2px 8px;
  border: 1px solid var(--accent-dim);
  border-radius: 3px;
  transition: all 0.2s;
}

.pdf-link:hover {
  background: var(--accent-dim);
  color: #fff;
}

.source-header .toggle {
  color: var(--text-dim);
  font-size: 12px;
  transition: transform 0.2s;
}

.source-header .toggle.open { transform: rotate(90deg); }

.source-snippet {
  padding: 0 16px 12px;
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.5;
}

.source-full {
  display: none;
  padding: 0 16px 16px;
  border-top: 1px solid var(--border);
  margin: 0 16px 16px;
  padding-top: 16px;
  font-size: 12px;
  line-height: 1.7;
  color: var(--text);
  white-space: pre-wrap;
  max-height: 500px;
  overflow-y: auto;
}

.source-full.visible { display: block; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }

/* Loading animation */
@keyframes pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}
.loading { animation: pulse 1.5s ease-in-out infinite; }

/* Generating answer banner */
.generating-banner {
  background: var(--surface);
  border: 1px solid var(--accent-dim);
  border-left: 3px solid var(--accent);
  border-radius: 4px;
  padding: 20px 24px;
  margin-bottom: 24px;
  font-size: 15px;
  color: var(--accent);
  animation: pulse 1.5s ease-in-out infinite;
}

.generating-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: var(--accent);
  border-radius: 50%;
  margin-right: 4px;
  animation: pulse 1s ease-in-out infinite;
}

/* Share button */
.share-btn {
  background: transparent;
  color: var(--accent);
  border: 1px solid var(--accent-dim);
  border-radius: 4px;
  padding: 6px 14px;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
  margin-top: 12px;
}
.share-btn:hover { background: var(--accent-dim); color: #fff; }
.share-link {
  margin-top: 12px;
  font-size: 12px;
  color: var(--text-dim);
}
.share-link a { color: var(--accent); }
.share-link .copied {
  color: var(--accent);
  margin-left: 8px;
}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Epstein Documents RAG</h1>
    <p>Search 4.1M+ vectors across court documents, depositions, and evidence</p>
  </header>

  <div class="search-box">
    <input type="text" id="query" placeholder="Search documents or ask a question..." autofocus>
    <button id="submit" onclick="doQuery()">Ask</button>
  </div>

  <div class="mode-toggle">
    <label id="mode-search" onclick="setMode('search')">
      <input type="radio" name="mode" value="search">
      Search &mdash; fast vector lookup
    </label>
    <label id="mode-ask" class="active" onclick="setMode('ask')">
      <input type="radio" name="mode" value="ask" checked>
      Ask &mdash; RAG with LLM answer
    </label>
  </div>

  <div class="status" id="status"></div>
  <div id="queries-container"></div>
  <div id="answer-container"></div>
  <div id="results-container"></div>

  <footer style="text-align:center; margin-top:48px; padding:16px 0; border-top:1px solid var(--border); font-size:11px; color:var(--text-dim);">
    Contact: <a href="mailto:findhiddensecrets@gmail.com" style="color:var(--accent);">findhiddensecrets@gmail.com</a>
  </footer>
</div>

<script>
let mode = "ask";
let lastQuery = "";
let lastAnswer = "";
let lastSources = [];
let lastQueriesGenerated = [];

function setMode(m) {
  mode = m;
  document.getElementById("mode-search").classList.toggle("active", m === "search");
  document.getElementById("mode-ask").classList.toggle("active", m === "ask");
  document.getElementById("submit").textContent = m === "search" ? "Search" : "Ask";
}

document.getElementById("query").addEventListener("keydown", function(e) {
  if (e.key === "Enter") doQuery();
});

async function doQuery() {
  const query = document.getElementById("query").value.trim();
  if (!query) return;

  const btn = document.getElementById("submit");
  const status = document.getElementById("status");
  const answerC = document.getElementById("answer-container");
  const resultsC = document.getElementById("results-container");
  const queriesC = document.getElementById("queries-container");

  btn.disabled = true;
  answerC.innerHTML = "";
  resultsC.innerHTML = "";
  queriesC.innerHTML = "";
  status.className = "status loading";
  lastQuery = query;
  lastAnswer = "";
  lastSources = [];
  lastQueriesGenerated = [];

  const t0 = performance.now();

  if (mode === "search") {
    status.textContent = "Searching vectors...";
    try {
      const resp = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 100 }),
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      renderSources(data.results, elapsed, resultsC);
      status.className = "status";
      status.textContent = data.results.length > 0
        ? data.results.length + " results in " + elapsed + "s"
        : "No results found.";
    } catch (err) {
      status.className = "status error";
      status.textContent = "Error: " + err.message;
    } finally {
      btn.disabled = false;
    }
    return;
  }

  // Ask mode: SSE streaming
  status.textContent = "Starting RAG pipeline...";
  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 100 }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = reader.read ? await reader.read() : { done: true };
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch { continue; }

        switch (event.type) {
          case "status":
            status.textContent = event.message;
            break;

          case "queries":
            lastQueriesGenerated = event.queries;
            queriesC.innerHTML = renderQueries(event.queries);
            break;

          case "sources": {
            lastSources = event.sources;
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            renderSources(event.sources, elapsed, resultsC);
            status.className = "status";
            status.textContent = "";
            answerC.innerHTML = '<div class="generating-banner"><span class="generating-dot"></span> Generating answer &mdash; analyzing ' + Math.min(event.sources.length, 50) + ' sources...</div>';
            break;
          }

          case "token": {
            if (!answerC.dataset.buf) {
              answerC.dataset.buf = "";
              answerC.innerHTML = '<div class="answer-box"><div class="label">LLM Answer</div><div class="answer-content"></div></div>';
            }
            answerC.dataset.buf += event.token;
            lastAnswer = answerC.dataset.buf;
            const el = answerC.querySelector(".answer-content");
            if (el) {
              el.innerHTML = typeof marked !== "undefined" ? DOMPurify.sanitize(marked.parse(answerC.dataset.buf)) : escapeHtml(answerC.dataset.buf);
            }
            break;
          }

          case "done": {
            const elapsed2 = ((performance.now() - t0) / 1000).toFixed(1);
            status.className = "status";
            status.textContent = "Done in " + elapsed2 + "s";
            delete answerC.dataset.buf;
            if (lastAnswer) {
              const box = answerC.querySelector(".answer-box");
              if (box) {
                const shareDiv = document.createElement("div");
                shareDiv.innerHTML = '<button class="share-btn" onclick="doShare(this)">Share</button>';
                box.appendChild(shareDiv);
              }
            }
            break;
          }

          case "error":
            status.className = "status error";
            status.textContent = "Error: " + event.message;
            delete answerC.dataset.buf;
            break;
        }
      }
    }
  } catch (err) {
    status.className = "status error";
    status.textContent = "Error: " + err.message;
  } finally {
    btn.disabled = false;
  }
}

function renderQueries(queries) {
  let chips = "";
  for (const q of queries) chips += '<div class="query-chip">' + escapeHtml(q) + '</div>';
  return '<div class="queries-box"><div class="label">Generated Search Queries</div>' + chips + '</div>';
}

function renderSources(sources, elapsed, container) {
  if (!sources || sources.length === 0) { container.innerHTML = ""; return; }
  let html = '<div class="results-header">' + sources.length + ' source documents (' + elapsed + 's)</div>';
  sources.forEach(function(s, i) {
    const snippet = (s.text || "").substring(0, 300).replace(/\\n/g, " ");
    const fullText = s.full_text || s.text || "";
    const pdfLink = s.pdf_url
      ? '<a class="pdf-link" href="' + escapeHtml(s.pdf_url) + '" target="_blank" rel="noopener" onclick="event.stopPropagation()">View PDF</a>'
      : "";
    html += '<div class="source-card">'
      + '<div class="source-header" onclick="toggleCard(' + i + ')">'
      + '<div class="meta">'
      + '<span class="doc-id">' + escapeHtml(s.source || "Unknown") + '</span>'
      + '<span class="score">' + (s.score || 0).toFixed(4) + '</span>'
      + pdfLink
      + '</div>'
      + '<span class="toggle" id="toggle-' + i + '">&#9654;</span>'
      + '</div>'
      + '<div class="source-snippet">' + escapeHtml(snippet) + (snippet.length >= 300 ? "..." : "") + '</div>'
      + '<div class="source-full" id="full-' + i + '">' + escapeHtml(fullText) + '</div>'
      + '</div>';
  });
  container.innerHTML = html;
}

function toggleCard(i) {
  const el = document.getElementById("full-" + i);
  const arrow = document.getElementById("toggle-" + i);
  el.classList.toggle("visible");
  arrow.classList.toggle("open");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function doShare(btn) {
  btn.disabled = true;
  btn.textContent = "Sharing...";
  try {
    const resp = await fetch("/api/share", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: lastQuery,
        answer: lastAnswer,
        sources: lastSources,
        queries_generated: lastQueriesGenerated,
      }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    const url = location.origin + data.share_url;
    navigator.clipboard.writeText(url).catch(function(){});
    btn.parentElement.innerHTML = '<div class="share-link">Link: <a href="' + escapeHtml(data.share_url) + '" target="_blank">' + escapeHtml(url) + '</a><span class="copied">Copied!</span></div>';
  } catch (err) {
    btn.textContent = "Share failed";
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
