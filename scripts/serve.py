#!/usr/bin/env python3
"""Web UI for Epstein Documents RAG system."""

import sys
sys.path.insert(0, "/storage/epstein_llm")

import logging
import re
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from src.vectorstore import search as vector_search, get_vector_store
from src.rag import ask
from src.download_doj import DATA_SETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Epstein Documents RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


# ---------------------------------------------------------------------------
# Load vector store on startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    logger.info("Loading vector store...")
    t0 = time.time()
    store = get_vector_store()
    logger.info(f"Vector store ready — {store.size:,} vectors loaded in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/api/search")
async def api_search(req: QueryRequest):
    """Raw vector search — no LLM, just retrieval."""
    results = vector_search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {
                "source": r.get("source", "Unknown"),
                "score": round(r.get("score", 0), 4),
                "text": r.get("text", ""),
                "full_text": r.get("full_text", ""),
            }
            for r in results
        ],
    }


@app.post("/api/ask")
async def api_ask(req: QueryRequest):
    """Full RAG pipeline — retrieval + LLM answer."""
    result = ask(req.query, top_k=req.top_k, show_sources=True)
    return result


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Epstein Documents RAG</title>
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

/* Answer box */
.answer-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 4px;
  padding: 20px 24px;
  margin-bottom: 24px;
  white-space: pre-wrap;
  line-height: 1.7;
}

.answer-box .label {
  font-size: 11px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 12px;
}

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
    <button id="submit" onclick="doQuery()">Search</button>
  </div>

  <div class="mode-toggle">
    <label id="mode-search" class="active" onclick="setMode('search')">
      <input type="radio" name="mode" value="search" checked>
      Search &mdash; fast vector lookup
    </label>
    <label id="mode-ask" onclick="setMode('ask')">
      <input type="radio" name="mode" value="ask">
      Ask &mdash; RAG with LLM answer
    </label>
  </div>

  <div class="status" id="status"></div>
  <div id="answer-container"></div>
  <div id="results-container"></div>
</div>

<script>
let mode = "search";

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

  btn.disabled = true;
  answerC.innerHTML = "";
  resultsC.innerHTML = "";
  status.className = "status loading";
  status.textContent = mode === "search" ? "Searching vectors..." : "Querying LLM — this may take a moment...";

  const endpoint = mode === "search" ? "/api/search" : "/api/ask";
  const t0 = performance.now();

  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 10 }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    // LLM answer (ask mode)
    if (mode === "ask" && data.answer) {
      answerC.innerHTML = `<div class="answer-box"><div class="label">LLM Answer</div>${escapeHtml(data.answer)}</div>`;
    }

    // Source documents
    const sources = mode === "search" ? data.results : (data.sources || []);
    if (sources.length > 0) {
      let html = `<div class="results-header">${sources.length} source documents (${elapsed}s)</div>`;
      sources.forEach((s, i) => {
        const snippet = (s.text || "").substring(0, 300).replace(/\\n/g, " ");
        const fullText = s.full_text || s.text || "";
        html += `
          <div class="source-card">
            <div class="source-header" onclick="toggleCard(${i})">
              <div class="meta">
                <span class="doc-id">${escapeHtml(s.source || "Unknown")}</span>
                <span class="score">${(s.score || 0).toFixed(4)}</span>
              </div>
              <span class="toggle" id="toggle-${i}">&#9654;</span>
            </div>
            <div class="source-snippet">${escapeHtml(snippet)}${snippet.length >= 300 ? "..." : ""}</div>
            <div class="source-full" id="full-${i}">${escapeHtml(fullText)}</div>
          </div>`;
      });
      resultsC.innerHTML = html;
    }

    status.className = "status";
    status.textContent = sources.length > 0
      ? `${sources.length} results in ${elapsed}s`
      : "No results found.";

  } catch (err) {
    status.className = "status error";
    status.textContent = "Error: " + err.message;
  } finally {
    btn.disabled = false;
  }
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
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
