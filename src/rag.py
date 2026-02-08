"""RAG query system using Gemini or GPT-5."""

import json as _json
import google.generativeai as genai
from openai import AzureOpenAI
from typing import Generator, Optional
import logging
import os

from .config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K_RESULTS
from .vectorstore import search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# LLM PROVIDER TOGGLE - Set to "gemini" or "gpt5"
# =============================================================================
LLM_PROVIDER = "gpt5"
# =============================================================================

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure Azure OpenAI GPT-5
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = "2025-01-01-preview"
GPT5_MODEL = "gpt-5-chat"

azure_client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)


SYSTEM_PROMPT = """You are an expert analyst of the Epstein court documents.
Your role is to answer questions based ONLY on the provided document excerpts.

Guidelines:
- Only use information from the provided context
- If the context doesn't contain relevant information, say so clearly
- Quote specific passages when making claims
- Be precise about names, dates, and facts
- Distinguish between allegations and proven facts
- Note that many people mentioned in these documents were not accused of wrongdoing
- Cite which document excerpt(s) you're drawing from

Context from court documents:
{context}
"""


def format_context(results: list[dict]) -> str:
    """Format search results into context for the prompt."""
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "Unknown")
        text = result.get("text", "")
        score = result.get("score", 0)
        context_parts.append(f"[Excerpt {i} - Source: {source} (relevance: {score:.2f})]\n{text}")
    return "\n\n---\n\n".join(context_parts)


def compose_prompt(query: str, context: str) -> str:
    """Compose the full prompt for Gemini."""
    return f"""{SYSTEM_PROMPT.format(context=context)}

User Question: {query}

Please provide a thorough answer based on the document excerpts above. Cite specific excerpts when making claims."""


def query_gemini(prompt: str, model_name: str = GEMINI_MODEL) -> str:
    """Send a prompt to Gemini and get a response."""
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise


def query_gpt5(prompt: str) -> str:
    """Send a prompt to Azure OpenAI GPT-5 and get a response."""
    try:
        response = azure_client.chat.completions.create(
            model=GPT5_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT-5 API error: {e}")
        raise


def query_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM provider."""
    if LLM_PROVIDER == "gpt5":
        logger.info("Using GPT-5")
        return query_gpt5(prompt)
    else:
        logger.info("Using Gemini")
        return query_gemini(prompt)


def ask(
    question: str,
    top_k: int = TOP_K_RESULTS,
    show_sources: bool = False
) -> dict:
    """
    Ask a question about the Epstein documents.

    Args:
        question: The question to ask
        top_k: Number of document chunks to retrieve
        show_sources: Include source excerpts in response

    Returns:
        Dict with 'answer', 'sources' (if show_sources), and 'query'
    """
    # Retrieve relevant chunks
    results = search(question, top_k)

    if not results:
        return {
            "answer": "No relevant documents found. Please ensure the vector store has been built.",
            "query": question,
            "sources": []
        }

    # Format context and compose prompt
    context = format_context(results)
    prompt = compose_prompt(question, context)

    # Query LLM
    answer = query_llm(prompt)

    response = {
        "answer": answer,
        "query": question,
    }

    if show_sources:
        response["sources"] = [
            {
                "text": r.get("text", "")[:500] + "..." if len(r.get("text", "")) > 500 else r.get("text", ""),
                "source": r.get("source", "Unknown"),
                "score": r.get("score", 0)
            }
            for r in results
        ]

    return response


def ask_with_context(question: str, context_texts: list[str]) -> str:
    """
    Ask a question with manually provided context.

    Useful for testing or when you want to bypass retrieval.
    """
    context = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{text}" for i, text in enumerate(context_texts))
    prompt = compose_prompt(question, context)
    return query_gemini(prompt)


# ---------------------------------------------------------------------------
# Multi-query RAG with streaming
# ---------------------------------------------------------------------------

QUERY_GEN_PROMPT = """You are a search query optimizer for a document retrieval system containing the Epstein court documents (depositions, emails, FBI reports, financial records, travel logs, etc.).

Given the user's question, generate 5-10 diverse search queries that would retrieve the most relevant document chunks. Each query should:
- Target different aspects or phrasings of the question
- Use specific names, dates, or terminology likely to appear in legal documents
- Be concise (under 20 words each)
- We are searching an embedding vector store which indexes based on semantic similarity, so prefer queries which will have similar semantics to the thing you are ultimately looking for.

Return ONLY a JSON array of strings, no other text.

User question: {question}"""


def generate_search_queries(question: str) -> list[str]:
    """Use the LLM to generate optimized search queries from a user question."""
    prompt = QUERY_GEN_PROMPT.format(question=question)
    try:
        raw = query_llm(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        queries = _json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:10]
    except Exception as e:
        logger.warning(f"Failed to parse search queries from LLM: {e}. Falling back to raw question.")
    return [question]


def merge_results(all_results: list[list[dict]], top_k: int) -> list[dict]:
    """Merge results from multiple queries, keeping best score per source."""
    best_by_source: dict[str, dict] = {}
    for results in all_results:
        for r in results:
            source = r.get("source", "")
            score = r.get("score", 0)
            if source not in best_by_source or score > best_by_source[source].get("score", 0):
                best_by_source[source] = r
    merged = sorted(best_by_source.values(), key=lambda r: r.get("score", 0), reverse=True)
    return merged[:top_k]


def ask_streaming(
    question: str,
    top_k: int = TOP_K_RESULTS,
) -> Generator[dict, None, None]:
    """Streaming RAG pipeline that yields SSE event dicts."""
    # Step 1: Generate search queries
    yield {"type": "status", "message": "Generating search queries..."}

    queries = generate_search_queries(question)
    yield {"type": "queries", "queries": queries}

    # Step 2: Execute searches
    yield {"type": "status", "message": f"Searching {len(queries)} queries..."}

    all_results = []
    for q in queries:
        results = search(q, top_k=top_k)
        all_results.append(results)

    # Step 3: Merge and deduplicate
    merged = merge_results(all_results, top_k=top_k)

    if not merged:
        yield {"type": "answer", "answer": "No relevant documents found."}
        yield {"type": "sources", "sources": []}
        return

    sources_for_client = [
        {
            "text": r.get("text", "")[:500] + ("..." if len(r.get("text", "")) > 500 else ""),
            "full_text": r.get("full_text", ""),
            "source": r.get("source", "Unknown"),
            "score": round(r.get("score", 0), 4),
        }
        for r in merged
    ]
    yield {"type": "sources", "sources": sources_for_client}

    # Step 4: Generate LLM answer
    yield {"type": "status", "message": "Generating answer..."}

    context = format_context(merged[:50])
    prompt = compose_prompt(question, context)

    try:
        answer = query_llm(prompt)
    except Exception as e:
        yield {"type": "error", "message": f"LLM error: {e}"}
        return

    yield {"type": "answer", "answer": answer}
