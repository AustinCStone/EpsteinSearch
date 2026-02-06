"""RAG query system using Gemini or GPT-5."""

import google.generativeai as genai
from openai import AzureOpenAI
from typing import Optional
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
