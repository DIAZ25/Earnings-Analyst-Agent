"""
Financials RAG Agent — retrieves relevant chunks from both the filing
and transcript FAISS indexes, then drafts a financial summary via LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.config import get_settings
from app.ingestion.vector_store import VectorStore
from app.observability.langfuse_client import observe


@observe(name="financials_rag")
def financials_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = state.get("settings") or get_settings()
    ticker = state["ticker"]
    query = state["query"]
    embedder = state["embedder"]

    # ── Retrieve from both doc types ──────────────────────────────────────────
    all_chunks: List[dict] = []

    for doc_type in ("filing", "transcript"):
        store = VectorStore(
            ticker=ticker,
            doc_type=doc_type,
            embedder=embedder,
            store_dir=settings.vector_store_dir,
        )
        results = store.search(query, k=6)
        for meta, score in results:
            all_chunks.append({**meta, "score": score, "doc_type": doc_type})

    # Sort by relevance score descending
    all_chunks.sort(key=lambda x: x["score"], reverse=True)

    # ── Draft briefing via LLM ────────────────────────────────────────────────
    context = "\n\n---\n\n".join(c["text"] for c in all_chunks[:10])
    draft = _draft_briefing(query, ticker, context, settings)

    return {
        **state,
        "retrieved_chunks": all_chunks,
        "draft_briefing": draft,
    }


def _draft_briefing(query: str, ticker: str, context: str, settings) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    system = (
        "You are a senior equity research analyst. "
        "Using ONLY the provided source excerpts, draft a structured financial briefing. "
        "Include: key financial metrics, any mentions of guidance, and notable management commentary. "
        "Cite specific figures exactly as they appear in the sources. "
        "If the sources do not contain enough information to answer, say so explicitly."
    )
    human = f"Query: {query}\n\nTicker: {ticker}\n\nSource excerpts:\n\n{context}"

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return response.content
