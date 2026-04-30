"""
Relevance Grader — filters retrieved chunks to only those that are
topically relevant to the query. Uses a fast LLM yes/no classification.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.config import get_settings
from app.observability.langfuse_client import observe


@observe(name="relevance_grader")
def relevance_grader_node(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = state.get("settings") or get_settings()
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {**state, "filtered_chunks": []}

    filtered = _grade_chunks(query, chunks, settings)

    # Fallback: if grader removes everything, keep top-3 by score
    if not filtered:
        filtered = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:3]

    return {**state, "filtered_chunks": filtered}


def _grade_chunks(query: str, chunks: List[dict], settings) -> List[dict]:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    system = (
        "You are a relevance classifier. "
        "Given a query and a document excerpt, respond with exactly one word: "
        "'yes' if the excerpt is relevant to the query, 'no' if it is not."
    )

    relevant = []
    for chunk in chunks:
        human = f"Query: {query}\n\nExcerpt:\n{chunk['text'][:800]}"
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        if "yes" in resp.content.lower():
            relevant.append(chunk)

    return relevant
