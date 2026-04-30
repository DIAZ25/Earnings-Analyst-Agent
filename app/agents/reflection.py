"""
Reflection Agent — critiques the draft when figures are ungrounded,
then produces a revised query to trigger a second retrieval pass.
"""

from __future__ import annotations

from typing import Any, Dict

from app.config import get_settings
from app.observability.langfuse_client import observe


@observe(name="reflection")
def reflection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = state.get("settings") or get_settings()

    draft = state.get("draft_briefing", "")
    ungrounded = state.get("ungrounded_figures", [])
    reflection_count = state.get("reflection_count", 0)

    # Produce a revised query that specifically asks for the missing figures
    revised_query = _build_revised_query(
        original_query=state["query"],
        draft=draft,
        ungrounded=ungrounded,
        settings=settings,
    )

    return {
        **state,
        "query": revised_query,           # updated query for next RAG pass
        "draft_briefing": "",             # clear draft so next pass starts fresh
        "reflection_count": reflection_count + 1,
    }


def _build_revised_query(
    original_query: str, draft: str, ungrounded: list, settings
) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    system = (
        "You are a research analyst reviewing an earnings briefing draft. "
        "The following figures could not be verified in the source documents: "
        f"{ungrounded}. "
        "Rewrite the original query to retrieve more specific source passages "
        "that contain these figures. Return ONLY the revised query string, nothing else."
    )
    human = f"Original query: {original_query}\n\nDraft excerpt:\n{draft[:800]}"

    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return resp.content.strip()
