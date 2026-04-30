"""
Figure Grader — audits every monetary figure in the draft briefing
against the source chunks. Sets figures_grounded=True only if all
cited figures are traceable to retrieved documents.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from app.config import get_settings
from app.observability.langfuse_client import observe

# Pattern: $94.9, 94.9B, $1.53, 12%, etc.
_FIGURE_PAT = re.compile(r"\$[\d,]+(?:\.\d+)?(?:\s*[BMK](?:illion)?)?|\d+(?:\.\d+)?%", re.IGNORECASE)


@observe(name="figure_grader")
def figure_grader_node(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = state.get("settings") or get_settings()
    draft = state.get("draft_briefing", "")
    chunks = state.get("filtered_chunks") or state.get("retrieved_chunks", [])

    cited_figures = _extract_figures(draft)
    if not cited_figures:
        # Nothing to ground — pass by default
        return {**state, "figures_grounded": True}

    source_text = " ".join(c["text"] for c in chunks)
    ungrounded = _find_ungrounded(cited_figures, source_text)

    grounded = len(ungrounded) == 0

    return {
        **state,
        "figures_grounded": grounded,
        "ungrounded_figures": ungrounded,
    }


def _extract_figures(text: str) -> List[str]:
    return _FIGURE_PAT.findall(text)


def _find_ungrounded(figures: List[str], source_text: str) -> List[str]:
    """
    A figure is considered grounded if its numeric value appears in the source text.
    We normalise both sides (strip $, commas) before matching.
    """
    ungrounded = []
    for fig in figures:
        norm = _normalise(fig)
        if norm not in _normalise(source_text):
            ungrounded.append(fig)
    return ungrounded


def _normalise(text: str) -> str:
    return re.sub(r"[$,%,]", "", text).strip()
