"""
Supervisor node — routes query type and initialises shared state.
No LLM call: pure routing logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict

# Keywords that suggest a pure-sentiment query (skip comparator heavy lifting)
_SENTIMENT_KEYWORDS = re.compile(
    r"\b(tone|sentiment|management|confident|cautious|language|wording)\b",
    re.IGNORECASE,
)
_COMPARISON_KEYWORDS = re.compile(
    r"\b(beat|miss|guidance|revenue|eps|earnings|quarter|vs\.?|versus)\b",
    re.IGNORECASE,
)


def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify query type and initialise pipeline fields.

    Adds to state:
        query_type : 'comparison' | 'sentiment' | 'general'
    """
    query = state.get("query", "")

    if _COMPARISON_KEYWORDS.search(query):
        query_type = "comparison"
    elif _SENTIMENT_KEYWORDS.search(query):
        query_type = "sentiment"
    else:
        query_type = "general"

    return {
        **state,
        "query_type": query_type,
        "retrieved_chunks": [],
        "filtered_chunks": [],
        "draft_briefing": "",
        "final_briefing": "",
        "sentiment": None,
        "guidance_comparisons": [],
        "figures_grounded": False,
        "reflection_count": state.get("reflection_count", 0),
    }
