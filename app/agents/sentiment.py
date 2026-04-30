"""
Sentiment Agent — scores executive language from the transcript chunks.

Outputs:
  confidence_index       : float [0, 1]
  hedging_count          : int
  assertive_count        : int
  guidance_specificity   : 'quantified' | 'qualitative' | 'absent'
  red_flags              : List[str]
  tone_delta             : float (vs prior call, 0.0 if no prior)
  summary                : str
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app.config import get_settings
from app.observability.langfuse_client import observe

_HEDGE_WORDS = [
    "approximately", "roughly", "around", "expect", "anticipate", "believe",
    "may", "might", "could", "should", "uncertain", "subject to", "headwinds",
    "challenges", "difficult", "softness", "volatile", "unclear",
]
_ASSERTIVE_WORDS = [
    "confident", "strong", "accelerating", "record", "exceeded", "delivered",
    "outperform", "leadership", "robust", "momentum", "committed", "will",
    "achieve", "growth", "ahead of",
]
_RED_FLAG_PATTERNS = [
    r"headwinds?", r"softness", r"below\s+expectations?", r"miss(?:ed)?",
    r"challeng(?:es?|ing)", r"uncertain(?:ty)?", r"impact(?:ed)?",
    r"restat(?:e|ed|ement)", r"investigat(?:ion|ing)", r"legal proceedings?",
    r"going concern", r"material weakness", r"impairment",
]


@observe(name="sentiment")
def sentiment_node(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = state.get("settings") or get_settings()

    # Use transcript chunks preferentially
    chunks = state.get("filtered_chunks") or state.get("retrieved_chunks", [])
    transcript_chunks = [c for c in chunks if c.get("doc_type") == "transcript"]
    text_pool = " ".join(c["text"] for c in (transcript_chunks or chunks)[:15])

    # Rule-based counts (fast)
    hedging_count = _count_matches(text_pool, _HEDGE_WORDS)
    assertive_count = _count_matches(text_pool, _ASSERTIVE_WORDS)
    red_flags = _detect_red_flags(text_pool)

    total = hedging_count + assertive_count
    confidence_index = round(assertive_count / total, 3) if total else 0.5

    guidance_specificity = _score_guidance_specificity(text_pool)

    # LLM summary
    sentiment_summary, tone_delta = _llm_sentiment_summary(
        text_pool[:3000], state["ticker"], settings
    )

    sentiment_result = {
        "confidence_index": confidence_index,
        "hedging_count": hedging_count,
        "assertive_count": assertive_count,
        "guidance_specificity": guidance_specificity,
        "red_flags": red_flags,
        "tone_delta": tone_delta,
        "summary": sentiment_summary,
    }

    return {**state, "sentiment": sentiment_result}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_matches(text: str, words: List[str]) -> int:
    count = 0
    for w in words:
        count += len(re.findall(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE))
    return count


def _detect_red_flags(text: str) -> List[str]:
    found = []
    for pat in _RED_FLAG_PATTERNS:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            found.append(matches[0].lower())
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def _score_guidance_specificity(text: str) -> str:
    quantified_re = re.compile(r"\$[\d,]+|\d+%|\d+\s*(?:billion|million)", re.IGNORECASE)
    qualitative_re = re.compile(
        r"\b(expect|anticipate|project|forecast|outlook|guidance)\b", re.IGNORECASE
    )
    if quantified_re.search(text):
        return "quantified"
    if qualitative_re.search(text):
        return "qualitative"
    return "absent"


def _llm_sentiment_summary(text: str, ticker: str, settings) -> tuple[str, float]:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    system = (
        "You are a linguistic analyst specialising in corporate earnings calls. "
        "Analyse management tone and respond ONLY in JSON with keys: "
        "'summary' (2-3 sentence narrative) and "
        "'tone_delta' (float between -1 and 1, where positive = more bullish than typical). "
        "Do not include any text outside the JSON object."
    )
    human = f"Ticker: {ticker}\n\nTranscript excerpt:\n{text}"

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        data = json.loads(resp.content.strip())
        return data.get("summary", ""), float(data.get("tone_delta", 0.0))
    except Exception:
        return "Unable to generate sentiment summary.", 0.0
