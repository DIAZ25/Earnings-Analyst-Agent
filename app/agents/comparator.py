"""
Comparator Node — pure Python beat/miss/inline computation.

Deliberately LLM-FREE to prevent hallucinated verdicts.
Loads prior guidance from the JSON cache and extracts actuals
from the draft briefing with a regex pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import GuidanceComparison, BeatMissInline
from app.observability.langfuse_client import observe

# Regex to find monetary figures: e.g. "$94.9 billion", "94.9B", "EPS of $1.53"
_FIGURE_RE = re.compile(
    r"(?P<metric>[A-Za-z /]+?)[\s:]+\$?(?P<value>[\d,]+\.?\d*)\s*(?P<unit>[BMK](?:illion|n)?)?",
    re.IGNORECASE,
)

_SCALE = {"b": 1e9, "bn": 1e9, "billion": 1e9, "m": 1e6, "mn": 1e6, "million": 1e6, "k": 1e3}


@observe(name="comparator")
def comparator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from app.config import get_settings
    settings = state.get("settings") or get_settings()

    ticker = state["ticker"]
    draft = state.get("draft_briefing", "")

    guidance = _load_guidance(ticker, settings.guidance_cache_dir)
    comparisons = _compute_comparisons(guidance, draft)

    return {**state, "guidance_comparisons": [c.model_dump() for c in comparisons]}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_guidance(ticker: str, cache_dir: str) -> List[dict]:
    path = Path(cache_dir) / f"{ticker.upper()}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _compute_comparisons(
    guidance: List[dict], draft: str
) -> List[GuidanceComparison]:
    comparisons = []

    # Extract figures from draft
    actuals = _extract_figures(draft)

    for entry in guidance:
        metric = entry["metric"].lower()
        prior_val = float(entry["value"])

        # Find best matching actual in draft
        actual_val = _match_metric(metric, actuals)

        if actual_val is None:
            comparisons.append(GuidanceComparison(
                metric=entry["metric"],
                prior_guidance=prior_val,
                actual=None,
                beat_miss_inline=BeatMissInline.unknown,
                variance_pct=None,
            ))
            continue

        variance_pct = round((actual_val - prior_val) / prior_val * 100, 2) if prior_val else None

        if variance_pct is None:
            verdict = BeatMissInline.unknown
        elif variance_pct > 1.0:
            verdict = BeatMissInline.beat
        elif variance_pct < -1.0:
            verdict = BeatMissInline.miss
        else:
            verdict = BeatMissInline.inline

        comparisons.append(GuidanceComparison(
            metric=entry["metric"],
            prior_guidance=prior_val,
            actual=actual_val,
            beat_miss_inline=verdict,
            variance_pct=variance_pct,
        ))

    return comparisons


def _extract_figures(text: str) -> Dict[str, float]:
    """Return {normalised_metric: value} from the draft."""
    figures: Dict[str, float] = {}
    for m in _FIGURE_RE.finditer(text):
        metric = m.group("metric").strip().lower()
        raw_val = float(m.group("value").replace(",", ""))
        unit = (m.group("unit") or "").lower()
        scale = _SCALE.get(unit, 1)
        figures[metric] = raw_val * scale
    return figures


def _match_metric(metric: str, actuals: Dict[str, float]) -> Optional[float]:
    """Fuzzy keyword match between guidance metric and extracted actuals."""
    keywords = set(metric.lower().split())
    best_key = None
    best_overlap = 0
    for key in actuals:
        overlap = len(keywords & set(key.split()))
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key
    if best_overlap == 0:
        return None
    return actuals[best_key]
