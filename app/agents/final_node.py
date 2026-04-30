"""
Final Node — assembles the clean markdown briefing and structured JSON output.
No LLM call.
"""

from __future__ import annotations

from typing import Any, Dict


def final_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ticker = state["ticker"]
    query = state["query"]
    draft = state.get("draft_briefing", "")
    comparisons = state.get("guidance_comparisons", [])
    sentiment = state.get("sentiment", {})
    grounded = state.get("figures_grounded", False)
    reflection_count = state.get("reflection_count", 0)

    briefing = _format_briefing(ticker, query, draft, comparisons, sentiment, grounded)

    return {
        **state,
        "final_briefing": briefing,
    }


def _format_briefing(
    ticker: str,
    query: str,
    draft: str,
    comparisons: list,
    sentiment: dict,
    grounded: bool,
) -> str:
    lines = [
        f"# Earnings Analyst Briefing — {ticker}",
        f"\n**Query:** {query}\n",
        "---",
        "\n## Financial Summary\n",
        draft,
    ]

    if comparisons:
        lines.append("\n## Guidance vs Actuals\n")
        lines.append("| Metric | Prior Guidance | Actual | Result | Variance |")
        lines.append("|--------|---------------|--------|--------|----------|")
        for c in comparisons:
            prior = f"{c['prior_guidance']:.2f}" if c.get("prior_guidance") is not None else "—"
            actual = f"{c['actual']:.2f}" if c.get("actual") is not None else "—"
            variance = f"{c['variance_pct']:+.1f}%" if c.get("variance_pct") is not None else "—"
            verdict = (c.get("beat_miss_inline") or "unknown").upper()
            lines.append(f"| {c['metric']} | {prior} | {actual} | {verdict} | {variance} |")

    if sentiment:
        lines.append("\n## Management Sentiment\n")
        ci = sentiment.get("confidence_index", 0)
        lines.append(f"- **Confidence Index:** {ci:.2f} / 1.00")
        lines.append(f"- **Guidance Specificity:** {sentiment.get('guidance_specificity', '—')}")
        flags = sentiment.get("red_flags", [])
        if flags:
            lines.append(f"- **Red Flags:** {', '.join(flags)}")
        lines.append(f"\n{sentiment.get('summary', '')}")

    lines.append("\n---")
    grounded_label = "✅ All figures verified against source documents" if grounded else "⚠️ Some figures could not be verified"
    lines.append(f"*{grounded_label}*")

    return "\n".join(lines)
