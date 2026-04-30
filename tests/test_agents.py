"""Tests for individual agent nodes."""

import pytest
from app.agents.supervisor import supervisor_node
from app.agents.comparator import _compute_comparisons, _extract_figures
from app.agents.figure_grader import _find_ungrounded, _extract_figures as fg_extract
from app.agents.sentiment import _count_matches, _detect_red_flags, _score_guidance_specificity
from app.models import BeatMissInline


class TestSupervisor:
    def _base_state(self, query):
        return {"query": query, "ticker": "AAPL", "trace_id": "test", "embedder": None}

    def test_comparison_query_classified(self):
        state = supervisor_node(self._base_state("Did Apple beat revenue guidance?"))
        assert state["query_type"] == "comparison"

    def test_sentiment_query_classified(self):
        state = supervisor_node(self._base_state("What was management's tone?"))
        assert state["query_type"] == "sentiment"

    def test_general_query_classified(self):
        state = supervisor_node(self._base_state("Tell me about Apple's business"))
        assert state["query_type"] == "general"

    def test_state_fields_initialised(self):
        state = supervisor_node(self._base_state("test"))
        for field in ["retrieved_chunks", "filtered_chunks", "draft_briefing",
                      "final_briefing", "sentiment", "guidance_comparisons"]:
            assert field in state


class TestComparator:
    def test_beat_verdict(self):
        guidance = [{"metric": "Revenue", "value": 90.0}]
        draft = "Revenue was $94.9 billion"
        comparisons = _compute_comparisons(guidance, draft)
        assert len(comparisons) == 1
        assert comparisons[0].beat_miss_inline == BeatMissInline.beat

    def test_miss_verdict(self):
        guidance = [{"metric": "Revenue", "value": 100.0}]
        draft = "Revenue was $94.9 billion"
        comparisons = _compute_comparisons(guidance, draft)
        assert comparisons[0].beat_miss_inline == BeatMissInline.miss

    def test_inline_verdict(self):
        guidance = [{"metric": "Revenue", "value": 95.0}]
        draft = "Revenue was $94.9 billion"
        comparisons = _compute_comparisons(guidance, draft)
        assert comparisons[0].beat_miss_inline == BeatMissInline.inline

    def test_unknown_when_no_actual(self):
        guidance = [{"metric": "Revenue", "value": 90.0}]
        draft = "The company reported strong results."
        comparisons = _compute_comparisons(guidance, draft)
        assert comparisons[0].beat_miss_inline == BeatMissInline.unknown

    def test_extract_figures_finds_billions(self):
        text = "Net revenue was $94.9 billion and EPS was $1.53"
        figures = _extract_figures(text)
        # Should find at least the revenue figure
        assert any("94" in k or "94" in str(v) for k, v in figures.items())


class TestFigureGrader:
    def test_grounded_figure_passes(self):
        figures = ["$94.9B", "$1.53"]
        source = "Net revenue was $94.9 billion. EPS was $1.53."
        ungrounded = _find_ungrounded(figures, source)
        assert ungrounded == []

    def test_hallucinated_figure_caught(self):
        figures = ["$999B"]
        source = "Net revenue was $94.9 billion."
        ungrounded = _find_ungrounded(figures, source)
        assert len(ungrounded) > 0


class TestSentiment:
    def test_hedging_count(self):
        text = "We expect headwinds and believe results may be uncertain."
        count = _count_matches(text, ["expect", "headwinds", "uncertain", "may", "believe"])
        assert count >= 4

    def test_red_flag_detection(self):
        text = "We face significant headwinds and softness in the market."
        flags = _detect_red_flags(text)
        assert "headwinds" in flags or "softness" in flags

    def test_guidance_specificity_quantified(self):
        text = "We expect revenue of $95 billion next quarter."
        spec = _score_guidance_specificity(text)
        assert spec == "quantified"

    def test_guidance_specificity_qualitative(self):
        text = "We expect continued growth and positive outlook."
        spec = _score_guidance_specificity(text)
        assert spec == "qualitative"

    def test_guidance_specificity_absent(self):
        text = "The team worked hard this quarter."
        spec = _score_guidance_specificity(text)
        assert spec == "absent"
