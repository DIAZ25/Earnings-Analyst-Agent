"""
LangGraph multi-agent orchestration graph.

Nodes (in order):
  supervisor → financials_rag → relevance_grader → comparator
           → sentiment → figure_grader → [reflection loop] → final
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, END

from app.agents.supervisor import supervisor_node
from app.agents.financials_rag import financials_rag_node
from app.agents.relevance_grader import relevance_grader_node
from app.agents.comparator import comparator_node
from app.agents.sentiment import sentiment_node
from app.agents.figure_grader import figure_grader_node
from app.agents.reflection import reflection_node
from app.agents.final_node import final_node

MAX_REFLECTIONS = 3

# ── State schema (TypedDict-compatible plain dict) ────────────────────────────

State = Dict[str, Any]


def _should_reflect(state: State) -> str:
    """Edge: if figures are grounded or we've hit max loops → final, else reflect."""
    if state.get("figures_grounded", False):
        return "final"
    if state.get("reflection_count", 0) >= MAX_REFLECTIONS:
        return "final"
    return "reflect"


def build_graph() -> StateGraph:
    g = StateGraph(dict)

    # Register nodes
    g.add_node("supervisor", supervisor_node)
    g.add_node("financials_rag", financials_rag_node)
    g.add_node("relevance_grader", relevance_grader_node)
    g.add_node("comparator", comparator_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("figure_grader", figure_grader_node)
    g.add_node("reflection", reflection_node)
    g.add_node("final", final_node)

    # Edges
    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "financials_rag")
    g.add_edge("financials_rag", "relevance_grader")
    g.add_edge("relevance_grader", "comparator")
    g.add_edge("comparator", "sentiment")
    g.add_edge("sentiment", "figure_grader")

    # Conditional: grounded → final, else reflect
    g.add_conditional_edges(
        "figure_grader",
        _should_reflect,
        {"final": "final", "reflect": "reflection"},
    )
    # After reflection, loop back to financials_rag for another retrieval pass
    g.add_edge("reflection", "financials_rag")
    g.add_edge("final", END)

    return g.compile()
