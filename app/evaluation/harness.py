"""
Evaluation harness — runs labelled ground-truth examples through the pipeline
and reports accuracy metrics.

Usage:
    python -m app.evaluation.harness --dataset data/eval/ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table

console = Console()


def run_harness(dataset_path: str) -> dict:
    """
    Load a JSON dataset of ground-truth examples and evaluate.

    Expected dataset format:
    [
      {
        "query": "Did Apple beat revenue guidance?",
        "ticker": "AAPL",
        "expected_verdict": "beat",   // beat | miss | inline | unknown
        "expected_metric": "Revenue"
      },
      ...
    ]
    """
    from app.agents.graph import build_graph
    from app.ingestion.embedder import get_embedder
    from app.config import get_settings

    settings = get_settings()
    embedder = get_embedder()
    graph = build_graph()

    with open(dataset_path) as f:
        examples = json.load(f)

    results = []
    correct = 0

    for i, ex in enumerate(examples):
        console.print(f"[cyan]Running example {i + 1}/{len(examples)}:[/] {ex['query']}")

        state = {
            "query": ex["query"],
            "ticker": ex["ticker"].upper(),
            "trace_id": f"eval-{i}",
            "embedder": embedder,
            "retrieved_chunks": [],
            "filtered_chunks": [],
            "draft_briefing": "",
            "final_briefing": "",
            "sentiment": None,
            "guidance_comparisons": [],
            "figures_grounded": False,
            "reflection_count": 0,
            "settings": settings,
        }

        try:
            result = graph.invoke(state)
        except Exception as e:
            console.print(f"[red]  ERROR: {e}[/]")
            results.append({"example": ex, "passed": False, "error": str(e)})
            continue

        # Check verdict accuracy
        comparisons = result.get("guidance_comparisons", [])
        verdict_match = False
        for comp in comparisons:
            if comp.get("metric", "").lower() == ex.get("expected_metric", "").lower():
                if comp.get("beat_miss_inline") == ex.get("expected_verdict"):
                    verdict_match = True
                break

        if verdict_match:
            correct += 1
            console.print(f"  [green]✓ PASS[/]")
        else:
            console.print(f"  [red]✗ FAIL[/] — expected {ex.get('expected_verdict')}")

        results.append({
            "example": ex,
            "passed": verdict_match,
            "comparisons": comparisons,
        })

    accuracy = correct / len(examples) if examples else 0

    # Summary table
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total examples", str(len(examples)))
    table.add_row("Correct", str(correct))
    table.add_row("Accuracy", f"{accuracy:.1%}")
    console.print(table)

    return {"accuracy": accuracy, "correct": correct, "total": len(examples), "results": results}


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--dataset", required=True, help="Path to ground-truth JSON file")
    parser.add_argument("--output", default="", help="Optional path to write results JSON")
    args = parser.parse_args()

    metrics = run_harness(args.dataset)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        console.print(f"[cyan]Results written to {args.output}[/]")

    sys.exit(0 if metrics["accuracy"] >= 0.7 else 1)


if __name__ == "__main__":
    main()
