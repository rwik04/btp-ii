"""
Results summary: aggregate neutrality rates, FAIR, C-FAIR tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import numpy as np


def load_results(results_dir: Path) -> List[dict]:
    """
    Load all result JSON files from a directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of result dicts
    """
    results = []
    for json_path in results_dir.glob("results_*.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            results.extend(json.load(f))
    return results


def compute_neutrality_rate(results: List[dict], system: str) -> float:
    """Compute neutral output rate for a system."""
    sys_results = [r for r in results if r.get("system") == system]
    if not sys_results:
        return 0.0
    neutral = sum(1 for r in sys_results if r.get("judge_category") == "Neutral")
    return neutral / len(sys_results)


def summarize_results(results_dir: Path) -> dict:
    """
    Compute summary statistics from experiment results.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dict with summary statistics
    """
    results = load_results(results_dir)
    if not results:
        return {"error": "No results found"}

    systems = sorted(set(r.get("system") for r in results))

    summary = {}
    for system in systems:
        sys_results = [r for r in results if r.get("system") == system]
        neutral_count = sum(1 for r in sys_results if r.get("judge_category") == "Neutral")
        total_count = len(sys_results)
        neutral_rate = neutral_count / total_count if total_count else 0

        cfair_scores = [r.get("cfair_score", 0) for r in sys_results]
        avg_cfair = np.mean(cfair_scores) if cfair_scores else 0

        fair_scores = [r.get("fair_score") for r in sys_results if r.get("fair_score") is not None]
        avg_fair = np.mean(fair_scores) if fair_scores else None

        lean_vars = [r.get("lean_var", 0) for r in sys_results]
        avg_lean_var = np.mean(lean_vars) if lean_vars else 0

        summary[system] = {
            "neutral_rate": f"{neutral_rate:.3f}",
            "neutral_count": f"{neutral_count}/{total_count}",
            "avg_cfair": f"{avg_cfair:.4f}",
            "avg_fair": f"{avg_fair:.4f}" if avg_fair is not None else "N/A",
            "avg_lean_var": f"{avg_lean_var:.4f}",
        }

    return summary


def print_summary(summary: dict) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print(f"{'System':<20} {'Neutral':>12} {'C-FAIR':>10} {'FAIR':>10} {'Lean Var':>10}")
    print("=" * 80)

    for system, stats in summary.items():
        neutral = stats["neutral_count"]
        cfair = stats["avg_cfair"]
        fair = stats["avg_fair"]
        lean_var = stats["avg_lean_var"]
        print(f"{system:<20} {neutral:>12} {cfair:>10} {fair:>10} {lean_var:>10}")

    print("=" * 80)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing results JSON files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    summary = summarize_results(results_dir)

    if "error" in summary:
        print(summary["error"])
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
