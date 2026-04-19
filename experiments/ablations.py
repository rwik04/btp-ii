"""
Ablation runner for LFGD.

Expands a parameter grid and runs run_eval.py logic for each config cell.
Unsupported dimensions are still tracked in the output summary so the
experiment table remains complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from data import load_twinviews_huggingface
from experiments.run_eval import run_experiment, write_results


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def expand_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config.get("grid", {})
    names = list(grid.keys())
    values = [_ensure_list(grid[n]) for n in names]

    base = config.get("base", {})
    runs: list[dict[str, Any]] = []
    for combo in product(*values):
        params = dict(zip(names, combo))
        run_cfg = deepcopy(base)
        run_cfg.setdefault("retrieval", {})
        run_cfg.setdefault("lfgd", {})
        run_cfg.setdefault("llm", {})
        run_cfg["retrieval"]["N"] = params.get("N", run_cfg["retrieval"].get("N", 20))
        run_cfg["retrieval"]["embedding_model"] = params.get(
            "embedding_model",
            run_cfg["retrieval"].get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        run_cfg["lfgd"]["alpha"] = params.get("alpha", run_cfg["lfgd"].get("alpha", 0.5))
        run_cfg["lfgd"]["tau"] = params.get("tau", run_cfg["lfgd"].get("tau", 0.05))
        runs.append({"config": run_cfg, "params": params})
    return runs


def write_ablation_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"ablation_summary_{timestamp}.csv"
    fieldnames = [
        "run_id",
        "alpha",
        "tau",
        "N",
        "embedding_model",
        "axis_source",
        "fairness_loss",
        "system",
        "neutral_count",
        "total",
        "neutral_rate",
        "avg_cfair",
        "avg_fair",
        "results_path",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LFGD ablation grid")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/ablation_grid.yaml",
        help="Path to ablation grid yaml",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=None,
        help="Limit number of topics (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.n_topics is not None:
        cfg["n_topics"] = args.n_topics

    output_dir = Path(cfg.get("output_dir", "results/ablations"))
    runs = expand_grid(cfg)
    paired_docs = load_twinviews_huggingface()

    summary_rows: list[dict[str, Any]] = []
    run_results_cache: dict[str, list[Any]] = {}

    for i, run in enumerate(runs, start=1):
        run_cfg = run["config"]
        params = run["params"]

        # Match run_eval topic selection: keep full dataset and pass n_topics via config.
        if cfg.get("n_topics") is not None:
            run_cfg["n_topics"] = int(cfg["n_topics"])

        run_cfg_key = hashlib.sha256(
            json.dumps(run_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        run_dir = output_dir / f"run_{i:03d}"
        if run_cfg_key in run_results_cache:
            print(f"[{i}/{len(runs)}] Reusing computed results for duplicate effective config")
            run_results = run_results_cache[run_cfg_key]
        else:
            print(f"[{i}/{len(runs)}] Executing effective config")
            run_results = run_experiment(run_cfg, paired_docs)
            run_results_cache[run_cfg_key] = run_results

        write_results(run_results, run_dir)

        systems = sorted({r.system for r in run_results})
        for system in systems:
            sys_res = [r for r in run_results if r.system == system]
            total = len(sys_res)
            neutral = sum(1 for r in sys_res if r.judge_category == "Neutral")
            avg_cfair = sum(r.cfair_score for r in sys_res) / total if total else 0.0
            fair_vals = [r.fair_score for r in sys_res if r.fair_score is not None]
            avg_fair = sum(fair_vals) / len(fair_vals) if fair_vals else ""
            summary_rows.append(
                {
                    "run_id": i,
                    "alpha": params.get("alpha", ""),
                    "tau": params.get("tau", ""),
                    "N": params.get("N", ""),
                    "embedding_model": params.get("embedding_model", ""),
                    "axis_source": params.get("axis_source", ""),
                    "fairness_loss": params.get("fairness_loss", ""),
                    "system": system,
                    "neutral_count": neutral,
                    "total": total,
                    "neutral_rate": (neutral / total) if total else 0.0,
                    "avg_cfair": avg_cfair,
                    "avg_fair": avg_fair,
                    "results_path": str(run_dir),
                }
            )

    summary_path = write_ablation_summary(summary_rows, output_dir)
    print(f"Ablation summary written to {summary_path}")


if __name__ == "__main__":
    main()
