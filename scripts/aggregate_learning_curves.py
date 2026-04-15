"""Learning-curve aggregator for the headline table (Batch D3).

Reads the CSVs written by HCMARLLogger for every (method, seed) combination
listed in config/experiment_matrix.yaml and produces a summary of IQM +
stratified bootstrap 95% CI on cumulative_reward at each anchor step
(default 1M / 3M / 5M).

Usage:
    python scripts/aggregate_learning_curves.py \
        --matrix config/experiment_matrix.yaml \
        --logs-root logs \
        --out aggregated_results.json

Exit codes:
    0 — aggregation produced a table for every (method, seed, anchor).
    2 — at least one CSV is missing or an anchor step is not reached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcmarl.aggregation import aggregate_by_method


def score_at_step(csv_path: str, anchor_step: int) -> float:
    """Return cumulative_reward at the last episode whose global_step
    is <= anchor_step. Raises FileNotFoundError / KeyError / ValueError
    so the caller can surface an actionable error instead of silent NaN."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "global_step" not in df.columns:
        raise KeyError(f"{csv_path}: missing 'global_step' column")
    if "cumulative_reward" not in df.columns:
        raise KeyError(f"{csv_path}: missing 'cumulative_reward' column")
    mask = df["global_step"] <= anchor_step
    if not mask.any():
        raise ValueError(
            f"{csv_path}: no episodes reach anchor step {anchor_step:,} "
            f"(max observed: {int(df['global_step'].max()):,})"
        )
    return float(df.loc[mask, "cumulative_reward"].iloc[-1])


def aggregate(matrix_path: str, logs_root: str) -> Dict:
    with open(matrix_path, encoding="utf-8") as f:
        matrix = yaml.safe_load(f)

    anchors: List[int] = list(matrix["curve_anchors_steps"])
    seeds: List[int] = list(matrix["headline"]["seeds"])
    methods: Dict[str, Dict] = matrix["headline"]["methods"]

    report: Dict = {"anchors": anchors, "seeds": seeds, "by_anchor": {}}
    errors: List[str] = []

    for anchor in anchors:
        scores_by_method: Dict[str, np.ndarray] = {}
        for method_key in methods:
            vec = []
            for seed in seeds:
                csv_path = os.path.join(
                    logs_root, method_key, f"seed_{seed}", "training_log.csv",
                )
                try:
                    vec.append(score_at_step(csv_path, anchor))
                except (FileNotFoundError, KeyError, ValueError) as e:
                    errors.append(f"[anchor {anchor}] {method_key} seed {seed}: {e}")
            scores_by_method[method_key] = np.asarray(vec, dtype=np.float64)
        report["by_anchor"][str(anchor)] = aggregate_by_method(scores_by_method)

    report["errors"] = errors
    report["complete"] = (len(errors) == 0)
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="config/experiment_matrix.yaml")
    p.add_argument("--logs-root", default="logs")
    p.add_argument("--out", default="aggregated_results.json")
    args = p.parse_args()

    report = aggregate(args.matrix, args.logs_root)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for anchor_str, by_method in report["by_anchor"].items():
        print(f"\n=== Anchor: {int(anchor_str):,} steps ===")
        for method, stats in by_method.items():
            print(f"  {method:12s}  IQM={stats['iqm']:>8.2f}  "
                  f"95% CI=[{stats['ci_lo']:>8.2f}, {stats['ci_hi']:>8.2f}]  "
                  f"(n={stats['n_seeds']})")
    if report["errors"]:
        print("\nERRORS:")
        for e in report["errors"]:
            print(f"  - {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
