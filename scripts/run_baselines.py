"""Headline grid launcher — Experiment A (HC-MARL vs. baselines).

Reads `config/experiment_matrix.yaml` as the single source of truth and
launches every (method, seed) pair under `headline:`. Per the Batch D
design that file is canonical — adding a new method or seed is a YAML
edit, never a code edit here.

The default grid is:
    methods : hcmarl, mappo, ippo, mappo_lag    (4)
    seeds   : 0..9                              (10)
    total   : 40 runs

Each run is launched as a sequential subprocess invocation of
scripts/train.py. Logs land at logs/{method}/seed_{seed}/training_log.csv,
which is what scripts/aggregate_learning_curves.py reads from.

Usage:
    python scripts/run_baselines.py --device cuda
    python scripts/run_baselines.py --methods mappo ippo --seeds 0 1 2
    python scripts/run_baselines.py --dry-run
    python scripts/run_baselines.py --device cuda \\
        --drive-backup-dir /content/drive/MyDrive/hcmarl_backup

Exit codes:
    0 — every launched subprocess returned 0.
    1 — at least one subprocess returned non-zero (failures listed at end).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

import yaml

# experiment_matrix.yaml lives next to this script's parent.
DEFAULT_MATRIX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "experiment_matrix.yaml",
)


def _load_matrix(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", default=DEFAULT_MATRIX,
                        help="Path to experiment_matrix.yaml (default: repo config/).")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Override method list. Default: every key under matrix['headline']['methods'].")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seed list. Default: matrix['headline']['seeds'].")
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu, cuda, or auto (default: auto).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without launching subprocesses.")
    parser.add_argument("--drive-backup-dir", type=str, default=None,
                        help="Mirror checkpoints+logs here every checkpoint interval "
                             "(use /content/drive/MyDrive/hcmarl_backup on Colab).")
    parser.add_argument("--budget-inr", type=float, default=0.0,
                        help="Hard kill-switch: halt each run when wall-clock spend reaches "
                             "this INR amount. Default 0 disables.")
    parser.add_argument("--cost-per-hour", type=float, default=49.0,
                        help="GPU instance hourly rate in INR (default 49.0 on-demand, 17 spot).")
    parser.add_argument("--budget-margin", type=float, default=0.95,
                        help="Trip kill-switch at budget_inr * margin (default 0.95).")
    args = parser.parse_args()

    matrix = _load_matrix(args.matrix)
    headline = matrix.get("headline") or {}
    methods_map: Dict[str, Dict] = headline.get("methods") or {}
    if not methods_map:
        print(f"ERROR: matrix file {args.matrix} has no headline.methods section",
              file=sys.stderr)
        return 1

    method_keys: List[str] = list(args.methods) if args.methods else list(methods_map.keys())
    seeds: List[int] = list(args.seeds) if args.seeds else list(headline.get("seeds") or [])
    if not seeds:
        print(f"ERROR: matrix file {args.matrix} has no headline.seeds list",
              file=sys.stderr)
        return 1

    # Validate methods against the matrix before we burn any GPU minutes.
    unknown = [m for m in method_keys if m not in methods_map]
    if unknown:
        print(f"ERROR: methods {unknown} not declared in matrix headline.methods "
              f"(known: {list(methods_map.keys())})", file=sys.stderr)
        return 1

    total = len(method_keys) * len(seeds)
    print(f"Headline grid: {len(method_keys)} methods x {len(seeds)} seeds = {total} runs")
    print(f"Methods: {method_keys}")
    print(f"Seeds:   {seeds}")
    if args.drive_backup_dir:
        print(f"Drive backup: {args.drive_backup_dir}")
    if args.dry_run:
        print("(dry-run mode — no subprocesses will be launched)\n")

    failures: List[tuple] = []
    for i, method_key in enumerate(method_keys):
        spec = methods_map[method_key]
        config_path = spec["config"]
        method_arg = spec.get("method", method_key)

        for j, seed in enumerate(seeds):
            run_id = i * len(seeds) + j + 1
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", config_path,
                "--method", method_arg,
                "--seed", str(seed),
                "--device", args.device,
                # run_name = method_key keeps the headline contract:
                # logs/{method_key}/seed_{seed}/ — what aggregate_learning_curves.py
                # iterates over.
                "--run-name", method_key,
            ]
            if args.drive_backup_dir:
                cmd += ["--drive-backup-dir", args.drive_backup_dir]
            if args.budget_inr > 0:
                cmd += ["--budget-inr", str(args.budget_inr),
                        "--cost-per-hour", str(args.cost_per_hour),
                        "--budget-margin", str(args.budget_margin)]

            print(f"\n[{run_id}/{total}] {method_key} (method={method_arg}) seed={seed}")
            print(f"  Config: {config_path}")
            print(f"  Command: {' '.join(cmd)}")

            if args.dry_run:
                continue
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  FAILED (exit code {result.returncode})")
                failures.append((method_key, seed, result.returncode))
            else:
                print(f"  DONE")

    print(f"\nAll {total} jobs {'would be ' if args.dry_run else ''}complete.")
    if failures:
        print(f"{len(failures)} failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
