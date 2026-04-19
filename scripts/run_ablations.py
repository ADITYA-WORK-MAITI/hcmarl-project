"""Build-up ablation launcher — Experiment A attribution ladder (Batch D2).

Reads `config/experiment_matrix.yaml` as the single source of truth and
launches every (rung, seed) pair under `ablation:`. Per the Batch D
design that file is canonical — adding a rung or changing a seed list
is a YAML edit, never a code edit here.

The build-up ladder (instead of the old remove-one design) reads as a
direct attribution: rung k - rung k-1 = the contribution of the component
added at step k. Default ladder:

    rung name           component added              config
    -----------------   --------------------------   ---------------------------------
    mappo               (none — bare MAPPO)          config/mappo_config.yaml
    plus_ecbf           ECBF safety filter            config/ablation_no_nswf.yaml
    plus_nswf           NSWF allocator                config/ablation_no_ecbf.yaml
    plus_ecbf_nswf      both ECBF + NSWF              config/ablation_no_mmicrl.yaml
    full_hcmarl         + MMICRL type discovery       config/hcmarl_full_config.yaml

5 rungs x 5 seeds = 25 runs.

Each rung is launched with `--run-name ablation_<rung>` so the logs land
at `logs/ablation_<rung>/seed_<s>/training_log.csv`. This keeps them
isolated from the headline runs at `logs/{method}/seed_{seed}/`
(in particular, ablation rung "mappo" would otherwise overwrite the
headline mappo logs at logs/mappo/seed_0/).

Usage:
    python scripts/run_ablations.py --device cuda
    python scripts/run_ablations.py --rungs plus_ecbf full_hcmarl
    python scripts/run_ablations.py --dry-run
    python scripts/run_ablations.py --device cuda \\
        --drive-backup-dir /content/drive/MyDrive/hcmarl_backup

Exit codes:
    0 — every launched subprocess returned 0.
    1 — at least one subprocess returned non-zero.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

import yaml

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
    parser.add_argument("--rungs", nargs="+", default=None,
                        help="Override rung-name list. Default: every rung under matrix['ablation']['rungs'].")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seed list. Default: matrix['ablation']['seeds'].")
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu, cuda, or auto (default: auto).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without launching subprocesses.")
    parser.add_argument("--drive-backup-dir", type=str, default=None,
                        help="Mirror checkpoints+logs here every checkpoint interval.")
    parser.add_argument("--budget-inr", type=float, default=0.0,
                        help="Hard kill-switch: halt each run when wall-clock spend reaches "
                             "this INR amount. Default 0 disables.")
    parser.add_argument("--cost-per-hour", type=float, default=49.0,
                        help="GPU instance hourly rate in INR (default 49.0 on-demand, 17 spot).")
    parser.add_argument("--budget-margin", type=float, default=0.95,
                        help="Trip kill-switch at budget_inr * margin (default 0.95).")
    args = parser.parse_args()

    matrix = _load_matrix(args.matrix)
    ablation = matrix.get("ablation") or {}
    rungs_list: List[Dict] = ablation.get("rungs") or []
    if not rungs_list:
        print(f"ERROR: matrix file {args.matrix} has no ablation.rungs section",
              file=sys.stderr)
        return 1

    rungs_by_name: Dict[str, Dict] = {r["name"]: r for r in rungs_list}
    rung_names: List[str] = list(args.rungs) if args.rungs else [r["name"] for r in rungs_list]
    seeds: List[int] = list(args.seeds) if args.seeds else list(ablation.get("seeds") or [])
    if not seeds:
        print(f"ERROR: matrix file {args.matrix} has no ablation.seeds list",
              file=sys.stderr)
        return 1

    unknown = [r for r in rung_names if r not in rungs_by_name]
    if unknown:
        print(f"ERROR: rungs {unknown} not declared in matrix ablation.rungs "
              f"(known: {list(rungs_by_name.keys())})", file=sys.stderr)
        return 1

    total = len(rung_names) * len(seeds)
    print(f"Ablation ladder: {len(rung_names)} rungs x {len(seeds)} seeds = {total} runs")
    print(f"Rungs: {rung_names}")
    print(f"Seeds: {seeds}")
    if args.drive_backup_dir:
        print(f"Drive backup: {args.drive_backup_dir}")
    if args.dry_run:
        print("(dry-run mode — no subprocesses will be launched)\n")

    failures: List[tuple] = []
    for i, rung_name in enumerate(rung_names):
        spec = rungs_by_name[rung_name]
        config_path = spec["config"]
        method_arg = spec["method"]
        # Prefix the per-rung log dir so ablation rung "mappo" does not
        # collide with headline method "mappo" at logs/mappo/seed_0/.
        run_name = f"ablation_{rung_name}"

        for j, seed in enumerate(seeds):
            run_id = i * len(seeds) + j + 1
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", config_path,
                "--method", method_arg,
                "--seed", str(seed),
                "--device", args.device,
                "--run-name", run_name,
            ]
            if args.drive_backup_dir:
                cmd += ["--drive-backup-dir", args.drive_backup_dir]
            if args.budget_inr > 0:
                cmd += ["--budget-inr", str(args.budget_inr),
                        "--cost-per-hour", str(args.cost_per_hour),
                        "--budget-margin", str(args.budget_margin)]

            print(f"\n[{run_id}/{total}] {rung_name} (method={method_arg}) seed={seed}")
            print(f"  Config: {config_path}")
            print(f"  Command: {' '.join(cmd)}")

            if args.dry_run:
                continue
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  FAILED (exit code {result.returncode})")
                failures.append((rung_name, seed, result.returncode))
            else:
                print(f"  DONE")

    print(f"\nAll {total} jobs {'would be ' if args.dry_run else ''}complete.")
    if failures:
        print(f"{len(failures)} failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
