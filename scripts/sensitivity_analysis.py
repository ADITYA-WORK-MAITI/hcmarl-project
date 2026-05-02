"""Path G calibration sensitivity analysis (R1 revision item 1).

Re-runs `scripts/evaluate.py` on existing checkpoints with the Path G
calibrated F, R parameters perturbed by +/- 20%, +/- 50% (and 0% as the
canonical reference). Reports headline metrics across the perturbation
grid. The intent is to demonstrate -- without spending any new GPU
training time -- whether HC-MARL's headline gains survive realistic
calibration error.

Method
------
We perturb each worker's per-muscle F and R values uniformly:

    F_pert = F_base * (1 + delta)
    R_pert = R_base * (1 + delta)
    delta in {-0.5, -0.2, 0.0, +0.2, +0.5}

The reperfusion multiplier r is held fixed (r=15 for non-grip muscles,
r=30 for hand-grip per Looft & Frey-Law 2018, 2020). Perturbing r would
collapse the reperfusion regime and ablate a different component;
sensitivity over r belongs in `ablation_no_reperfusion`.

For each perturbation level, we regenerate `config/pathg_profiles_pert_<lvl>.json`
and invoke `scripts/evaluate.py` on the existing checkpoints under
`checkpoints/<method>/seed_<s>/checkpoint_final.pt`. Eval metrics are
collected per (method, seed, perturbation_level) and written to a tidy
CSV at `results/sensitivity/sensitivity_metrics.csv`.

Aggregation
-----------
After all (method, seed, level) cells have evaluation outputs, we
report:
  - IQM + 95% bootstrap CI of cumulative_reward across seeds, per
    (method, level)
  - safety_violation_rate IQM + CI per (method, level)
  - Headline-shift table: |IQM(level=+/-50%) - IQM(level=0%)| / IQM(0%)
    -- a relative change in headline metric. < 10% across all methods
    is the credibility-win threshold.

Usage
-----
After EXP1 + EXP2 land on L40S and `checkpoints/<method>/` is populated:

    python scripts/sensitivity_analysis.py \\
        --methods hcmarl mappo mappo_lag macpo happo shielded_mappo \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --levels -0.5 -0.2 0.0 0.2 0.5 \\
        --n-eval-episodes 50 \\
        --device cuda \\
        --out-dir results/sensitivity

Pre-launch dry run (validates the perturbed JSON files render correctly
without invoking eval):

    python scripts/sensitivity_analysis.py --dry-run

Exit codes: 0 on success, 1 on any subprocess failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROFILES = os.path.join(REPO_ROOT, "config", "pathg_profiles.json")


def _load_profiles(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _perturb(profiles_doc: Dict, delta: float) -> Dict:
    """Return a deep-copied profiles document with F and R scaled by
    (1+delta) for every (worker, muscle). r is preserved.

    All other fields (worker_id, source_subject, task_type, *_source) are
    passed through unchanged so the perturbed file is structurally
    identical to the canonical and can be loaded by the same code path.
    """
    out = json.loads(json.dumps(profiles_doc))  # deep copy
    out["description"] = (
        f"{profiles_doc.get('description','Path G profiles')} | "
        f"PERTURBED for sensitivity analysis: F,R scaled by (1+{delta:+.2f})"
    )
    out["sensitivity_perturbation"] = {"delta_F": delta, "delta_R": delta}
    factor = 1.0 + delta
    for prof in out.get("profiles", []):
        muscles = prof.get("muscles", {})
        for mname, mvals in muscles.items():
            if "F" in mvals:
                mvals["F"] = float(mvals["F"]) * factor
            if "R" in mvals:
                mvals["R"] = float(mvals["R"]) * factor
    return out


def _write_perturbed_profiles(base_doc: Dict, levels: List[float], out_dir: str) -> Dict[float, str]:
    """Write one perturbed JSON per level. Returns {level: filepath}."""
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[float, str] = {}
    for lvl in levels:
        tag = f"{lvl:+.2f}".replace("+", "p").replace("-", "n").replace(".", "_")
        path = os.path.join(out_dir, f"pathg_profiles_pert_{tag}.json")
        doc = _perturb(base_doc, lvl)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        out[lvl] = path
    return out


def _eval_one(method: str, seed: int, ckpt: str, config: str,
              perturbed_profiles_path: str, n_episodes: int,
              device: str, out_dir: str) -> Optional[Dict]:
    """Run evaluate.py with the perturbed profiles JSON in place of the
    canonical one. evaluate.py reads cfg["mmicrl_results"] -> pathg
    profiles via the env builder; we override by setting an env var
    that train.py / evaluate.py honour as the path to the profiles JSON.

    Returns parsed metrics dict on success, None on subprocess failure.
    """
    env = os.environ.copy()
    env["HCMARL_PATHG_PROFILES_OVERRIDE"] = perturbed_profiles_path
    metrics_out = os.path.join(out_dir, f"{method}_seed{seed}_metrics.json")
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", ckpt,
        "--config", config,
        "--method", method,
        "--n-episodes", str(n_episodes),
        "--device", device,
        "--seed", str(seed),
        "--output", metrics_out,
    ]
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        print(f"  EVAL FAILED: rc={rc} for {method}/seed_{seed}")
        return None
    if not os.path.exists(metrics_out):
        print(f"  EVAL produced no metrics file: {metrics_out}")
        return None
    with open(metrics_out, encoding="utf-8") as f:
        return json.load(f)


def _config_for_method(method: str) -> str:
    """Map method name to its canonical config path. The hcmarl method
    uses 'hcmarl_full_config.yaml' (everything-on full stack); the rest
    follow the '<method>_config.yaml' naming convention."""
    if method == "hcmarl":
        return os.path.join(REPO_ROOT, "config", "hcmarl_full_config.yaml")
    return os.path.join(REPO_ROOT, "config", f"{method}_config.yaml")


def _ckpt_for_seed(method: str, seed: int) -> str:
    return os.path.join(
        REPO_ROOT, "checkpoints", method, f"seed_{seed}", "checkpoint_final.pt"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--methods", nargs="+",
                   default=["hcmarl", "mappo", "mappo_lag", "macpo", "happo",
                            "shielded_mappo"],
                   help="Methods to evaluate. Default: full headline lineup.")
    p.add_argument("--seeds", nargs="+", type=int,
                   default=list(range(10)),
                   help="Seeds to evaluate. Default: 0..9.")
    p.add_argument("--levels", nargs="+", type=float,
                   default=[-0.5, -0.2, 0.0, 0.2, 0.5],
                   help="Perturbation levels (relative). Default: -50%, -20%, 0, +20%, +50%.")
    p.add_argument("--n-eval-episodes", type=int, default=50,
                   help="Episodes per evaluation. Default 50.")
    p.add_argument("--device", default="auto",
                   help="cpu, cuda, or auto.")
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "sensitivity"),
                   help="Output directory for perturbed JSONs + per-run metrics + CSV.")
    p.add_argument("--profiles", default=DEFAULT_PROFILES,
                   help="Path to canonical pathg_profiles.json.")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate perturbed JSONs and print the eval grid without "
                        "running subprocesses. Useful for pre-launch validation.")
    args = p.parse_args()

    base_doc = _load_profiles(args.profiles)
    print(f"Loaded canonical profiles: {args.profiles}")
    print(f"  n_workers = {base_doc.get('n_workers')}")
    print(f"  F_range   = {base_doc.get('F_range')}")
    print(f"  Source    = {base_doc.get('source')}")

    pert_dir = os.path.join(args.out_dir, "perturbed_profiles")
    pert_paths = _write_perturbed_profiles(base_doc, args.levels, pert_dir)
    print(f"Wrote {len(pert_paths)} perturbed JSONs to {pert_dir}/")
    for lvl, path in pert_paths.items():
        print(f"  delta={lvl:+.2f}  -> {os.path.relpath(path, REPO_ROOT)}")

    if args.dry_run:
        print("\nDry-run grid:")
        n_jobs = len(args.methods) * len(args.seeds) * len(args.levels)
        print(f"  Total cells: {len(args.methods)} methods x "
              f"{len(args.seeds)} seeds x {len(args.levels)} levels = {n_jobs}")
        for m in args.methods:
            ckpt0 = _ckpt_for_seed(m, args.seeds[0])
            cfg = _config_for_method(m)
            ckpt_state = "EXISTS" if os.path.exists(ckpt0) else "MISSING (run EXP1 first)"
            cfg_state = "EXISTS" if os.path.exists(cfg) else "MISSING"
            print(f"    {m:18s}  ckpt={ckpt_state}  cfg={cfg_state}")
        return 0

    rows = []
    failures = []
    csv_path = os.path.join(args.out_dir, "sensitivity_metrics.csv")
    cells_dir = os.path.join(args.out_dir, "per_cell")
    os.makedirs(cells_dir, exist_ok=True)

    total = len(args.methods) * len(args.seeds) * len(args.levels)
    done = 0
    for method in args.methods:
        cfg = _config_for_method(method)
        if not os.path.exists(cfg):
            print(f"SKIP method={method}: config missing at {cfg}")
            continue
        for seed in args.seeds:
            ckpt = _ckpt_for_seed(method, seed)
            if not os.path.exists(ckpt):
                print(f"SKIP {method}/seed_{seed}: checkpoint missing at "
                      f"{os.path.relpath(ckpt, REPO_ROOT)}")
                continue
            for lvl in args.levels:
                done += 1
                cell_tag = f"{method}_seed{seed}_lvl{lvl:+.2f}"
                cell_dir = os.path.join(cells_dir, cell_tag.replace("+", "p").replace("-", "n").replace(".", "_"))
                os.makedirs(cell_dir, exist_ok=True)
                print(f"\n[{done}/{total}] {cell_tag}")
                metrics = _eval_one(
                    method=method, seed=seed, ckpt=ckpt, config=cfg,
                    perturbed_profiles_path=pert_paths[lvl],
                    n_episodes=args.n_eval_episodes,
                    device=args.device, out_dir=cell_dir,
                )
                if metrics is None:
                    failures.append((method, seed, lvl))
                    continue
                row = {
                    "method": method, "seed": seed, "delta": lvl,
                    "n_eval_episodes": args.n_eval_episodes,
                }
                # Flatten any scalar metric the eval JSON exposes.
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        row[k] = float(v)
                rows.append(row)

    if not rows:
        print("\nNo cells evaluated. Did EXP1 finish? Are checkpoints under checkpoints/<method>/seed_<s>/?")
        return 1

    cols = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {csv_path}")
    if failures:
        print(f"{len(failures)} cells failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
