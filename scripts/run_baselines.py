"""
HC-MARL Phase 4 (#68): Batch Baseline Launcher
Iterate over 6 baselines x 5 seeds, train each, checkpoint.

Usage:
    python scripts/run_baselines.py --device cuda
    python scripts/run_baselines.py --methods mappo ippo --seeds 0 1 2
"""
import argparse
import os
import sys
import yaml
import subprocess

BASELINES = ["mappo", "ippo", "mappo_lag", "ppo_lag", "cpo", "macpo"]
SEEDS = [0, 1, 2, 3, 4]

CONFIG_MAP = {
    "mappo": "config/mappo_config.yaml",
    "ippo": "config/ippo_config.yaml",
    "mappo_lag": "config/mappo_lag_config.yaml",
    "ppo_lag": "config/ppo_lag_config.yaml",
    "cpo": "config/cpo_config.yaml",
    "macpo": "config/macpo_config.yaml",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=BASELINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    total = len(args.methods) * len(args.seeds)
    print(f"Running {total} training jobs: {len(args.methods)} methods x {len(args.seeds)} seeds")

    for i, method in enumerate(args.methods):
        for j, seed in enumerate(args.seeds):
            run_id = i * len(args.seeds) + j + 1
            config = CONFIG_MAP.get(method, f"config/{method}_config.yaml")
            if not os.path.exists(config):
                config = "config/hcmarl_full_config.yaml"

            cmd = [
                sys.executable, "scripts/train.py",
                "--config", config,
                "--method", method,
                "--seed", str(seed),
                "--device", args.device,
            ]

            print(f"\n[{run_id}/{total}] {method} seed={seed}")
            print(f"  Config: {config}")
            print(f"  Command: {' '.join(cmd)}")

            if not args.dry_run:
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  FAILED (exit code {result.returncode})")
                else:
                    print(f"  DONE")

    print(f"\nAll {total} jobs {'would be ' if args.dry_run else ''}complete.")


if __name__ == "__main__":
    main()
