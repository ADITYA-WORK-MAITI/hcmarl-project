"""
HC-MARL Phase 4 (#130): Batch Ablation Launcher
5 ablations x 5 seeds = 25 runs.

Ablations:
  1. no_ecbf        - ECBF safety filter removed
  2. no_nswf        - NSWF replaced with round-robin
  3. no_mmicrl      - Fixed Theta_max (no learned types)
  4. no_reperfusion - r=1 always (no reperfusion switch)
  5. no_divergent   - Constant D_i=kappa (no fatigue dependence)
"""
import argparse
import os
import sys
import subprocess

ABLATIONS = ["no_ecbf", "no_nswf", "no_mmicrl", "no_reperfusion", "no_divergent"]
SEEDS = [0, 1, 2, 3, 4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", default=ABLATIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = len(args.ablations) * len(args.seeds)
    print(f"Running {total} ablation jobs")

    for ablation in args.ablations:
        for seed in args.seeds:
            config = f"config/ablation_{ablation}.yaml"
            if not os.path.exists(config):
                print(f"  WARNING: {config} not found, using hcmarl_full_config.yaml")
                config = "config/hcmarl_full_config.yaml"

            cmd = [
                sys.executable, "scripts/train.py",
                "--config", config,
                "--method", "hcmarl",
                "--seed", str(seed),
                "--device", args.device,
            ]
            print(f"  Ablation={ablation} seed={seed}: {' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd)

    print(f"\nAll {total} ablation jobs {'would be ' if args.dry_run else ''}complete.")


if __name__ == "__main__":
    main()
