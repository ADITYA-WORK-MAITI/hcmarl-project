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

# C-17: Per-ablation CLI flags to ensure each ablation actually changes behavior.
# Belt-and-suspenders: config YAML keys AND CLI flags both work.
ABLATION_FLAGS = {
    "no_ecbf":        ["--ecbf-mode", "off"],
    "no_nswf":        ["--no-nswf"],
    "no_mmicrl":      [],  # config mmicrl.enabled=false + use_fixed_theta=true handles this
    "no_reperfusion": [],  # config muscle_groups.*.r=1 is read by train.py
    "no_divergent":   ["--disagreement-type", "constant"],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", default=ABLATIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drive-backup-dir", type=str, default=None)
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
            # Append ablation-specific CLI flags (C-17)
            cmd.extend(ABLATION_FLAGS.get(ablation, []))
            if args.drive_backup_dir:
                cmd += ["--drive-backup-dir", args.drive_backup_dir]

            print(f"  Ablation={ablation} seed={seed}: {' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd)

    print(f"\nAll {total} ablation jobs {'would be ' if args.dry_run else ''}complete.")


if __name__ == "__main__":
    main()
