"""
HC-MARL Phase 4 (#187): Batch Scaling Launcher
N={3,4,6,8,12} x 5 seeds = 25 runs.
"""
import argparse
import os
import sys
import subprocess

WORKER_COUNTS = [3, 4, 6, 8, 12]
SEEDS = [0, 1, 2, 3, 4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", type=int, default=WORKER_COUNTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drive-backup-dir", type=str, default=None)
    args = parser.parse_args()

    total = len(args.workers) * len(args.seeds)
    print(f"Running {total} scaling jobs")

    for n in args.workers:
        for seed in args.seeds:
            config = f"config/scaling_n{n}.yaml"
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
            if args.drive_backup_dir:
                cmd += ["--drive-backup-dir", args.drive_backup_dir]
            print(f"  N={n} seed={seed}: {' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd)

    print(f"\nAll {total} scaling jobs {'would be ' if args.dry_run else ''}complete.")


if __name__ == "__main__":
    main()
