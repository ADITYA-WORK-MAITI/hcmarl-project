"""
HC-MARL Phase 4 (#244): Safety-Gym Validation Launcher
4 methods x 2 envs x 5 seeds = 40 runs.
"""
import argparse
import os
import sys
import subprocess

METHODS = ["ppo_lag", "cpo", "focops", "ecbf_ppo"]
ENVS = ["pointgoal", "antvelocity"]
SEEDS = [0, 1, 2, 3, 4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = len(args.methods) * len(args.envs) * len(args.seeds)
    print(f"Running {total} Safety-Gym jobs")

    for env_name in args.envs:
        for method in args.methods:
            for seed in args.seeds:
                config = f"config/safety_gym_{env_name}.yaml"
                ckpt_dir = f"checkpoints/safety_gym/{env_name}/{method}/seed_{seed}"
                log_dir = f"logs/safety_gym/{env_name}/{method}/seed_{seed}"
                os.makedirs(ckpt_dir, exist_ok=True)
                os.makedirs(log_dir, exist_ok=True)

                cmd = [
                    sys.executable, "scripts/train.py",
                    "--config", config if os.path.exists(config) else "config/hcmarl_full_config.yaml",
                    "--method", method if method != "ecbf_ppo" else "hcmarl",
                    "--seed", str(seed),
                    "--device", args.device,
                ]
                print(f"  {env_name}/{method} seed={seed}")
                if not args.dry_run:
                    subprocess.run(cmd)

    print(f"\nAll {total} Safety-Gym jobs {'would be ' if args.dry_run else ''}complete.")

if __name__ == "__main__":
    main()
