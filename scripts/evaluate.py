"""
HC-MARL Phase 4 (#55): Evaluation Script
==========================================
Load trained checkpoint, run N episodes, compute all 9 HC-MARL metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/hcmarl/seed_0/checkpoint_final.pt \
        --config config/hcmarl_full_config.yaml --method hcmarl --n-episodes 100
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
from hcmarl.logger import HCMARLLogger
from scripts.train import create_agent
import yaml


def evaluate(cfg, method, checkpoint_path, n_episodes=100, seed=42, device="cpu"):
    """Run evaluation and compute all 9 metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg = cfg.get("environment", {})
    n_workers = env_cfg.get("n_workers", 6)
    max_steps = env_cfg.get("max_steps", 480)

    env = WarehousePettingZoo(n_workers=n_workers, max_steps=max_steps,
                               theta_max=env_cfg.get("theta_max", None))
    obs_dim = env.obs_dim
    global_obs_dim = env.global_obs_dim
    n_actions = env.n_tasks

    agent = create_agent(method, obs_dim, global_obs_dim, n_actions, n_workers, cfg, device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")

    all_metrics = {k: [] for k in HCMARLLogger.METRIC_NAMES}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        total_violations = 0
        safe_steps = 0
        tasks_per_worker = np.zeros(n_workers)
        peak_mf = 0.0
        forced_rests = 0
        recovery_times = []
        in_violation = {i: False for i in range(n_workers)}
        violation_start = {i: 0 for i in range(n_workers)}

        for step in range(max_steps):
            global_state = env._get_global_obs()
            result = agent.get_actions(obs, global_state)
            actions = result[0] if isinstance(result, tuple) else result

            obs, rewards, terms, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values())

            step_violations = 0
            for i, (agent_id, info) in enumerate(sorted(infos.items())):
                fatigue = info.get("fatigue", {})
                task = info.get("task", "rest")
                for m, mf in fatigue.items():
                    theta = env.theta_max.get(m, 1.0)
                    if mf > theta:
                        step_violations += 1
                        if not in_violation[i]:
                            in_violation[i] = True
                            violation_start[i] = step
                    else:
                        if in_violation[i]:
                            recovery_times.append(step - violation_start[i])
                            in_violation[i] = False
                    peak_mf = max(peak_mf, mf)
                if task != "rest":
                    tasks_per_worker[i] += 1
                else:
                    avg_mf = np.mean(list(fatigue.values())) if fatigue else 0
                    if avg_mf > 0.3:
                        forced_rests += 1

            total_violations += step_violations
            if step_violations == 0:
                safe_steps += 1

            if all(terms.values()):
                break

        n_steps = step + 1
        n = n_workers
        jain = float((tasks_per_worker.sum()**2) / (n * (tasks_per_worker**2).sum() + 1e-8)) if tasks_per_worker.sum() > 0 else 1.0

        all_metrics["violation_rate"].append(total_violations / max(1, n_steps * n_workers * env.n_muscles))
        all_metrics["cumulative_cost"].append(float(total_violations))
        all_metrics["safety_rate"].append(safe_steps / max(1, n_steps))
        all_metrics["tasks_completed"].append(float(tasks_per_worker.sum()))
        all_metrics["cumulative_reward"].append(total_reward)
        all_metrics["jain_fairness"].append(jain)
        all_metrics["peak_fatigue"].append(peak_mf)
        all_metrics["forced_rest_rate"].append(forced_rests / max(1, n_steps * n_workers))
        all_metrics["constraint_recovery_time"].append(np.mean(recovery_times) if recovery_times else 0.0)

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: R={total_reward:.1f}, Safe={safe_steps/n_steps:.2%}")

    # Compute mean ± std
    results = {}
    for k, vals in all_metrics.items():
        results[f"{k}_mean"] = float(np.mean(vals))
        results[f"{k}_std"] = float(np.std(vals))

    results["method"] = method
    results["n_episodes"] = n_episodes
    results["seed"] = seed
    results["checkpoint"] = checkpoint_path

    return results


def main():
    parser = argparse.ArgumentParser(description="HC-MARL Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, default="hcmarl")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = evaluate(cfg, args.method, args.checkpoint, args.n_episodes, args.seed, args.device)

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {args.method} (seed {args.seed})")
    print(f"{'='*60}")
    for k in HCMARLLogger.METRIC_NAMES:
        print(f"  {k:30s}: {results[f'{k}_mean']:8.4f} +/- {results[f'{k}_std']:8.4f}")

    # Save
    out_path = args.output or f"results/{args.method}_seed{args.seed}_eval.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
