"""
HC-MARL Phase 2: Training & Experiments
=========================================
Experiment runner for:
  - MAPPO training with ECBF + NSWF (HC-MARL full)
  - 10 baseline comparisons (5 seeds each)
  - 5 ablation studies (5 seeds each)
  - Scaling tests N = {3, 4, 6, 8, 12}

Metrics tracked (9 total):
  1. violation_rate       - fraction of steps with MF > Θmax
  2. cumulative_cost      - total constraint violations
  3. safety_rate          - fraction of steps with 0 violations
  4. tasks_completed      - total productive (non-rest) assignments
  5. cumulative_reward    - total episode reward
  6. jain_fairness        - Jain's fairness index on task counts
  7. peak_fatigue         - max MF across all workers and muscles
  8. forced_rest_rate     - fraction of steps where NSWF forced rest
  9. constraint_recovery  - avg steps to recover from near-violation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import time
import copy

from hcmarl.warehouse_env import WarehouseMultiAgentEnv, SingleWorkerWarehouseEnv
from hcmarl.mappo_agent import MAPPOAgent, train_mappo_episode
from hcmarl.baselines import create_all_baselines


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    env: WarehouseMultiAgentEnv,
    policy,
    n_episodes: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Run policy in environment and compute all 9 HC-MARL metrics.

    Args:
        env: WarehouseMultiAgentEnv instance
        policy: object with get_actions(observations, **kwargs) -> Dict[str, int]
        n_episodes: number of evaluation episodes
        seed: random seed

    Returns:
        Dict with averaged metrics over n_episodes.
    """
    np.random.seed(seed)
    all_metrics = {
        "violation_rate": [],
        "cumulative_cost": [],
        "safety_rate": [],
        "tasks_completed": [],
        "cumulative_reward": [],
        "jain_fairness": [],
        "peak_fatigue": [],
        "forced_rest_rate": [],
        "constraint_recovery": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        total_violations = 0
        total_steps = 0
        safe_steps = 0
        tasks_per_worker = np.zeros(env.n_workers)
        rest_forced = 0
        peak_mf = 0.0
        recovery_times = []
        in_violation = {i: False for i in range(env.n_workers)}
        violation_start = {i: 0 for i in range(env.n_workers)}

        for step in range(env.max_steps):
            actions = policy.get_actions(obs)
            obs, rewards, terms, truncs, infos = env.step(actions)

            total_reward += sum(rewards.values())
            total_steps += 1

            step_violations = 0
            for i, (agent, info) in enumerate(sorted(infos.items())):
                fatigue = info.get("fatigue", {})
                task_name = info.get("task", "rest")

                # Track tasks per worker
                if task_name != "rest":
                    tasks_per_worker[i] += 1

                # Check violations
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

                # Forced rest detection (NSWF surplus negative)
                if task_name == "rest":
                    avg_mf = np.mean(list(fatigue.values())) if fatigue else 0
                    if avg_mf > 0.3:  # rest was likely forced, not chosen
                        rest_forced += 1

            total_violations += step_violations
            if step_violations == 0:
                safe_steps += 1

            if all(terms.values()):
                break

        # Jain's fairness index
        n = env.n_workers
        if tasks_per_worker.sum() > 0:
            jain = (tasks_per_worker.sum() ** 2) / (n * (tasks_per_worker ** 2).sum() + 1e-8)
        else:
            jain = 1.0

        all_metrics["violation_rate"].append(total_violations / max(1, total_steps * env.n_workers * env.n_muscles))
        all_metrics["cumulative_cost"].append(float(total_violations))
        all_metrics["safety_rate"].append(safe_steps / max(1, total_steps))
        all_metrics["tasks_completed"].append(int(tasks_per_worker.sum()))
        all_metrics["cumulative_reward"].append(total_reward)
        all_metrics["jain_fairness"].append(jain)
        all_metrics["peak_fatigue"].append(peak_mf)
        all_metrics["forced_rest_rate"].append(rest_forced / max(1, total_steps * env.n_workers))
        all_metrics["constraint_recovery"].append(
            np.mean(recovery_times) if recovery_times else 0.0
        )

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ---------------------------------------------------------------------------
# Baseline comparison experiment
# ---------------------------------------------------------------------------

def run_baseline_comparison(
    n_workers: int = 4,
    max_steps: int = 60,
    n_seeds: int = 5,
    n_eval_episodes: int = 5,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run all 10 baselines + HC-MARL (MAPPO with ECBF+NSWF) across multiple seeds.
    Returns: {method_name: {metric: mean_value}}.
    """
    results = {}

    for seed in range(n_seeds):
        env = WarehouseMultiAgentEnv(n_workers=n_workers, max_steps=max_steps)
        obs_dim = 3 * env.n_muscles + 1
        n_actions = env.n_tasks

        # Create baselines
        baselines = create_all_baselines(obs_dim, n_actions, env.n_muscles, seed)

        # HC-MARL (MAPPO agent)
        global_obs_dim = n_workers * env.n_muscles * 3 + 1
        mappo = MAPPOAgent(
            n_agents=n_workers, obs_dim=obs_dim,
            global_obs_dim=global_obs_dim, n_actions=n_actions, seed=seed,
        )

        all_policies = baselines + [mappo]
        policy_names = [b.name for b in baselines] + ["HC-MARL (MAPPO+ECBF+NSWF)"]

        for policy, name in zip(all_policies, policy_names):
            metrics = compute_metrics(env, policy, n_eval_episodes, seed)

            if name not in results:
                results[name] = {k: [] for k in metrics}
            for k, v in metrics.items():
                results[name][k].append(v)

        if verbose:
            print(f"Seed {seed + 1}/{n_seeds} complete.")

    # Average across seeds
    averaged = {}
    for name, metric_lists in results.items():
        averaged[name] = {k: float(np.mean(v)) for k, v in metric_lists.items()}

    return averaged


# ---------------------------------------------------------------------------
# Ablation studies
# ---------------------------------------------------------------------------

def run_ablation_studies(
    n_workers: int = 4,
    max_steps: int = 60,
    n_seeds: int = 5,
    n_eval_episodes: int = 5,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    5 ablation studies:
      1. Full HC-MARL (MAPPO + ECBF + NSWF)
      2. No ECBF (remove safety filter)
      3. No NSWF (random allocation instead of Nash)
      4. No Divergent Di (use Di = κ·MF² instead of κ·MF²/(1-MF))
      5. No Reperfusion (r = 1 instead of r = 15)
    """
    results = {}

    for seed in range(n_seeds):
        configs = {}

        # Ablation 1: Full HC-MARL
        configs["Full HC-MARL"] = {
            "use_ecbf": True, "use_nswf": True,
            "divergent_di": True, "reperfusion": True,
        }

        # Ablation 2: No ECBF
        configs["No ECBF"] = {
            "use_ecbf": False, "use_nswf": True,
            "divergent_di": True, "reperfusion": True,
        }

        # Ablation 3: No NSWF (random allocation)
        configs["No NSWF"] = {
            "use_ecbf": True, "use_nswf": False,
            "divergent_di": True, "reperfusion": True,
        }

        # Ablation 4: No Divergent Di
        configs["No Divergent Di"] = {
            "use_ecbf": True, "use_nswf": True,
            "divergent_di": False, "reperfusion": True,
        }

        # Ablation 5: No Reperfusion (r=1)
        configs["No Reperfusion (r=1)"] = {
            "use_ecbf": True, "use_nswf": True,
            "divergent_di": True, "reperfusion": False,
        }

        for ablation_name, config in configs.items():
            # Isometric (F, R) from Table 1 for sustained warehouse holds
            muscle_groups = {
                "shoulder": {"F": 0.0146, "R": 0.00058, "r": 15 if config["reperfusion"] else 1},
                "elbow":    {"F": 0.00912, "R": 0.00094, "r": 15 if config["reperfusion"] else 1},
                "grip":     {"F": 0.00794, "R": 0.00109, "r": 30 if config["reperfusion"] else 1},  # Looft et al. (2018) Table 2: r=30 for hand grip
            }

            env = WarehouseMultiAgentEnv(
                n_workers=n_workers, max_steps=max_steps,
                muscle_groups=muscle_groups,
            )

            obs_dim = 3 * env.n_muscles + 1
            n_actions = env.n_tasks

            # Use appropriate policy based on ablation
            if config["use_nswf"]:
                global_obs_dim = n_workers * env.n_muscles * 3 + 1
                policy = MAPPOAgent(
                    n_agents=n_workers, obs_dim=obs_dim,
                    global_obs_dim=global_obs_dim, n_actions=n_actions, seed=seed,
                )
            else:
                from hcmarl.baselines import RandomBaseline
                policy = RandomBaseline(n_actions, seed)

            metrics = compute_metrics(env, policy, n_eval_episodes, seed)

            if ablation_name not in results:
                results[ablation_name] = {k: [] for k in metrics}
            for k, v in metrics.items():
                results[ablation_name][k].append(v)

        if verbose:
            print(f"Ablation seed {seed + 1}/{n_seeds} complete.")

    averaged = {}
    for name, metric_lists in results.items():
        averaged[name] = {k: float(np.mean(v)) for k, v in metric_lists.items()}

    return averaged


# ---------------------------------------------------------------------------
# Scaling tests
# ---------------------------------------------------------------------------

def run_scaling_tests(
    worker_counts: List[int] = None,
    max_steps: int = 60,
    n_seeds: int = 3,
    n_eval_episodes: int = 3,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    Test HC-MARL performance across different worker counts.
    Default: N = {3, 4, 6, 8, 12}.
    """
    if worker_counts is None:
        worker_counts = [3, 4, 6, 8, 12]

    results = {}

    for n_workers in worker_counts:
        seed_results = {
            "violation_rate": [], "cumulative_reward": [],
            "jain_fairness": [], "peak_fatigue": [],
            "tasks_completed": [], "wall_time": [],
        }

        for seed in range(n_seeds):
            env = WarehouseMultiAgentEnv(n_workers=n_workers, max_steps=max_steps)
            obs_dim = 3 * env.n_muscles + 1
            n_actions = env.n_tasks
            global_obs_dim = n_workers * env.n_muscles * 3 + 1

            policy = MAPPOAgent(
                n_agents=n_workers, obs_dim=obs_dim,
                global_obs_dim=global_obs_dim, n_actions=n_actions, seed=seed,
            )

            t0 = time.time()
            metrics = compute_metrics(env, policy, n_eval_episodes, seed)
            wall_time = time.time() - t0

            for k in seed_results:
                if k == "wall_time":
                    seed_results[k].append(wall_time)
                elif k in metrics:
                    seed_results[k].append(metrics[k])

        results[n_workers] = {k: float(np.mean(v)) for k, v in seed_results.items()}

        if verbose:
            print(f"N={n_workers}: reward={results[n_workers]['cumulative_reward']:.1f}, "
                  f"violations={results[n_workers]['violation_rate']:.4f}, "
                  f"fairness={results[n_workers]['jain_fairness']:.3f}, "
                  f"wall_time={results[n_workers]['wall_time']:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Full experiment suite
# ---------------------------------------------------------------------------

def run_all_experiments(
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the complete Phase 2 experiment suite and save results."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    if verbose:
        print("=" * 60)
        print("HC-MARL Phase 2: Full Experiment Suite")
        print("=" * 60)

    # 1. Baseline comparison
    if verbose:
        print("\n--- Baseline Comparison (10 methods, 5 seeds) ---")
    baseline_results = run_baseline_comparison(verbose=verbose)
    all_results["baselines"] = baseline_results

    # 2. Ablation studies
    if verbose:
        print("\n--- Ablation Studies (5 ablations, 5 seeds) ---")
    ablation_results = run_ablation_studies(verbose=verbose)
    all_results["ablations"] = ablation_results

    # 3. Scaling tests
    if verbose:
        print("\n--- Scaling Tests (N = {3, 4, 6, 8, 12}) ---")
    scaling_results = run_scaling_tests(verbose=verbose)
    all_results["scaling"] = {str(k): v for k, v in scaling_results.items()}

    # Save results
    results_path = os.path.join(output_dir, "phase2_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_all_experiments(verbose=True)
