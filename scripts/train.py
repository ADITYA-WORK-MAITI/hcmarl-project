"""
HC-MARL Phase 4 (#54): Main Training Script
=============================================
Load config, init env + agent, train loop, checkpoint, log to W&B/CSV.
Optionally runs MMICRL pre-training to discover worker types from real data.

Usage:
    python scripts/train.py --config config/hcmarl_full_config.yaml --seed 0
    python scripts/train.py --config config/mappo_config.yaml --seed 0 --method mappo
    python scripts/train.py --config config/hcmarl_full_config.yaml --seed 0 --device cuda
    python scripts/train.py --config config/hcmarl_full_config.yaml --seed 0 --mmicrl --pamap2-path /path/to/PAMAP2/Protocol
"""

import argparse
import os
import sys
import time
import yaml
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
from hcmarl.agents.mappo import MAPPO
from hcmarl.agents.mappo_lag import MAPPOLagrangian
from hcmarl.agents.ippo import IPPO
from hcmarl.agents.hcmarl_agent import HCMARLAgent
from hcmarl.baselines.omnisafe_wrapper import OmniSafeWrapper
from hcmarl.baselines.safepo_wrapper import SafePOWrapper
from hcmarl.logger import HCMARLLogger


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "hcmarl": "HC-MARL (MAPPO + ECBF + NSWF)",
    "mappo": "MAPPO (no safety filter)",
    "ippo": "IPPO (independent, no centralised critic)",
    "mappo_lag": "MAPPO-Lagrangian (cost critic + dual variable)",
    "ppo_lag": "PPO-Lagrangian (OmniSafe)",
    "cpo": "CPO (OmniSafe)",
    "macpo": "MACPO (SafePO)",
}


def create_agent(method, obs_dim, global_obs_dim, n_actions, n_agents, cfg, device):
    """Instantiate the correct agent class based on method name."""
    algo = cfg.get("algorithm", {})
    lr_actor = algo.get("lr_actor", 3e-4)
    lr_critic = algo.get("lr_critic", 1e-3)
    gamma = algo.get("gamma", 0.99)
    gae_lambda = algo.get("gae_lambda", 0.95)
    clip_eps = algo.get("clip_eps", 0.2)
    entropy_coeff = algo.get("entropy_coeff", 0.01)
    max_grad_norm = algo.get("max_grad_norm", 0.5)
    n_epochs = algo.get("n_epochs", 10)
    batch_size = algo.get("batch_size", 256)
    hidden_dim = algo.get("hidden_dim", 64)

    if method == "hcmarl":
        return HCMARLAgent(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            theta_max=cfg.get("environment", {}).get("theta_max", {}),
            ecbf_params=cfg.get("ecbf", {}),
            device=device,
        )
    elif method == "mappo":
        return MAPPO(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            lr_actor=lr_actor, lr_critic=lr_critic,
            gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
            entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size, device=device,
        )
    elif method == "ippo":
        return IPPO(
            obs_dim=obs_dim, n_actions=n_actions, n_agents=n_agents,
            lr=lr_actor, gamma=gamma, gae_lambda=gae_lambda,
            clip_eps=clip_eps, entropy_coeff=entropy_coeff,
            max_grad_norm=max_grad_norm, n_epochs=n_epochs,
            batch_size=batch_size, hidden_dim=hidden_dim, device=device,
        )
    elif method == "mappo_lag":
        return MAPPOLagrangian(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            lr_actor=lr_actor, lr_critic=lr_critic,
            lr_lambda=algo.get("lambda_lr", 5e-3),
            gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
            entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            cost_limit=algo.get("cost_limit", 0.1),
            lambda_init=algo.get("lambda_init", 0.5),
            device=device,
        )
    elif method in ("ppo_lag", "cpo"):
        algo_name = "PPOLag" if method == "ppo_lag" else "CPO"
        return OmniSafeWrapper(algo_name, obs_dim, n_actions, cfg, device)
    elif method == "macpo":
        return SafePOWrapper(obs_dim, n_actions, n_agents, cfg, device)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# MMICRL pre-training
# ---------------------------------------------------------------------------

def run_mmicrl_pretrain(cfg, pamap2_path=None, log_dir="logs"):
    """
    Run MMICRL type discovery from real PAMAP2 data.
    Returns learned per-type theta_max thresholds.
    """
    from hcmarl.mmicrl import DemonstrationCollector, MMICRL

    mmicrl_cfg = cfg.get("mmicrl", {})
    n_types = mmicrl_cfg.get("n_types", 3)
    n_muscles = mmicrl_cfg.get("n_muscles", 3)

    collector = DemonstrationCollector(n_muscles=n_muscles)

    if pamap2_path:
        print(f"Loading PAMAP2 real data from: {pamap2_path}")
        n_loaded = collector.load_real_demos(
            source="pamap2",
            path=pamap2_path,
            window_size=mmicrl_cfg.get("window_size", 256),
            stride=mmicrl_cfg.get("stride", 128),
        )
        print(f"  Loaded {n_loaded} trajectory windows from PAMAP2")
    else:
        # No real data path provided — collect from env with random policy
        print("No PAMAP2 path provided. Collecting demos from env with random policy...")
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env_cfg = cfg.get("environment", {})
        env = WarehousePettingZoo(
            n_workers=env_cfg.get("n_workers", 6),
            max_steps=env_cfg.get("max_steps", 480),
        )

        class RandomPolicy:
            def __init__(self, n_tasks):
                self.n_tasks = n_tasks
            def get_actions(self, obs):
                return {agent_id: np.random.randint(0, self.n_tasks) for agent_id in obs}

        n_loaded = collector.collect_from_env(
            env, RandomPolicy(env.n_tasks),
            n_episodes=mmicrl_cfg.get("n_collection_episodes", 50),
        )
        print(f"  Collected {n_loaded} episodes from environment")

    # Fit MMICRL
    mmicrl = MMICRL(
        n_types=n_types,
        lambda1=mmicrl_cfg.get("lambda1", 1.0),
        lambda2=mmicrl_cfg.get("lambda2", 1.0),
        n_muscles=n_muscles,
    )
    results = mmicrl.fit(collector)

    # Print results
    print(f"\n--- MMICRL Type Discovery Results ---")
    print(f"  Demonstrations: {results['n_demonstrations']}")
    print(f"  Types discovered: {results['n_types_discovered']}")
    print(f"  Type proportions: {[f'{p:.2f}' for p in results['type_proportions']]}")
    print(f"  Mutual information I(tau;z): {results['mutual_information']:.4f}")
    print(f"  Objective value: {results['objective_value']:.4f}")
    print(f"  Learned thresholds per type:")
    for k, thetas in results['theta_per_type'].items():
        print(f"    Type {k}: {{{', '.join(f'{m}: {v:.3f}' for m, v in thetas.items())}}}")

    # Save MMICRL results
    mmicrl_log_dir = os.path.join(log_dir, "mmicrl")
    os.makedirs(mmicrl_log_dir, exist_ok=True)
    with open(os.path.join(mmicrl_log_dir, "mmicrl_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {mmicrl_log_dir}/mmicrl_results.json")

    return results, mmicrl


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg, method, seed, device, resume_from=None, mmicrl_results=None, mmicrl_model=None,
          ecbf_mode="on"):
    """Full training loop with logging and checkpointing."""
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg = cfg.get("environment", {})
    train_cfg = cfg.get("training", {})
    n_workers = env_cfg.get("n_workers", 6)
    max_steps = env_cfg.get("max_steps", 480)

    total_steps = train_cfg.get("total_steps", 5_000_000)
    eval_interval = train_cfg.get("eval_interval", 50_000)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 100_000)
    n_eval_episodes = train_cfg.get("n_eval_episodes", 10)

    # If MMICRL discovered thresholds, inject them into config
    theta_max = env_cfg.get("theta_max", None)
    if mmicrl_results and method == "hcmarl":
        theta_per_type = mmicrl_results.get("theta_per_type", {})
        type_proportions = mmicrl_results.get("type_proportions", [])
        if theta_per_type:
            type_keys = sorted(theta_per_type.keys(), key=lambda k: int(k))
            n_types = len(type_keys)
            # Proportional assignment: distribute workers according to
            # discovered type proportions (not round-robin)
            theta_max = {}
            if type_proportions and len(type_proportions) == n_types:
                # Assign workers proportionally to type prevalence
                counts = np.round(np.array(type_proportions) * n_workers).astype(int)
                # Fix rounding so total == n_workers
                diff = n_workers - counts.sum()
                counts[np.argmax(counts)] += diff
                worker_type_map = []
                for t_idx, count in enumerate(counts):
                    worker_type_map.extend([t_idx] * count)
                for w in range(n_workers):
                    type_k = type_keys[worker_type_map[w]]
                    theta_max[f"worker_{w}"] = theta_per_type[type_k]
            else:
                # Fallback: assign most conservative (lowest) thresholds
                conservative = {}
                for type_k in type_keys:
                    for muscle, val in theta_per_type[type_k].items():
                        if muscle not in conservative or val < conservative[muscle]:
                            conservative[muscle] = val
                theta_max = {f"worker_{w}": dict(conservative) for w in range(n_workers)}
            print(f"Using MMICRL-learned thresholds for {n_workers} workers ({n_types} types, proportional assignment)")

    # Directories
    run_name = f"{method}_seed{seed}"
    ckpt_dir = os.path.join("checkpoints", method, f"seed_{seed}")
    log_dir = os.path.join("logs", method, f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Environment
    env = WarehousePettingZoo(
        n_workers=n_workers, max_steps=max_steps,
        theta_max=theta_max, ecbf_mode=ecbf_mode,
    )
    obs_dim = env.obs_dim
    global_obs_dim = env.global_obs_dim
    n_actions = env.n_tasks

    # Agent
    agent = create_agent(method, obs_dim, global_obs_dim, n_actions, n_workers, cfg, device)

    # Resume
    if resume_from and os.path.exists(resume_from):
        agent.load(resume_from)
        print(f"Resumed from {resume_from}")

    # Logger
    logger = HCMARLLogger(
        log_dir=log_dir,
        use_wandb=cfg.get("logging", {}).get("use_wandb", False),
        wandb_project=cfg.get("logging", {}).get("project_name", "hcmarl"),
        run_name=run_name,
        config=cfg,
    )

    # Save config (with MMICRL results if available)
    save_cfg = dict(cfg)
    if mmicrl_results:
        save_cfg["mmicrl_results"] = mmicrl_results
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(save_cfg, f)

    # Detect agent type for buffer storage
    is_mappo_lag = isinstance(agent, MAPPOLagrangian)
    is_safepo = isinstance(agent, SafePOWrapper)
    is_lagrangian = is_mappo_lag or (is_safepo and hasattr(agent, 'buffer') and agent.buffer is not None)
    is_ippo = isinstance(agent, IPPO)
    is_hcmarl = hasattr(agent, 'mappo') and not is_safepo

    # Training
    global_step = 0
    episode_count = 0
    best_reward = -float("inf")
    cost_ema = 0.0  # Exponential moving average of per-step cost rate
    next_checkpoint_step = checkpoint_interval
    start_time = time.time()

    print(f"{'='*60}")
    print(f"HC-MARL Training: {METHODS.get(method, method)}")
    print(f"Seed: {seed} | Device: {device} | Workers: {n_workers}")
    print(f"Total steps: {total_steps:,} | Eval every: {eval_interval:,}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"{'='*60}")

    while global_step < total_steps:
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0
        tasks_per_worker = np.zeros(n_workers)
        peak_mf = 0.0
        safe_steps = 0
        forced_rests = 0
        total_ecbf_interventions = 0
        total_ecbf_opportunities = 0  # muscles where C_nominal > 0

        for step in range(max_steps):
            global_state = env._get_global_obs()

            # Get actions from agent
            result = agent.get_actions(obs, global_state) if not is_ippo else agent.get_actions(obs)
            if isinstance(result, tuple):
                actions = result[0]
                log_probs = result[1] if len(result) > 1 else {}
                value = result[2] if len(result) > 2 else 0.0
                cost_value = result[3] if len(result) > 3 else 0.0
            else:
                actions = result
                log_probs, value, cost_value = {}, 0.0, 0.0

            # Step environment
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            episode_steps += 1
            global_step += 1

            # Track metrics
            step_violations = 0
            for i, (agent_id, info) in enumerate(sorted(infos.items())):
                fatigue = info.get("fatigue", {})
                task = info.get("task", "rest")
                violations = info.get("violations", 0)
                step_violations += violations
                if task != "rest":
                    tasks_per_worker[i] += 1
                else:
                    avg_mf = np.mean(list(fatigue.values())) if fatigue else 0
                    if avg_mf > 0.3:
                        forced_rests += 1
                for m, mf in fatigue.items():
                    peak_mf = max(peak_mf, mf)
                # ECBF intervention tracking for SAI
                ecbf_int = info.get("ecbf_interventions", 0)
                total_ecbf_interventions += ecbf_int
                # Count muscles with nonzero demand as intervention opportunities
                if task != "rest":
                    total_ecbf_opportunities += len(fatigue)

            episode_cost += step_violations
            if step_violations == 0:
                safe_steps += 1

            # Store transitions based on agent type
            step_cost = float(step_violations > 0)

            if is_hcmarl and hasattr(agent.mappo, 'buffer'):
                # HCMARLAgent wraps MAPPO — store obs used to SELECT action
                for agent_id in sorted(actions.keys()):
                    agent.mappo.buffer.store(
                        obs=obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                        value=value,
                    )
            elif is_lagrangian:
                # MAPPO-Lagrangian or SafePO fallback: store with cost
                for agent_id in sorted(actions.keys()):
                    agent.buffer.store(
                        obs=obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        cost=step_cost,
                        done=float(terms[agent_id]),
                        value=value,
                        cost_value=cost_value,
                    )
            elif is_ippo:
                # IPPO: per-agent buffer storage
                for agent_id in sorted(actions.keys()):
                    idx = int(agent_id.split("_")[1])
                    agent.store_transition(
                        agent_idx=idx,
                        obs=obs[agent_id],
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                    )
            elif hasattr(agent, 'buffer'):
                # Generic MAPPO / other agents with buffer
                for agent_id in sorted(actions.keys()):
                    agent.buffer.store(
                        obs=obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                        value=value,
                    )

            obs = next_obs

            if all(terms.values()):
                break

        # PPO update at end of episode
        update_info = {}
        if hasattr(agent, 'update'):
            update_info = agent.update() or {}
        elif is_hcmarl and hasattr(agent.mappo, 'update'):
            update_info = agent.mappo.update() or {}

        # Lagrangian lambda update with EMA-smoothed cost
        if hasattr(agent, 'update_lambda'):
            mean_cost = episode_cost / max(1, episode_steps)
            ema_alpha = 0.05  # smoothing factor — slow update for stable dual variable
            cost_ema = (1 - ema_alpha) * cost_ema + ema_alpha * mean_cost
            agent.update_lambda(cost_ema)

        # Compute episode metrics
        episode_count += 1
        n = n_workers
        jain = float((tasks_per_worker.sum()**2) / (n * (tasks_per_worker**2).sum() + 1e-8)) if tasks_per_worker.sum() > 0 else 1.0

        # Safety Autonomy Index: fraction of work-muscle slots NOT clipped by ECBF
        sai = 1.0 - (total_ecbf_interventions / max(1, total_ecbf_opportunities))

        ep_metrics = {
            "episode": episode_count,
            "global_step": global_step,
            "cumulative_reward": episode_reward,
            "cumulative_cost": episode_cost,
            "violation_rate": episode_cost / max(1, episode_steps * n_workers),
            "safety_rate": safe_steps / max(1, episode_steps),
            "tasks_completed": float(tasks_per_worker.sum()),
            "jain_fairness": jain,
            "peak_fatigue": peak_mf,
            "forced_rest_rate": forced_rests / max(1, episode_steps * n_workers),
            "safety_autonomy_index": sai,
            "ecbf_interventions": total_ecbf_interventions,
            "wall_time": time.time() - start_time,
        }
        if is_lagrangian:
            ep_metrics["lambda"] = agent.lam
        ep_metrics.update(update_info)

        logger.log_episode(ep_metrics)

        # Print progress
        if episode_count % 50 == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            lam_str = f" lam={agent.lam:.3f}" if is_lagrangian else ""
            print(
                f"[Ep {episode_count:5d} | Step {global_step:>9,}/{total_steps:,} | "
                f"{sps:.0f} sps] "
                f"R={episode_reward:7.1f} C={episode_cost:4.0f} "
                f"Safe={ep_metrics['safety_rate']:.2%} SAI={sai:.3f} "
                f"Jain={jain:.3f} PeakMF={peak_mf:.3f}{lam_str}"
            )

        # Checkpoint at regular intervals
        if global_step >= next_checkpoint_step:
            if hasattr(agent, 'save'):
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{global_step}.pt")
                agent.save(ckpt_path)
            next_checkpoint_step += checkpoint_interval

        # Best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            if hasattr(agent, 'save'):
                agent.save(os.path.join(ckpt_dir, "checkpoint_best.pt"))

    # Final checkpoint
    if hasattr(agent, 'save'):
        agent.save(os.path.join(ckpt_dir, "checkpoint_final.pt"))

    # Save final metrics summary
    elapsed = time.time() - start_time
    summary = {
        "method": method,
        "seed": seed,
        "total_episodes": episode_count,
        "total_steps": global_step,
        "best_reward": best_reward,
        "wall_time_seconds": elapsed,
        "wall_time_hours": elapsed / 3600,
    }
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete: {episode_count} episodes, {global_step:,} steps")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Wall time: {elapsed/3600:.2f} hours")
    print(f"Checkpoint: {ckpt_dir}/checkpoint_final.pt")
    print(f"{'='*60}")

    logger.close()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HC-MARL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--method", type=str, default="hcmarl", choices=list(METHODS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mmicrl", action="store_true", help="Run MMICRL pre-training before RL")
    parser.add_argument("--pamap2-path", type=str, default=None, help="Path to PAMAP2 Protocol/ directory")
    parser.add_argument("--ecbf-mode", type=str, default="on", choices=["on", "off"],
                        help="ECBF safety filter mode: 'on' (default) or 'off' (ablation)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # MMICRL pre-training
    mmicrl_results = None
    mmicrl_model = None
    if args.mmicrl:
        mmicrl_results, mmicrl_model = run_mmicrl_pretrain(cfg, args.pamap2_path)

    train(cfg, args.method, args.seed, device, args.resume, mmicrl_results, mmicrl_model,
          ecbf_mode=args.ecbf_mode)


if __name__ == "__main__":
    main()
