"""
HC-MARL Phase 4 (#54): Main Training Script
=============================================
Load config, init env + agent, train loop, checkpoint, log to W&B/CSV.

Usage:
    python scripts/train.py --config config/hcmarl_full_config.yaml --seed 0
    python scripts/train.py --config config/mappo_config.yaml --seed 0 --method mappo
    python scripts/train.py --config config/hcmarl_full_config.yaml --seed 0 --device cuda
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
    critic_hidden_dim = algo.get("critic_hidden_dim", 128)

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
            lr=lr_actor, gamma=gamma, clip_eps=clip_eps,
            hidden_dim=hidden_dim, device=device,
        )
    elif method == "mappo_lag":
        return MAPPOLagrangian(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            lr_actor=lr_actor, lr_critic=lr_critic,
            lr_lambda=algo.get("lambda_lr", 5e-3),
            gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
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
# Training loop
# ---------------------------------------------------------------------------

def train(cfg, method, seed, device, resume_from=None):
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

    # Directories
    run_name = f"{method}_seed{seed}"
    ckpt_dir = os.path.join("checkpoints", method, f"seed_{seed}")
    log_dir = os.path.join("logs", method, f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Environment
    env = WarehousePettingZoo(
        n_workers=n_workers, max_steps=max_steps,
        theta_max=env_cfg.get("theta_max", None),
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

    # Save config
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # Training
    global_step = 0
    episode_count = 0
    best_reward = -float("inf")
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

        for step in range(max_steps):
            global_state = env._get_global_obs()

            # Get actions from agent
            result = agent.get_actions(obs, global_state)
            if isinstance(result, tuple) and len(result) >= 3:
                actions = result[0]
                log_probs = result[1] if len(result) > 1 else {}
                value = result[2] if len(result) > 2 else 0.0
            else:
                actions = result
                log_probs, value = {}, 0.0

            # Step environment
            obs, rewards, terms, truncs, infos = env.step(actions)
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

            episode_cost += step_violations
            if step_violations == 0:
                safe_steps += 1

            # Store transition for on-policy methods
            if hasattr(agent, 'mappo') and hasattr(agent.mappo, 'buffer'):
                for agent_id in sorted(actions.keys()):
                    idx = int(agent_id.split("_")[1])
                    agent.mappo.buffer.store(
                        obs=obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                        value=value,
                    )
            elif hasattr(agent, 'buffer'):
                for agent_id in sorted(actions.keys()):
                    agent.buffer.store(
                        obs=obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                        value=value,
                    )

            if all(terms.values()):
                break

        # PPO update at end of episode
        update_info = {}
        if hasattr(agent, 'update'):
            update_info = agent.update() or {}
        elif hasattr(agent, 'mappo') and hasattr(agent.mappo, 'update'):
            update_info = agent.mappo.update() or {}

        # Lagrangian lambda update
        if hasattr(agent, 'update_lambda'):
            mean_cost = episode_cost / max(1, episode_steps)
            agent.update_lambda(mean_cost)

        # Compute episode metrics
        episode_count += 1
        n = n_workers
        jain = float((tasks_per_worker.sum()**2) / (n * (tasks_per_worker**2).sum() + 1e-8)) if tasks_per_worker.sum() > 0 else 1.0

        ep_metrics = {
            "episode": episode_count,
            "global_step": global_step,
            "cumulative_reward": episode_reward,
            "cumulative_cost": episode_cost,
            "violation_rate": episode_cost / max(1, episode_steps * n_workers * env.n_muscles),
            "safety_rate": safe_steps / max(1, episode_steps),
            "tasks_completed": float(tasks_per_worker.sum()),
            "jain_fairness": jain,
            "peak_fatigue": peak_mf,
            "forced_rest_rate": forced_rests / max(1, episode_steps * n_workers),
            "constraint_recovery_time": 0.0,
            "wall_time": time.time() - start_time,
        }
        ep_metrics.update(update_info)

        logger.log_episode(ep_metrics)

        # Print progress
        if episode_count % 50 == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            print(
                f"[Ep {episode_count:5d} | Step {global_step:>9,}/{total_steps:,} | "
                f"{sps:.0f} sps] "
                f"R={episode_reward:7.1f} C={episode_cost:4.0f} "
                f"Safe={ep_metrics['safety_rate']:.2%} "
                f"Jain={jain:.3f} PeakMF={peak_mf:.3f}"
            )

        # Checkpoint
        if global_step >= checkpoint_interval * (global_step // checkpoint_interval):
            if hasattr(agent, 'save'):
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{global_step}.pt")
                agent.save(ckpt_path)

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

    train(cfg, args.method, args.seed, device, args.resume)


if __name__ == "__main__":
    main()
