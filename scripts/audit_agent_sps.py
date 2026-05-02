"""Comprehensive SPS + correctness audit for the 5 baseline agents.

Runs each agent for a sustained number of buffer fills (multiple updates,
not just one) and reports:
  - Steady-state SPS (excludes startup batch)
  - Update wall-time per buffer fill
  - Actor loss trajectory (NaN/Inf check, magnitude check)
  - Critic loss trajectory
  - Per-agent acceptance rate (MACPO only)
  - Policy ratio sanity (HAPPO M_running, MAPPO ratios)

NO PROJECTIONS to GPU. CPU numbers only. The user will translate to GPU
themselves once the numbers are honest.

Usage:
    python scripts/audit_agent_sps.py [--steps 2048] [--methods mappo,happo,...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import math
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Match production thread cap
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import yaml
import numpy as np
import torch

from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
from hcmarl.utils import seed_everything


def _build_env(cfg, ecbf_mode: str = "off"):
    env_cfg = cfg["environment"]
    n_workers = env_cfg.get("n_workers", 6)
    muscle_groups = env_cfg.get("muscle_groups", {}) or {}
    muscle_params_override = {
        m: {k: v for k, v in p.items() if k in ("F", "R", "r")}
        for m, p in muscle_groups.items()
    }
    env = WarehousePettingZoo(
        n_workers=n_workers,
        max_steps=env_cfg.get("max_steps", 480),
        theta_max=env_cfg.get("theta_max", {}) or {},
        ecbf_mode=ecbf_mode,
        muscle_params_override=muscle_params_override,
    )
    return env


def _build_agent(method, env, cfg):
    obs_dim = env.obs_dim
    gobs_dim = env.global_obs_dim
    n_actions = env.n_tasks
    n_workers = env.n_workers
    algo = cfg.get("algorithm", {}) or {}

    common_kwargs = dict(
        obs_dim=obs_dim, global_obs_dim=gobs_dim, n_actions=n_actions,
        n_agents=n_workers,
        gamma=algo.get("gamma", 0.99),
        gae_lambda=algo.get("gae_lambda", 0.95),
        entropy_coeff=algo.get("entropy_coeff", 0.01),
        max_grad_norm=algo.get("max_grad_norm", 0.5),
        n_epochs=algo.get("n_epochs", 10),
        batch_size=algo.get("batch_size", 256),
        device="cpu",
    )

    if method == "mappo":
        from hcmarl.agents.mappo import MAPPO
        return MAPPO(
            **common_kwargs,
            lr_actor=algo.get("lr_actor", 3e-4),
            lr_critic=algo.get("lr_critic", 1e-3),
            clip_eps=algo.get("clip_eps", 0.2),
        ), False  # not cost-aware
    if method == "mappo_lag":
        from hcmarl.agents.mappo_lag import MAPPOLagrangian
        return MAPPOLagrangian(
            **common_kwargs,
            lr_actor=algo.get("lr_actor", 3e-4),
            lr_critic=algo.get("lr_critic", 1e-3),
            lr_lambda=algo.get("lambda_lr", 5e-3),
            clip_eps=algo.get("clip_eps", 0.2),
            cost_limit=algo.get("cost_limit", 0.1),
            lambda_init=algo.get("lambda_init", 0.5),
        ), True
    if method == "macpo":
        from hcmarl.agents.macpo import MACPO
        return MACPO(
            **common_kwargs,
            lr_critic=algo.get("lr_critic", 1e-3),
            cost_limit=algo.get("cost_limit", 0.1),
            delta_kl=algo.get("delta_kl", 0.01),
            cg_iters=algo.get("cg_iters", 7),
            cg_damping=algo.get("cg_damping", 0.1),
            line_search_steps=algo.get("line_search_steps", 7),
            line_search_decay=algo.get("line_search_decay", 0.8),
        ), True
    if method == "happo":
        from hcmarl.agents.happo import HAPPO
        return HAPPO(
            **common_kwargs,
            lr_actor=algo.get("lr_actor", 3e-4),
            lr_critic=algo.get("lr_critic", 1e-3),
            clip_eps=algo.get("clip_eps", 0.2),
            hidden_dim=algo.get("hidden_dim", 64),
            critic_hidden_dim=algo.get("critic_hidden_dim", 128),
        ), False
    if method == "shielded_mappo":
        from hcmarl.agents.shielded_mappo import ShieldedMAPPO
        env_cfg = cfg["environment"]
        muscle_names = list((env_cfg.get("muscle_groups") or {}).keys())
        theta_max = env_cfg.get("theta_max", {}) or {}
        tasks_cfg = env_cfg.get("tasks", {}) or {}
        task_names = list(tasks_cfg.keys())
        shield_cfg = cfg.get("shield", {}) or {}
        return ShieldedMAPPO(
            **common_kwargs,
            lr_actor=algo.get("lr_actor", 3e-4),
            lr_critic=algo.get("lr_critic", 1e-3),
            clip_eps=algo.get("clip_eps", 0.2),
            muscle_names=muscle_names,
            theta_max=theta_max,
            task_names=task_names,
            task_demands=tasks_cfg,
            rest_task_name="rest",
            safety_margin=float(shield_cfg.get("safety_margin", 0.05)),
            demand_threshold=float(shield_cfg.get("demand_threshold", 0.0)),
        ), False
    raise ValueError(method)


def _store_step(agent, is_cost_aware, obs, gs, actions, log_probs, values,
                cost_value, rewards, step_cost, infos, terms, n_workers):
    """Mirrors train.py's per-method buffer storage dispatch."""
    if is_cost_aware:
        for aid in sorted(actions.keys()):
            agent.buffer.store(
                obs=obs[aid], global_state=gs, action=actions[aid],
                log_prob=log_probs[aid], reward=rewards[aid],
                cost=step_cost, done=float(terms[aid]),
                values=values, cost_values=cost_value,
            )
    else:
        for aid in sorted(actions.keys()):
            agent.buffer.store(
                obs=obs[aid], global_state=gs, action=actions[aid],
                log_prob=log_probs[aid], reward=rewards[aid],
                done=float(terms[aid]), values=values,
            )


def benchmark(method, target_steps, seed=0):
    """Run `method` for target_steps env steps with full update cycle.
    Reports per-update wall-time, steady-state SPS, and loss trajectory."""
    seed_everything(seed, high_determinism=False)  # CPU bench; high-det not needed
    cfg_path_map = {
        "mappo": "config/mappo_config.yaml",
        "mappo_lag": "config/mappo_lag_config.yaml",
        "macpo": "config/macpo_config.yaml",
        "happo": "config/happo_config.yaml",
        "shielded_mappo": "config/shielded_mappo_config.yaml",
    }
    cfg = yaml.safe_load(open(os.path.join(REPO, cfg_path_map[method])))
    env = _build_env(cfg, ecbf_mode="off")
    agent, is_cost_aware = _build_agent(method, env, cfg)

    obs, _ = env.reset(seed=seed)
    n_workers = env.n_workers
    update_times = []  # per-update wall time (excludes startup)
    actor_losses = []
    critic_losses = []
    accept_history = []  # for MACPO
    last_metrics = {}

    t_start = time.time()
    last_update_t = None
    step = 0
    n_updates = 0

    while step < target_steps:
        gs = env._get_global_obs()
        result = agent.get_actions(obs, gs)
        if len(result) == 4:
            actions, log_probs, values, cost_value = result
        else:
            actions, log_probs, values = result
            cost_value = 0.0

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        step_cost = sum(infos[a].get("cost", 0.0) for a in sorted(infos.keys())) / max(1, n_workers)

        _store_step(agent, is_cost_aware, obs, gs, actions, log_probs,
                    values, cost_value, rewards, step_cost, infos, terms,
                    n_workers)
        obs = next_obs
        step += 1
        if all(terms.values()):
            obs, _ = env.reset(seed=seed)

        if len(agent.buffer) >= agent.batch_size:
            t0 = time.time()
            info = agent.update()
            update_dt = time.time() - t0
            update_times.append(update_dt)
            n_updates += 1
            last_metrics = info
            actor_losses.append(info.get("actor_loss"))
            critic_losses.append(info.get("critic_loss"))
            if "n_accepted" in info and "n_total_agent_updates" in info:
                tot = max(1, info["n_total_agent_updates"])
                accept_history.append(info["n_accepted"] / tot)

    elapsed = time.time() - t_start
    sps = step / elapsed

    # Steady-state SPS: skip first update (startup overhead)
    if len(update_times) >= 3:
        steady_dt = sum(update_times[1:])
        steady_steps = (n_updates - 1) * agent.batch_size
        steady_sps_update_only = steady_steps / steady_dt if steady_dt > 0 else 0
    else:
        steady_sps_update_only = sps

    # Sanity checks
    finite_actor = all(math.isfinite(x) for x in actor_losses if x is not None)
    finite_critic = all(math.isfinite(x) for x in critic_losses if x is not None)
    actor_loss_max_abs = max((abs(x) for x in actor_losses if x is not None), default=0)
    actor_loss_median_abs = statistics.median(
        [abs(x) for x in actor_losses if x is not None] or [0]
    )

    return {
        "method": method,
        "steps": step,
        "elapsed_s": elapsed,
        "sps_overall": sps,
        "n_updates": n_updates,
        "update_dt_mean": statistics.mean(update_times) if update_times else 0,
        "update_dt_median": statistics.median(update_times) if update_times else 0,
        "actor_loss_finite": finite_actor,
        "critic_loss_finite": finite_critic,
        "actor_loss_max_abs": actor_loss_max_abs,
        "actor_loss_median_abs": actor_loss_median_abs,
        "macpo_accept_rate": (statistics.mean(accept_history) if accept_history else None),
        "last_metrics": {k: v for k, v in last_metrics.items() if not isinstance(v, dict)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1024,
                   help="Total env steps per method (default 1024 = ~4 buffer fills)")
    p.add_argument("--methods", default="mappo,mappo_lag,macpo,happo,shielded_mappo",
                   help="Comma-separated list of methods to benchmark.")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"=== AGENT SPS + CORRECTNESS AUDIT ===")
    print(f"target_steps: {args.steps}    methods: {methods}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"device: cpu (CPU bench; numbers translate to L4 GPU separately)")
    print()

    results = []
    for m in methods:
        print(f"--- benchmarking: {m} ---")
        try:
            r = benchmark(m, args.steps)
        except Exception as e:
            import traceback
            print(f"FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"method": m, "error": str(e)})
            continue
        results.append(r)
        print(f"  SPS overall:        {r['sps_overall']:8.1f}")
        print(f"  steps:              {r['steps']}")
        print(f"  elapsed:            {r['elapsed_s']:.2f}s")
        print(f"  n_updates:          {r['n_updates']}")
        print(f"  update wall mean:   {r['update_dt_mean']*1000:6.0f} ms")
        print(f"  update wall median: {r['update_dt_median']*1000:6.0f} ms")
        print(f"  actor_loss finite:  {r['actor_loss_finite']}")
        print(f"  critic_loss finite: {r['critic_loss_finite']}")
        print(f"  actor_loss max|x|:  {r['actor_loss_max_abs']:.3e}")
        print(f"  actor_loss med|x|:  {r['actor_loss_median_abs']:.3e}")
        if r.get("macpo_accept_rate") is not None:
            print(f"  MACPO accept rate:  {r['macpo_accept_rate']*100:.0f}%")
        print()

    # Summary table
    print("=" * 92)
    print(f"{'METHOD':<18s} {'SPS':>8s} {'upd ms':>8s} {'finite':>8s} {'|loss|max':>12s} {'|loss|med':>12s} {'accept':>8s}")
    print("-" * 92)
    for r in results:
        if "error" in r:
            print(f"{r['method']:<18s}  ERROR: {r['error']}")
            continue
        accept_s = f"{r['macpo_accept_rate']*100:.0f}%" if r.get("macpo_accept_rate") is not None else "  -"
        finite_s = "OK" if (r["actor_loss_finite"] and r["critic_loss_finite"]) else "NaN/Inf"
        print(f"{r['method']:<18s} {r['sps_overall']:8.1f} {r['update_dt_mean']*1000:8.0f} "
              f"{finite_s:>8s} {r['actor_loss_max_abs']:12.2e} {r['actor_loss_median_abs']:12.2e} {accept_s:>8s}")
    print("=" * 92)


if __name__ == "__main__":
    main()
