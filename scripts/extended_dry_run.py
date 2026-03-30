"""
Extended dry run: all 6 trainable methods, 20 episodes each (~1200 steps).
Checks for:
  - No crashes or exceptions
  - No NaN/Inf in losses, rewards, gradients
  - Buffer storage and PPO update work correctly
  - Save/load round-trips produce valid checkpoints
  - Numerical stability over extended runs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import tempfile
import shutil
import traceback

from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
from hcmarl.agents.mappo import MAPPO
from hcmarl.agents.mappo_lag import MAPPOLagrangian
from hcmarl.agents.ippo import IPPO
from hcmarl.agents.hcmarl_agent import HCMARLAgent
from hcmarl.baselines.omnisafe_wrapper import OmniSafeWrapper
from hcmarl.baselines.safepo_wrapper import SafePOWrapper


N_EPISODES = 20
N_WORKERS = 4
MAX_STEPS = 60
DEVICE = "cpu"

METHODS = {
    "mappo": lambda obs, glob, act, n: MAPPO(
        obs_dim=obs, global_obs_dim=glob, n_actions=act, n_agents=n,
        lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
        clip_eps=0.2, entropy_coeff=0.01, max_grad_norm=0.5,
        n_epochs=4, batch_size=64, device=DEVICE,
    ),
    "mappo_lag": lambda obs, glob, act, n: MAPPOLagrangian(
        obs_dim=obs, global_obs_dim=glob, n_actions=act, n_agents=n,
        lr_actor=3e-4, lr_critic=1e-3, lr_lambda=5e-3,
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
        entropy_coeff=0.01, max_grad_norm=0.5,
        n_epochs=4, batch_size=64,
        cost_limit=0.1, lambda_init=0.5, device=DEVICE,
    ),
    "ippo": lambda obs, glob, act, n: IPPO(
        obs_dim=obs, n_actions=act, n_agents=n,
        lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_eps=0.2, entropy_coeff=0.01, max_grad_norm=0.5,
        n_epochs=4, batch_size=64, hidden_dim=64, device=DEVICE,
    ),
    "hcmarl": lambda obs, glob, act, n: HCMARLAgent(
        obs_dim=obs, global_obs_dim=glob, n_actions=act, n_agents=n,
        theta_max={"shoulder": 0.35, "elbow": 0.35, "grip": 0.35},
        ecbf_params={}, device=DEVICE,
    ),
    "omnisafe_ppolag": lambda obs, glob, act, n: OmniSafeWrapper(
        "PPOLag", obs, act, {}, DEVICE,
    ),
    "safepo_macpo": lambda obs, glob, act, n: SafePOWrapper(
        obs, act, n, {}, DEVICE,
    ),
}


def check_finite(name, value):
    """Check that a value contains no NaN or Inf."""
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            return f"{name} is {value}"
    elif isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            return f"{name} has {np.sum(~np.isfinite(value))} non-finite values"
    elif isinstance(value, torch.Tensor):
        if not torch.all(torch.isfinite(value)):
            return f"{name} has non-finite values"
    return None


def run_method(method_name, create_fn):
    """Run a single method for N_EPISODES episodes with full checks."""
    print(f"\n{'='*60}")
    print(f"  {method_name} — {N_EPISODES} episodes x {MAX_STEPS} steps")
    print(f"{'='*60}")

    env = WarehousePettingZoo(n_workers=N_WORKERS, max_steps=MAX_STEPS)
    obs_dim = env.obs_dim
    global_obs_dim = env.global_obs_dim
    n_actions = env.n_tasks

    agent = create_fn(obs_dim, global_obs_dim, n_actions, N_WORKERS)

    is_mappo_lag = isinstance(agent, MAPPOLagrangian)
    is_safepo = isinstance(agent, SafePOWrapper)
    is_omnisafe = isinstance(agent, OmniSafeWrapper)
    is_lagrangian = is_mappo_lag or (is_safepo and hasattr(agent, 'buffer') and agent.buffer is not None)
    is_ippo = isinstance(agent, IPPO)
    is_hcmarl = hasattr(agent, 'mappo') and not is_safepo

    total_steps = 0
    total_rewards = []
    all_losses = []
    errors = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        for step in range(MAX_STEPS):
            global_state = env._get_global_obs()

            # Get actions
            if is_ippo or is_omnisafe:
                result = agent.get_actions(obs)
            else:
                result = agent.get_actions(obs, global_state)
            if isinstance(result, tuple):
                actions = result[0]
                log_probs = result[1] if len(result) > 1 else {}
                value = result[2] if len(result) > 2 else 0.0
                cost_value = result[3] if len(result) > 3 else 0.0
            else:
                actions = result
                log_probs, value, cost_value = {}, 0.0, 0.0

            # Check actions are valid
            for agent_id, action in actions.items():
                err = check_finite(f"ep{ep}_step{step}_{agent_id}_action", action)
                if err:
                    errors.append(err)

            # Step
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            ep_reward += sum(rewards.values())
            ep_steps += 1
            total_steps += 1

            # Check rewards
            for agent_id, r in rewards.items():
                err = check_finite(f"ep{ep}_step{step}_{agent_id}_reward", r)
                if err:
                    errors.append(err)

            # Store transitions
            step_cost = float(sum(info.get("violations", 0) for info in infos.values()) > 0)

            if is_hcmarl and hasattr(agent.mappo, 'buffer'):
                for agent_id in sorted(actions.keys()):
                    agent.mappo.buffer.store(
                        obs=next_obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                        value=value,
                    )
            elif is_lagrangian:
                for agent_id in sorted(actions.keys()):
                    agent.buffer.store(
                        obs=next_obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        cost=step_cost,
                        done=float(terms[agent_id]),
                        value=value,
                        cost_value=cost_value,
                    )
            elif is_ippo:
                for agent_id in sorted(actions.keys()):
                    idx = int(agent_id.split("_")[1])
                    agent.store_transition(
                        agent_idx=idx,
                        obs=next_obs[agent_id],
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                    )
            elif hasattr(agent, 'buffer'):
                for agent_id in sorted(actions.keys()):
                    agent.buffer.store(
                        obs=next_obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                        value=value,
                    )

            obs = next_obs
            if all(terms.values()):
                break

        # PPO update
        update_info = {}
        if hasattr(agent, 'update'):
            update_info = agent.update() or {}
        elif is_hcmarl and hasattr(agent.mappo, 'update'):
            update_info = agent.mappo.update() or {}

        # Check update outputs for NaN
        if isinstance(update_info, dict):
            for k, v in update_info.items():
                if isinstance(v, (int, float)):
                    err = check_finite(f"ep{ep}_update_{k}", v)
                    if err:
                        errors.append(err)
                    all_losses.append(v)

        total_rewards.append(ep_reward)

        if (ep + 1) % 5 == 0:
            avg_r = np.mean(total_rewards[-5:])
            print(f"  Episode {ep+1:3d} | reward={ep_reward:+8.2f} | "
                  f"avg5={avg_r:+8.2f} | steps={ep_steps}")

    # Check gradient norms (for torch-based agents)
    if hasattr(agent, 'actor') or (is_hcmarl and hasattr(agent.mappo, 'actor')):
        actor = agent.actor if hasattr(agent, 'actor') else agent.mappo.actor
        for name, param in actor.named_parameters():
            if param.grad is not None:
                err = check_finite(f"grad_{name}", param.grad)
                if err:
                    errors.append(err)

    # Save/load test
    tmpdir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(tmpdir, f"{method_name}_checkpoint.pt")
        if hasattr(agent, 'save'):
            agent.save(save_path)
            # Reload
            agent2 = create_fn(obs_dim, global_obs_dim, n_actions, N_WORKERS)
            agent2.load(save_path)
            # Quick inference check
            obs, _ = env.reset()
            global_state = env._get_global_obs()
            if is_ippo or is_omnisafe:
                result2 = agent2.get_actions(obs)
            else:
                result2 = agent2.get_actions(obs, global_state)
            print(f"  Save/load: OK (checkpoint {os.path.getsize(save_path)} bytes)")
        else:
            print(f"  Save/load: skipped (no save method)")
    except Exception as e:
        errors.append(f"Save/load failed: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Summary
    print(f"\n  --- Summary ---")
    print(f"  Total steps: {total_steps}")
    print(f"  Reward: mean={np.mean(total_rewards):.2f}, "
          f"std={np.std(total_rewards):.2f}, "
          f"min={np.min(total_rewards):.2f}, "
          f"max={np.max(total_rewards):.2f}")

    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")
        return False
    else:
        print(f"  ERRORS: 0")
        return True


def main():
    print("=" * 60)
    print("HC-MARL Extended Dry Run")
    print(f"{N_EPISODES} episodes x {MAX_STEPS} steps x {N_WORKERS} workers")
    print("=" * 60)

    results = {}
    for method_name, create_fn in METHODS.items():
        try:
            passed = run_method(method_name, create_fn)
            results[method_name] = "PASS" if passed else "FAIL (numerical)"
        except Exception as e:
            print(f"\n  EXCEPTION: {e}")
            traceback.print_exc()
            results[method_name] = f"FAIL (exception: {e})"

    # Final report
    print("\n" + "=" * 60)
    print("EXTENDED DRY RUN RESULTS")
    print("=" * 60)
    all_pass = True
    for method, status in results.items():
        icon = "PASS" if "PASS" in status else "FAIL"
        print(f"  [{icon}] {method}: {status}")
        if "FAIL" in status:
            all_pass = False

    if all_pass:
        print(f"\nALL {len(results)} METHODS PASSED")
    else:
        print(f"\nSOME METHODS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
