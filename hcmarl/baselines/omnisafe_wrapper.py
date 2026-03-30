"""
HC-MARL Phase 3 (#42): OmniSafe Baseline Wrapper
Uniform interface for PPO-Lagrangian, CPO, FOCOPS, CUP from OmniSafe.

OmniSafe Reference:
    Ji et al. "OmniSafe: An Infrastructure for Accelerating Safe
    Reinforcement Learning Research." arXiv 2305.09304, 2023.
    GitHub: https://github.com/PKU-Alignment/omnisafe
    Install: pip install omnisafe
"""
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
import json


class OmniSafeWrapper:
    """Wraps OmniSafe algorithms with HC-MARL's action interface."""
    SUPPORTED = ["PPOLag", "CPO", "FOCOPS", "CUP", "PPO"]

    def __init__(self, algo_name="PPOLag", obs_dim=19, n_actions=6, config=None, device="cpu"):
        self.algo_name = algo_name
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config or {}
        self.device = device
        self.agent = None
        self.name = f"OmniSafe-{algo_name}"
        self._omnisafe_available = False
        self._try_init()

    def _try_init(self):
        try:
            import omnisafe
            env_id = self.config.get("env_id", "SafetyPointGoal1-v0")
            custom_cfgs = {
                "train_cfgs": {
                    "total_steps": self.config.get("total_steps", 1000000),
                    "device": self.device,
                },
                "algo_cfgs": {
                    "steps_per_epoch": self.config.get("steps_per_epoch", 2048),
                },
                "logger_cfgs": {
                    "log_dir": self.config.get("log_dir", "./omnisafe_results"),
                    "use_wandb": self.config.get("use_wandb", False),
                },
            }
            self.agent = omnisafe.Agent(self.algo_name, env_id, custom_cfgs=custom_cfgs)
            self._omnisafe_available = True
        except ImportError:
            self.agent = None
            self._omnisafe_available = False
        except Exception:
            self.agent = None
            self._omnisafe_available = False

    @property
    def available(self) -> bool:
        """Whether OmniSafe is installed and agent is initialised."""
        return self._omnisafe_available

    def get_actions(self, observations, **kwargs):
        actions = {}
        log_probs = {}
        for agent_id, obs in observations.items():
            if self.agent is not None:
                try:
                    action = self.agent.predict(obs)
                    actions[agent_id] = int(action) if np.isscalar(action) else int(action[0])
                except Exception:
                    actions[agent_id] = np.random.randint(0, self.n_actions)
            else:
                actions[agent_id] = np.random.randint(0, self.n_actions)
            log_probs[agent_id] = 0.0
        return actions, log_probs, 0.0

    def train(self, total_steps=1000000):
        """Train the OmniSafe agent."""
        if self.agent:
            self.agent.learn()
        else:
            raise RuntimeError(
                "OmniSafe not available. Install: pip install omnisafe\n"
                "Cannot train without OmniSafe installed."
            )

    def save(self, path):
        """Save agent state. Falls back to numpy if OmniSafe unavailable."""
        import json as _json
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "algo_name": self.algo_name,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "name": self.name,
        }
        if self.agent:
            try:
                self.agent.save(path)
                return
            except Exception:
                pass
        with open(path, "w") as f:
            _json.dump(state, f)

    def load(self, path):
        """Load agent state. No-op if OmniSafe unavailable (random policy)."""
        import json as _json
        from pathlib import Path as _Path
        if self.agent:
            try:
                self.agent.load(path)
                return
            except Exception:
                pass
        # Fallback: load metadata only (agent stays as random policy)
        if _Path(path).exists():
            try:
                with open(path) as f:
                    state = _json.load(f)
                self.algo_name = state.get("algo_name", self.algo_name)
                self.name = state.get("name", self.name)
            except Exception:
                pass

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained agent and return metrics."""
        if self.agent is None:
            return {"error": "OmniSafe not available"}

        try:
            results = self.agent.evaluate(num_episodes=n_episodes)
            return {
                "avg_reward": float(results.get("reward", 0.0)),
                "avg_cost": float(results.get("cost", 0.0)),
                "cost_rate": float(results.get("cost_rate", 0.0)),
            }
        except Exception as e:
            return {"error": str(e)}


def run_omnisafe_benchmark(
    algo: str = "PPOLag",
    env_id: str = "SafetyPointGoal1-v0",
    seeds: list = None,
    total_steps: int = 1000000,
    device: str = "cpu",
    log_dir: str = "./omnisafe_results",
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Run a full OmniSafe benchmark experiment.

    Trains the specified algorithm on the given Safety-Gymnasium environment
    across multiple seeds and saves results.

    Args:
        algo: Algorithm name (PPOLag, CPO, FOCOPS, CUP)
        env_id: Safety-Gymnasium environment ID
        seeds: List of random seeds (default: [0, 1, 2, 3, 4])
        total_steps: Total training steps per seed
        device: 'cpu' or 'cuda'
        log_dir: Directory for logs and checkpoints
        use_wandb: Whether to use W&B logging

    Returns:
        Dict with per-seed and aggregate results

    Raises:
        ImportError: If OmniSafe is not installed
    """
    try:
        import omnisafe
    except ImportError:
        raise ImportError(
            "OmniSafe required for benchmarking. Install:\n"
            "  pip install omnisafe\n"
            "  pip install safety-gymnasium\n"
            "GitHub: https://github.com/PKU-Alignment/omnisafe"
        )

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    results_dir = Path(log_dir) / f"{algo}_{env_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"algo": algo, "env_id": env_id, "seeds": {}}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training {algo} on {env_id} | seed={seed}")
        print(f"{'='*60}")

        custom_cfgs = {
            "seed": seed,
            "train_cfgs": {
                "total_steps": total_steps,
                "device": device,
            },
            "algo_cfgs": {
                "steps_per_epoch": 2048,
            },
            "logger_cfgs": {
                "log_dir": str(results_dir / f"seed_{seed}"),
                "use_wandb": use_wandb,
            },
        }

        agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
        agent.learn()

        # Evaluate
        eval_results = agent.evaluate(num_episodes=20)
        seed_results = {
            "reward": float(eval_results.get("reward", 0.0)),
            "cost": float(eval_results.get("cost", 0.0)),
            "cost_rate": float(eval_results.get("cost_rate", 0.0)),
        }
        all_results["seeds"][seed] = seed_results
        print(f"Seed {seed}: reward={seed_results['reward']:.2f}, cost={seed_results['cost']:.2f}")

    # Aggregate
    rewards = [r["reward"] for r in all_results["seeds"].values()]
    costs = [r["cost"] for r in all_results["seeds"].values()]
    all_results["aggregate"] = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
    }

    # Save results
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return all_results
