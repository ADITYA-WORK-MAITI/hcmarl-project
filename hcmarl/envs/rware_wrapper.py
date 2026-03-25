"""
HC-MARL RWARE (Robotic Warehouse) Benchmark Wrapper
=====================================================
Wraps the RWARE multi-agent warehouse environment to provide
the same interface as our PettingZoo wrapper, enabling
direct comparison on a recognised multi-agent benchmark.

RWARE Reference:
    Papoudakis et al. "Benchmarking Multi-Agent Deep Reinforcement
    Learning Algorithms in Cooperative Tasks." NeurIPS 2021 Datasets Track.
    GitHub: https://github.com/semitable/robotic-warehouse

Install: pip install rware
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings


class RWAREWrapper:
    """
    Wraps RWARE gymnasium environment with HC-MARL's multi-agent interface.

    Maps RWARE's discrete actions (forward, turn left/right, toggle load)
    to our task allocation framework, and exposes observations in our
    standard format.

    Args:
        env_id: RWARE environment ID (e.g., 'rware-tiny-2ag-v2')
        n_agents: Number of agents (must match env_id)
        max_steps: Maximum episode steps
        seed: Random seed
    """

    # Standard RWARE environment IDs
    ENVS = {
        "tiny-2ag": "rware-tiny-2ag-v2",
        "tiny-4ag": "rware-tiny-4ag-v2",
        "small-4ag": "rware-small-4ag-v2",
        "medium-4ag": "rware-medium-4ag-v2",
    }

    # RWARE action space: 0=noop, 1=forward, 2=left, 3=right, 4=toggle
    ACTION_NAMES = ["noop", "forward", "turn_left", "turn_right", "toggle_load"]

    def __init__(
        self,
        env_id: str = "rware-tiny-2ag-v2",
        max_steps: int = 500,
        seed: int = 42,
        config: Optional[Dict] = None,
    ):
        self.env_id = env_id
        self.max_steps = max_steps
        self.seed = seed
        self.config = config or {}
        self._env = None
        self._step_count = 0

        # Try to create the environment
        self._try_init()

    def _try_init(self):
        """Attempt to initialise RWARE environment."""
        try:
            import rware  # noqa: F401 — registers envs
            import gymnasium as gym

            self._env = gym.make(self.env_id, max_steps=self.max_steps)
            self.n_agents = self._env.unwrapped.n_agents
            self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
            self.n_actions = 5  # RWARE action space size
            self._available = True

        except ImportError:
            warnings.warn(
                "RWARE not installed. Install: pip install rware\n"
                "Falling back to mock mode for testing."
            )
            self.n_agents = int(self.env_id.split("ag")[0].split("-")[-1].replace("", "") or 2)
            self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
            self.n_actions = 5
            self._available = False

    @property
    def available(self) -> bool:
        """Whether real RWARE is available."""
        return self._available

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment, return observations dict."""
        self._step_count = 0

        if self._available:
            obs, info = self._env.reset(seed=seed or self.seed)
            obs_dict = {
                f"agent_{i}": np.array(obs[i], dtype=np.float32)
                for i in range(self.n_agents)
            }
            return obs_dict, info
        else:
            # Mock mode for testing without rware installed
            obs_dict = {
                agent: np.zeros(15, dtype=np.float32)
                for agent in self.possible_agents
            }
            return obs_dict, {}

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Step environment with action dict.

        Args:
            actions: Dict mapping agent_id to discrete action (0-4)

        Returns:
            (observations, rewards, terminations, truncations, infos)
            All dicts keyed by agent_id.
        """
        self._step_count += 1

        if self._available:
            # Convert dict to list in agent order
            action_list = [actions.get(f"agent_{i}", 0) for i in range(self.n_agents)]

            obs_list, reward_list, term, trunc, info = self._env.step(action_list)

            obs = {f"agent_{i}": np.array(obs_list[i], dtype=np.float32)
                   for i in range(self.n_agents)}
            rewards = {f"agent_{i}": float(reward_list[i])
                       for i in range(self.n_agents)}

            # Handle both old and new gym API
            if isinstance(term, (bool, np.bool_)):
                terms = {agent: bool(term) for agent in self.possible_agents}
                truncs = {agent: bool(trunc) for agent in self.possible_agents}
            else:
                terms = {f"agent_{i}": bool(term[i]) for i in range(self.n_agents)}
                truncs = {f"agent_{i}": bool(trunc[i]) for i in range(self.n_agents)}

            infos = {agent: info if isinstance(info, dict) else {}
                     for agent in self.possible_agents}

            return obs, rewards, terms, truncs, infos
        else:
            # Mock mode
            obs = {agent: np.zeros(15, dtype=np.float32)
                   for agent in self.possible_agents}
            rewards = {agent: 0.0 for agent in self.possible_agents}
            done = self._step_count >= self.max_steps
            terms = {agent: done for agent in self.possible_agents}
            truncs = {agent: False for agent in self.possible_agents}
            infos = {agent: {} for agent in self.possible_agents}
            return obs, rewards, terms, truncs, infos

    def close(self):
        """Close underlying environment."""
        if self._available and self._env is not None:
            self._env.close()

    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        if self._available:
            return self._env.observation_space[0].shape[0]
        return 15  # mock

    def get_action_dim(self) -> int:
        """Get number of discrete actions."""
        return self.n_actions

    def map_task_to_action(self, task_name: str) -> int:
        """
        Map HC-MARL task names to RWARE actions for cross-env comparison.

        Approximate mapping:
            heavy_lift  -> toggle_load (4)
            carry       -> forward (1)
            light_sort  -> toggle_load (4)
            push_cart   -> forward (1)
            rest        -> noop (0)
        """
        mapping = {
            "heavy_lift": 4,
            "carry": 1,
            "light_sort": 4,
            "push_cart": 1,
            "overhead_reach": 4,
            "rest": 0,
        }
        return mapping.get(task_name, 0)
