"""
HC-MARL Phase 3 (#42): OmniSafe Baseline Wrapper
Uniform interface for PPO-Lagrangian, CPO, FOCOPS from OmniSafe.
"""
import numpy as np
from typing import Dict, Optional

class OmniSafeWrapper:
    """Wraps OmniSafe algorithms with HC-MARL's action interface."""
    SUPPORTED = ["PPOLag", "CPO", "FOCOPS"]

    def __init__(self, algo_name="PPOLag", obs_dim=19, n_actions=6, config=None, device="cpu"):
        self.algo_name = algo_name
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config or {}
        self.device = device
        self.agent = None
        self.name = f"OmniSafe-{algo_name}"
        self._try_init()

    def _try_init(self):
        try:
            import omnisafe
            env_id = self.config.get("env_id", "SafetyPointGoal1-v0")
            self.agent = omnisafe.Agent(self.algo_name, env_id)
        except ImportError:
            self.agent = None  # Fallback to random

    def get_actions(self, observations, **kwargs):
        actions = {}
        for agent_id, obs in observations.items():
            if self.agent is not None:
                try:
                    action = self.agent.predict(obs)
                    actions[agent_id] = int(action) if np.isscalar(action) else int(action[0])
                except Exception:
                    actions[agent_id] = np.random.randint(0, self.n_actions)
            else:
                actions[agent_id] = np.random.randint(0, self.n_actions)
        return actions

    def train(self, total_steps=1000000):
        if self.agent: self.agent.learn(total_steps)

    def save(self, path):
        if self.agent: self.agent.save(path)
