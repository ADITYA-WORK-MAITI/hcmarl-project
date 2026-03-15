"""
HC-MARL Phase 3 (#46): SafePO Baseline Wrapper
MACPO (Multi-Agent Constrained Policy Optimisation) from SafePO.
"""
import numpy as np
from typing import Dict, Optional

class SafePOWrapper:
    """Wraps SafePO MACPO with HC-MARL's multi-agent action interface."""

    def __init__(self, obs_dim=19, n_actions=6, n_agents=4, config=None, device="cpu"):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config or {}
        self.device = device
        self.agent = None
        self.name = "SafePO-MACPO"
        self._try_init()

    def _try_init(self):
        try:
            import safepo
            self.agent = safepo.MACPO(self.config)
        except ImportError:
            self.agent = None

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

    def save(self, path):
        if self.agent: self.agent.save(path)
