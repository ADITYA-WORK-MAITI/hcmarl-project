"""
HC-MARL Phase 3 (#39): Full HC-MARL Agent
MAPPO policy -> ECBF filter -> 3CC-r plant (Section 7.3 pipeline).
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from hcmarl.agents.mappo import MAPPO

class HCMARLAgent:
    """Complete HC-MARL agent: MAPPO + ECBF safety filter + NSWF allocation."""
    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 theta_max=None, ecbf_params=None, use_nswf=True, device="cpu"):
        self.mappo = MAPPO(obs_dim, global_obs_dim, n_actions, n_agents, device=device)
        self.theta_max = theta_max or {}
        self.ecbf_alpha1 = ecbf_params.get("alpha1", 0.5) if ecbf_params else 0.5
        self.ecbf_alpha2 = ecbf_params.get("alpha2", 0.5) if ecbf_params else 0.5
        self.ecbf_alpha3 = ecbf_params.get("alpha3", 0.5) if ecbf_params else 0.5
        self.use_nswf = use_nswf
        self.n_agents = n_agents
        self.n_actions = n_actions

    def get_actions(self, observations, global_state=None):
        """Get MAPPO actions (ECBF filtering happens in env integration step)."""
        if global_state is None:
            global_state = np.concatenate(list(observations.values()))
        actions, log_probs, value = self.mappo.get_actions(observations, global_state)
        return actions, log_probs, value

    def save(self, path):
        self.mappo.save(path)

    def load(self, path):
        self.mappo.load(path)
