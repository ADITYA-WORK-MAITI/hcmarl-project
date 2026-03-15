"""
HC-MARL Phase 3 (#49): IPPO
Independent PPO — no parameter sharing, no centralised critic.
Each worker trains its own actor-critic independently.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from hcmarl.agents.networks import ActorNetwork, CriticNetwork

class IPPO:
    """Independent PPO: per-agent actor + per-agent local critic."""
    def __init__(self, obs_dim, n_actions, n_agents, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, hidden_dim=64, device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.actors = [ActorNetwork(obs_dim, n_actions, hidden_dim).to(self.device) for _ in range(n_agents)]
        self.critics = [CriticNetwork(obs_dim, hidden_dim).to(self.device) for _ in range(n_agents)]
        self.actor_optims = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.critic_optims = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]

    def get_actions(self, observations, **kwargs):
        actions, log_probs = {}, {}
        for i, (agent_id, obs) in enumerate(sorted(observations.items())):
            obs_t = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, lp, _ = self.actors[i].get_action(obs_t)
            actions[agent_id] = action.item()
            log_probs[agent_id] = lp.item()
        return actions, log_probs, 0.0

    def save(self, path):
        torch.save({"actors": [a.state_dict() for a in self.actors],
                     "critics": [c.state_dict() for c in self.critics]}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, sd in enumerate(ckpt["actors"]): self.actors[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["critics"]): self.critics[i].load_state_dict(sd)
