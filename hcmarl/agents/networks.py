"""
HC-MARL Phase 3 (#34): Actor-Critic Neural Networks
MLP-based actor and critic with orthogonal initialization.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

def init_weights(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ActorNetwork(nn.Module):
    """Decentralised actor: π(a|o_i) — maps local observation to action distribution."""
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.net.apply(lambda m: init_weights(m, np.sqrt(2)))
        init_weights(self.net[-1], gain=0.01)  # small init for output

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # logits

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class CriticNetwork(nn.Module):
    """Centralised critic: V(s) — maps global state to scalar value."""
    def __init__(self, global_obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.net.apply(lambda m: init_weights(m, np.sqrt(2)))
        init_weights(self.net[-1], gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class CostCriticNetwork(nn.Module):
    """Cost critic for Lagrangian methods: V_c(s) — predicts cumulative cost."""
    def __init__(self, global_obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.net.apply(lambda m: init_weights(m, np.sqrt(2)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
