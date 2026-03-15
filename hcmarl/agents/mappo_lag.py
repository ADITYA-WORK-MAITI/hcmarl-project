"""
HC-MARL Phase 3 (#36): MAPPO-Lagrangian
MAPPO with cost critic and dual variable lambda for constraint satisfaction.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from hcmarl.agents.networks import ActorNetwork, CriticNetwork, CostCriticNetwork

class MAPPOLagrangian:
    """MAPPO with Lagrangian constraint handling."""
    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 lr_actor=3e-4, lr_critic=1e-3, lr_lambda=5e-3,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 cost_limit=0.1, lambda_init=0.5, device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.cost_limit = cost_limit
        self.log_lambda = nn.Parameter(torch.tensor(np.log(max(lambda_init, 1e-8)), device=self.device))
        self.actor = ActorNetwork(obs_dim, n_actions).to(self.device)
        self.critic = CriticNetwork(global_obs_dim).to(self.device)
        self.cost_critic = CostCriticNetwork(global_obs_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.cost_critic_optim = optim.Adam(self.cost_critic.parameters(), lr=lr_critic)
        self.lambda_optim = optim.Adam([self.log_lambda], lr=lr_lambda)

    @property
    def lam(self):
        return self.log_lambda.exp().item()

    def get_actions(self, observations, global_state):
        actions, log_probs = {}, {}
        gs = torch.FloatTensor(global_state).to(self.device)
        value = self.critic(gs).item()
        cost_value = self.cost_critic(gs).item()
        for agent_id, obs in observations.items():
            obs_t = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, lp, _ = self.actor.get_action(obs_t)
            actions[agent_id] = action.item()
            log_probs[agent_id] = lp.item()
        return actions, log_probs, value, cost_value

    def update_lambda(self, mean_cost):
        """Dual gradient ascent: increase lambda when cost > limit."""
        loss = -self.log_lambda.exp() * (mean_cost - self.cost_limit)
        self.lambda_optim.zero_grad()
        loss.backward()
        self.lambda_optim.step()

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                     "cost_critic": self.cost_critic.state_dict(), "log_lambda": self.log_lambda.data}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.cost_critic.load_state_dict(ckpt["cost_critic"])
        self.log_lambda.data = ckpt["log_lambda"]
