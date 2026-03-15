"""
HC-MARL Phase 3 (#35): MAPPO Algorithm
Multi-Agent PPO with centralised critic and decentralised actors.
Ref: Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from hcmarl.agents.networks import ActorNetwork, CriticNetwork


class RolloutBuffer:
    """Stores rollout data for PPO update."""
    def __init__(self):
        self.obs, self.global_states, self.actions = [], [], []
        self.log_probs, self.rewards, self.dones, self.values = [], [], [], []

    def store(self, obs, global_state, action, log_prob, reward, done, value):
        self.obs.append(obs); self.global_states.append(global_state)
        self.actions.append(action); self.log_probs.append(log_prob)
        self.rewards.append(reward); self.dones.append(done); self.values.append(value)

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        rewards = np.array(self.rewards); dones = np.array(self.dones); values = np.array(self.values)
        T = len(rewards)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T-1 else values[t+1]
            next_non_term = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * next_non_term - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def clear(self):
        self.__init__()


class MAPPO:
    """Multi-Agent PPO with shared actor weights and centralised critic."""

    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coeff=0.01, max_grad_norm=0.5,
                 n_epochs=10, batch_size=256, device="cpu"):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = ActorNetwork(obs_dim, n_actions).to(self.device)
        self.critic = CriticNetwork(global_obs_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer = RolloutBuffer()

    def get_actions(self, observations: Dict[str, np.ndarray], global_state: np.ndarray):
        actions, log_probs = {}, {}
        gs = torch.FloatTensor(global_state).to(self.device)
        value = self.critic(gs).item()
        for agent_id, obs in observations.items():
            obs_t = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, lp, _ = self.actor.get_action(obs_t)
            actions[agent_id] = action.item()
            log_probs[agent_id] = lp.item()
        return actions, log_probs, value

    def update(self):
        if len(self.buffer.obs) < self.batch_size:
            return {}
        obs = torch.FloatTensor(np.array(self.buffer.obs)).to(self.device)
        gs = torch.FloatTensor(np.array(self.buffer.global_states)).to(self.device)
        acts = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        old_lp = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)

        with torch.no_grad():
            last_val = self.critic(gs[-1]).item()
        advantages, returns = self.buffer.compute_returns(last_val, self.gamma, self.gae_lambda)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            new_lp, entropy = self.actor.evaluate(obs, acts)
            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optim.step()

            values = self.critic(gs)
            critic_loss = nn.functional.mse_loss(values, ret_t)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()
            total_loss += (actor_loss.item() + critic_loss.item())

        self.buffer.clear()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
