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
from hcmarl.agents.mappo import RolloutBuffer


class IPPO:
    """Independent PPO: per-agent actor + per-agent local critic."""
    def __init__(self, obs_dim, n_actions, n_agents, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, entropy_coeff=0.01,
                 max_grad_norm=0.5, n_epochs=10, batch_size=256,
                 hidden_dim=64, device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.actors = [ActorNetwork(obs_dim, n_actions, hidden_dim).to(self.device) for _ in range(n_agents)]
        self.critics = [CriticNetwork(obs_dim, hidden_dim).to(self.device) for _ in range(n_agents)]
        self.actor_optims = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.critic_optims = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]
        # Each agent gets its own single-agent buffer
        self._agent_ids = [f"agent_{i}" for i in range(n_agents)]
        self.buffers = [RolloutBuffer(agent_ids=[aid]) for aid in self._agent_ids]

    def get_actions(self, observations, **kwargs):
        actions, log_probs = {}, {}
        for i, (agent_id, obs) in enumerate(sorted(observations.items())):
            obs_t = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, lp, _ = self.actors[i].get_action(obs_t)
            actions[agent_id] = action.item()
            log_probs[agent_id] = lp.item()
        return actions, log_probs, 0.0

    def store_transition(self, agent_idx, obs, action, log_prob, reward, done):
        """Store a transition for a specific agent using store_step (not legacy store)."""
        value = self.critics[agent_idx](torch.FloatTensor(obs).to(self.device)).item()
        aid = self._agent_ids[agent_idx]
        self.buffers[agent_idx].store_step(
            obs_dict={aid: obs},
            global_state=obs,
            actions_dict={aid: action},
            log_probs_dict={aid: log_prob},
            rewards_dict={aid: reward},
            done=done,
            value=value,
        )

    def update(self):
        """Per-agent PPO updates."""
        total_actor_loss, total_critic_loss = 0.0, 0.0
        updated = 0

        for i in range(self.n_agents):
            buf = self.buffers[i]
            if len(buf) < self.batch_size:
                continue

            obs, _, acts, old_lp = buf.get_flat_tensors(self.device)

            with torch.no_grad():
                last_val = self.critics[i](obs[-1]).item()
            advantages, returns = buf.compute_returns(last_val, self.gamma, self.gae_lambda)
            adv_t = torch.FloatTensor(advantages).to(self.device)
            ret_t = torch.FloatTensor(returns).to(self.device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            for _ in range(self.n_epochs):
                new_lp, entropy = self.actors[i].evaluate(obs, acts)
                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()

                self.actor_optims[i].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                self.actor_optims[i].step()

                values = self.critics[i](obs)
                critic_loss = nn.functional.mse_loss(values, ret_t)
                self.critic_optims[i].zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
                self.critic_optims[i].step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            buf.clear()
            updated += 1

        if updated == 0:
            return {}
        return {
            "actor_loss": total_actor_loss / updated,
            "critic_loss": total_critic_loss / updated,
        }

    def save(self, path):
        torch.save({"actors": [a.state_dict() for a in self.actors],
                     "critics": [c.state_dict() for c in self.critics]}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, sd in enumerate(ckpt["actors"]): self.actors[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["critics"]): self.critics[i].load_state_dict(sd)
