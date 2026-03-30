"""
HC-MARL Phase 3 (#36): MAPPO-Lagrangian
MAPPO with cost critic and dual variable lambda for constraint satisfaction.
Actor loss: L_clip + λ * cost_advantage (penalise unsafe actions).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from hcmarl.agents.networks import ActorNetwork, CriticNetwork, CostCriticNetwork


class LagrangianRolloutBuffer:
    """Per-agent rollout storage with cost tracking for Lagrangian PPO.

    Like MAPPO's RolloutBuffer, stores data per-agent so GAE is computed
    correctly per-agent (each agent has its own reward/cost stream).
    Flattened for PPO update since the actor shares weights.
    """
    def __init__(self, agent_ids=None):
        self.agent_ids = agent_ids or []
        self.clear()

    def clear(self):
        self._obs = {a: [] for a in self.agent_ids}
        self._global_states = []
        self._actions = {a: [] for a in self.agent_ids}
        self._log_probs = {a: [] for a in self.agent_ids}
        self._rewards = {a: [] for a in self.agent_ids}
        self._costs = {a: [] for a in self.agent_ids}
        self._dones = []
        self._values = []
        self._cost_values = []
        self._n_steps = 0
        self._legacy_pending = None

    def store(self, obs, global_state, action, log_prob, reward, cost, done, value, cost_value):
        """Per-agent store. Accumulates calls then flushes per timestep."""
        if self._legacy_pending is None:
            self._legacy_pending = {
                'obs': {}, 'actions': {}, 'log_probs': {}, 'rewards': {},
                'costs': {},
                'global_state': global_state, 'done': done,
                'value': value, 'cost_value': cost_value,
                '_call_count': 0,
            }

        if not self.agent_ids:
            raise ValueError("LagrangianRolloutBuffer: agent_ids not set. "
                             "Pass agent_ids at construction time.")

        idx = self._legacy_pending['_call_count']
        if idx < len(self.agent_ids):
            a = self.agent_ids[idx]
            self._legacy_pending['obs'][a] = obs
            self._legacy_pending['actions'][a] = action
            self._legacy_pending['log_probs'][a] = log_prob
            self._legacy_pending['rewards'][a] = reward
            self._legacy_pending['costs'][a] = cost
        self._legacy_pending['_call_count'] += 1

        if self._legacy_pending['_call_count'] >= len(self.agent_ids):
            p = self._legacy_pending
            for a in self.agent_ids:
                self._obs[a].append(p['obs'][a])
                self._actions[a].append(p['actions'][a])
                self._log_probs[a].append(p['log_probs'][a])
                self._rewards[a].append(p['rewards'][a])
                self._costs[a].append(p['costs'][a])
            self._global_states.append(p['global_state'])
            self._dones.append(float(p['done']))
            self._values.append(p['value'])
            self._cost_values.append(p['cost_value'])
            self._n_steps += 1
            self._legacy_pending = None

    def compute_returns(self, last_value, last_cost_value, gamma=0.99, gae_lambda=0.95):
        """Compute reward GAE and cost GAE per-agent, then flatten."""
        T = self._n_steps
        if T == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

        values = np.array(self._values)
        cost_values = np.array(self._cost_values)
        dones = np.array(self._dones)

        all_advantages, all_returns = [], []
        all_cost_advantages, all_cost_returns = [], []

        for a in self.agent_ids:
            rewards_a = np.array(self._rewards[a])
            costs_a = np.array(self._costs[a])

            # Reward GAE
            adv_a = np.zeros(T)
            last_gae = 0.0
            for t in reversed(range(T)):
                next_val = last_value if t == T - 1 else values[t + 1]
                next_non_term = 1.0 - dones[t]
                delta = rewards_a[t] + gamma * next_val * next_non_term - values[t]
                last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
                adv_a[t] = last_gae
            all_advantages.append(adv_a)
            all_returns.append(adv_a + values)

            # Cost GAE
            cadv_a = np.zeros(T)
            last_gae = 0.0
            for t in reversed(range(T)):
                next_cv = last_cost_value if t == T - 1 else cost_values[t + 1]
                next_non_term = 1.0 - dones[t]
                delta = costs_a[t] + gamma * next_cv * next_non_term - cost_values[t]
                last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
                cadv_a[t] = last_gae
            all_cost_advantages.append(cadv_a)
            all_cost_returns.append(cadv_a + cost_values)

        # Interleave: (T, N) -> (T*N,)
        advantages = np.stack(all_advantages, axis=1).reshape(-1)
        returns = np.stack(all_returns, axis=1).reshape(-1)
        cost_advantages = np.stack(all_cost_advantages, axis=1).reshape(-1)
        cost_returns = np.stack(all_cost_returns, axis=1).reshape(-1)

        return advantages, returns, cost_advantages, cost_returns

    def get_flat_tensors(self, device):
        """Get flattened tensors for PPO update. Shape: (T*N, ...)."""
        T = self._n_steps
        obs_list, gs_list, acts_list, lp_list = [], [], [], []
        for t in range(T):
            for a in self.agent_ids:
                obs_list.append(self._obs[a][t])
                acts_list.append(self._actions[a][t])
                lp_list.append(self._log_probs[a][t])
            for _ in self.agent_ids:
                gs_list.append(self._global_states[t])

        import torch
        obs = torch.FloatTensor(np.array(obs_list)).to(device)
        gs = torch.FloatTensor(np.array(gs_list)).to(device)
        acts = torch.LongTensor(np.array(acts_list)).to(device)
        lps = torch.FloatTensor(np.array(lp_list)).to(device)
        return obs, gs, acts, lps

    def __len__(self):
        return self._n_steps * max(len(self.agent_ids), 1)


class MAPPOLagrangian:
    """MAPPO with Lagrangian constraint handling."""
    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 lr_actor=3e-4, lr_critic=1e-3, lr_lambda=5e-3,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coeff=0.01, max_grad_norm=0.5,
                 n_epochs=10, batch_size=256,
                 cost_limit=0.1, lambda_init=0.5, device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cost_limit = cost_limit

        self.log_lambda = nn.Parameter(torch.tensor(np.log(max(lambda_init, 1e-8)), device=self.device))
        self.actor = ActorNetwork(obs_dim, n_actions).to(self.device)
        self.critic = CriticNetwork(global_obs_dim).to(self.device)
        self.cost_critic = CostCriticNetwork(global_obs_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.cost_critic_optim = optim.Adam(self.cost_critic.parameters(), lr=lr_critic)
        self.lambda_optim = optim.Adam([self.log_lambda], lr=lr_lambda)
        agent_ids = [f"worker_{i}" for i in range(n_agents)]
        self.buffer = LagrangianRolloutBuffer(agent_ids=agent_ids)

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

    def update(self):
        """PPO update with Lagrangian cost penalty on actor loss."""
        if len(self.buffer) < self.batch_size:
            return {}

        obs, gs, acts, old_lp = self.buffer.get_flat_tensors(self.device)

        with torch.no_grad():
            last_gs = torch.FloatTensor(
                self.buffer._global_states[-1]
            ).to(self.device)
            last_val = self.critic(last_gs).item()
            last_cv = self.cost_critic(last_gs).item()

        advantages, returns, cost_advantages, cost_returns = self.buffer.compute_returns(
            last_val, last_cv, self.gamma, self.gae_lambda
        )

        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        cadv_t = torch.FloatTensor(cost_advantages).to(self.device)
        cret_t = torch.FloatTensor(cost_returns).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        cadv_t = (cadv_t - cadv_t.mean()) / (cadv_t.std() + 1e-8)

        lam = self.log_lambda.exp().detach()

        for _ in range(self.n_epochs):
            new_lp, entropy = self.actor.evaluate(obs, acts)
            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            reward_loss = -torch.min(surr1, surr2).mean()

            # Lagrangian cost penalty: λ * max(surr) is conservative (safe) clipping
            cost_surr1 = ratio * cadv_t
            cost_surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * cadv_t
            cost_loss = lam * torch.max(cost_surr1, cost_surr2).mean()

            actor_loss = reward_loss + cost_loss - self.entropy_coeff * entropy.mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optim.step()

            # Reward critic
            values = self.critic(gs)
            critic_loss = nn.functional.mse_loss(values, ret_t)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()

            # Cost critic
            cost_values = self.cost_critic(gs)
            cost_critic_loss = nn.functional.mse_loss(cost_values, cret_t)
            self.cost_critic_optim.zero_grad()
            cost_critic_loss.backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_critic_optim.step()

        self.buffer.clear()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "cost_critic_loss": cost_critic_loss.item(),
            "lambda": self.lam,
        }

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
