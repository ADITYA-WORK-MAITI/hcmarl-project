"""
HC-MARL Phase 3 (#35): MAPPO Algorithm
Multi-Agent PPO with centralised critic and decentralised actors.
Ref: Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022).

Buffer structure: Per-agent trajectory storage. GAE computed per-agent.
Flattened for PPO update since the actor has shared weights.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from hcmarl.agents.networks import ActorNetwork, CriticNetwork


class RolloutBuffer:
    """Per-agent rollout storage for correct GAE computation.

    Stores data as lists-of-dicts keyed by agent_id. Each agent's
    trajectory is kept separate so GAE is computed correctly per-agent
    (each agent has its own reward stream).

    For the PPO update, per-agent data is flattened since the actor
    shares weights across all agents (CTDE paradigm).
    """
    def __init__(self, agent_ids=None):
        self.agent_ids = agent_ids or []
        self.clear()

    def clear(self):
        # Per-agent storage: {agent_id: [list of values per timestep]}
        self._obs = {a: [] for a in self.agent_ids}
        self._global_states = []  # Shared across agents (one per timestep)
        self._actions = {a: [] for a in self.agent_ids}
        self._log_probs = {a: [] for a in self.agent_ids}
        self._rewards = {a: [] for a in self.agent_ids}
        self._dones = []  # Shared (episode-level termination)
        self._values = []  # Centralised critic: one value per timestep
        self._n_steps = 0

    def store_step(self, obs_dict, global_state, actions_dict, log_probs_dict,
                   rewards_dict, done, value):
        """Store one timestep for ALL agents simultaneously.

        Args:
            obs_dict: {agent_id: obs_array}
            global_state: global observation for centralised critic
            actions_dict: {agent_id: action_int}
            log_probs_dict: {agent_id: log_prob_float}
            rewards_dict: {agent_id: reward_float}
            done: bool/float, shared episode termination
            value: float, centralised critic value for this timestep
        """
        # Initialise agent_ids on first call if not set
        if not self.agent_ids:
            self.agent_ids = sorted(obs_dict.keys())
            self._obs = {a: [] for a in self.agent_ids}
            self._actions = {a: [] for a in self.agent_ids}
            self._log_probs = {a: [] for a in self.agent_ids}
            self._rewards = {a: [] for a in self.agent_ids}

        for a in self.agent_ids:
            self._obs[a].append(obs_dict[a])
            self._actions[a].append(actions_dict[a])
            self._log_probs[a].append(log_probs_dict.get(a, 0.0))
            self._rewards[a].append(rewards_dict[a])

        self._global_states.append(global_state)
        self._dones.append(float(done))
        self._values.append(value)
        self._n_steps += 1

    # Legacy compatibility: store() for per-agent calls from old train.py
    # This is a shim — new code should use store_step()
    _legacy_pending = None

    def store(self, obs, global_state, action, log_prob, reward, done, value):
        """Legacy per-agent store. Accumulates calls then flushes per timestep."""
        if self._legacy_pending is None:
            self._legacy_pending = {
                'obs': {}, 'actions': {}, 'log_probs': {}, 'rewards': {},
                'global_state': global_state, 'done': done, 'value': value,
                '_call_count': 0,
            }

        # Assign to next agent in order
        if not self.agent_ids:
            raise ValueError("RolloutBuffer: agent_ids not set. "
                             "Pass agent_ids at construction time.")

        idx = self._legacy_pending['_call_count']
        if idx < len(self.agent_ids):
            a = self.agent_ids[idx]
            self._legacy_pending['obs'][a] = obs
            self._legacy_pending['actions'][a] = action
            self._legacy_pending['log_probs'][a] = log_prob
            self._legacy_pending['rewards'][a] = reward
        self._legacy_pending['_call_count'] += 1

        # When all agents have stored, flush
        if self._legacy_pending['_call_count'] >= len(self.agent_ids):
            self.store_step(
                self._legacy_pending['obs'],
                self._legacy_pending['global_state'],
                self._legacy_pending['actions'],
                self._legacy_pending['log_probs'],
                self._legacy_pending['rewards'],
                self._legacy_pending['done'],
                self._legacy_pending['value'],
            )
            self._legacy_pending = None

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute GAE per-agent, then flatten.

        Returns:
            advantages: (T * N_agents,) flattened
            returns: (T * N_agents,) flattened
        """
        T = self._n_steps
        if T == 0:
            return np.zeros(0), np.zeros(0)

        values = np.array(self._values)  # (T,)
        dones = np.array(self._dones)    # (T,)
        N = len(self.agent_ids)

        all_advantages = []
        all_returns = []

        for a in self.agent_ids:
            rewards_a = np.array(self._rewards[a])  # (T,)
            adv_a = np.zeros(T)
            last_gae = 0.0
            for t in reversed(range(T)):
                next_val = last_value if t == T - 1 else values[t + 1]
                next_non_term = 1.0 - dones[t]
                delta = rewards_a[t] + gamma * next_val * next_non_term - values[t]
                last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
                adv_a[t] = last_gae
            ret_a = adv_a + values
            all_advantages.append(adv_a)
            all_returns.append(ret_a)

        # Interleave: for each timestep t, stack all agents
        # Result shape: (T * N,)
        advantages = np.stack(all_advantages, axis=1).reshape(-1)  # (T, N) -> (T*N,)
        returns = np.stack(all_returns, axis=1).reshape(-1)

        return advantages, returns

    def get_flat_tensors(self, device):
        """Get flattened tensors for PPO update.

        Returns obs, global_states, actions, log_probs as (T*N, ...) tensors.
        Global states are repeated N times per timestep (same for all agents).
        """
        T = self._n_steps
        N = len(self.agent_ids)

        # Flatten obs: (T, N, obs_dim) -> (T*N, obs_dim)
        obs_list = []
        for t in range(T):
            for a in self.agent_ids:
                obs_list.append(self._obs[a][t])
        obs = torch.FloatTensor(np.array(obs_list)).to(device)

        # Global states: repeat for each agent at each timestep
        gs_list = []
        for t in range(T):
            for _ in self.agent_ids:
                gs_list.append(self._global_states[t])
        gs = torch.FloatTensor(np.array(gs_list)).to(device)

        # Actions and log_probs: flatten same way
        acts_list, lp_list = [], []
        for t in range(T):
            for a in self.agent_ids:
                acts_list.append(self._actions[a][t])
                lp_list.append(self._log_probs[a][t])
        acts = torch.LongTensor(np.array(acts_list)).to(device)
        lps = torch.FloatTensor(np.array(lp_list)).to(device)

        return obs, gs, acts, lps

    def __len__(self):
        return self._n_steps * max(len(self.agent_ids), 1)


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

        agent_ids = [f"worker_{i}" for i in range(n_agents)]
        self.buffer = RolloutBuffer(agent_ids=agent_ids)

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
        if len(self.buffer) < self.batch_size:
            return {}

        obs, gs, acts, old_lp = self.buffer.get_flat_tensors(self.device)

        with torch.no_grad():
            # Use last global state for bootstrap value
            last_gs = torch.FloatTensor(
                self.buffer._global_states[-1]
            ).to(self.device)
            last_val = self.critic(last_gs).item()

        advantages, returns = self.buffer.compute_returns(
            last_val, self.gamma, self.gae_lambda
        )
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
