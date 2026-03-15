"""
HC-MARL Phase 2: MAPPO Agent
==============================
Multi-Agent Proximal Policy Optimisation with centralised critic.
Each worker has a decentralised actor π_θ(a_i | o_i) and shares a
centralised value function V_φ(s) where s is the global state.

Architecture:
  Actor:  o_i → MLP → π(a_i | o_i)    [decentralised, per-worker]
  Critic: s   → MLP → V(s)             [centralised, shared]

Reference: Yu et al. "The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games" (NeurIPS 2022).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


class MAPPOBuffer:
    """Rollout buffer for MAPPO training. Stores transitions for all agents."""

    def __init__(self, n_agents: int, obs_dim: int, global_obs_dim: int, buffer_size: int = 2048):
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim

        self.observations = np.zeros((buffer_size, n_agents, obs_dim), dtype=np.float32)
        self.global_states = np.zeros((buffer_size, global_obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_agents), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_agents), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def store(self, obs, global_state, actions, rewards, dones, log_probs, values):
        idx = self.ptr
        self.observations[idx] = obs
        self.global_states[idx] = global_state
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.log_probs[idx] = log_probs
        self.values[idx] = values
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def compute_gae(self, last_values: np.ndarray, gamma: float = 0.99, lam: float = 0.95):
        """Compute Generalised Advantage Estimation."""
        size = self.buffer_size if self.full else self.ptr
        for agent_idx in range(self.n_agents):
            last_gae = 0.0
            for t in reversed(range(size)):
                if t == size - 1:
                    next_value = last_values[agent_idx]
                    next_non_terminal = 1.0 - self.dones[t, agent_idx]
                else:
                    next_value = self.values[t + 1, agent_idx]
                    next_non_terminal = 1.0 - self.dones[t, agent_idx]

                delta = (
                    self.rewards[t, agent_idx]
                    + gamma * next_value * next_non_terminal
                    - self.values[t, agent_idx]
                )
                last_gae = delta + gamma * lam * next_non_terminal * last_gae
                self.advantages[t, agent_idx] = last_gae
                self.returns[t, agent_idx] = last_gae + self.values[t, agent_idx]

    def get_batches(self, batch_size: int = 256):
        """Yield mini-batches for PPO update."""
        size = self.buffer_size if self.full else self.ptr
        indices = np.arange(size)
        np.random.shuffle(indices)
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_idx = indices[start:end]
            yield {
                "observations": self.observations[batch_idx],
                "global_states": self.global_states[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "returns": self.returns[batch_idx],
                "values": self.values[batch_idx],
            }

    def reset(self):
        self.ptr = 0
        self.full = False


class MAPPOActorNumpy:
    """
    Simple softmax actor using numpy (no PyTorch dependency).
    For full training, replace with PyTorch MLP.

    Policy: π(a|o) = softmax(W2 · ReLU(W1 · o + b1) + b2)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = rng.randn(hidden_dim, obs_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(n_actions, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(n_actions, dtype=np.float32)

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Returns action logits."""
        h = np.maximum(0, self.W1 @ obs + self.b1)  # ReLU
        logits = self.W2 @ h + self.b2
        return logits

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Sample action and return (action, log_prob)."""
        logits = self.forward(obs)
        # Stable softmax
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()

        action = np.random.choice(self.n_actions, p=probs)
        log_prob = np.log(probs[action])
        return int(action), float(log_prob)

    def get_probs(self, obs: np.ndarray) -> np.ndarray:
        logits = self.forward(obs)
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return np.clip(probs, 1e-8, 1.0)


class MAPPOCriticNumpy:
    """
    Simple centralised value function using numpy.
    V(s) = W2 · ReLU(W1 · s + b1) + b2 (scalar output)
    """

    def __init__(self, global_obs_dim: int, hidden_dim: int = 128, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / global_obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = rng.randn(hidden_dim, global_obs_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(1, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, global_obs: np.ndarray) -> float:
        h = np.maximum(0, self.W1 @ global_obs + self.b1)
        value = (self.W2 @ h + self.b2)[0]
        return float(value)


class MAPPOAgent:
    """
    Complete MAPPO agent managing N decentralised actors + 1 centralised critic.

    Usage:
        agent = MAPPOAgent(n_agents=4, obs_dim=10, global_obs_dim=41, n_actions=4)
        actions, log_probs, values = agent.get_actions(observations, global_state)
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        global_obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        seed: int = 42,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

        # Shared actor weights (parameter sharing across agents)
        self.actor = MAPPOActorNumpy(obs_dim, n_actions, hidden_dim, seed)

        # Centralised critic
        self.critic = MAPPOCriticNumpy(global_obs_dim, critic_hidden_dim, seed + 1)

        # Buffer
        self.buffer = MAPPOBuffer(n_agents, obs_dim, global_obs_dim)

    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        global_state: np.ndarray,
    ) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
        """
        Get actions for all agents.
        Returns: (action_dict, log_probs_array, values_array)
        """
        actions = {}
        log_probs = np.zeros(self.n_agents, dtype=np.float32)
        values = np.zeros(self.n_agents, dtype=np.float32)

        for i, agent_id in enumerate(sorted(observations.keys())):
            obs = observations[agent_id]
            action, log_prob = self.actor.get_action(obs)
            value = self.critic.forward(global_state)

            actions[agent_id] = action
            log_probs[i] = log_prob
            values[i] = value

        return actions, log_probs, values

    def get_values(self, global_state: np.ndarray) -> np.ndarray:
        """Get value estimates for all agents (same since critic is shared)."""
        v = self.critic.forward(global_state)
        return np.full(self.n_agents, v, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training loop (numpy-only version for Phase 1 validation)
# ---------------------------------------------------------------------------

def train_mappo_episode(
    env,  # WarehouseMultiAgentEnv
    agent: MAPPOAgent,
    n_episodes: int = 100,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Run MAPPO data collection for multiple episodes.
    NOTE: This is the numpy-only version for demonstration.
    Full gradient-based training requires PyTorch (see training.py).
    """
    metrics = {
        "episode_rewards": [],
        "episode_violations": [],
        "episode_fatigue": [],
        "episode_tasks_completed": [],
    }

    for ep in range(n_episodes):
        obs, infos = env.reset()
        episode_reward = 0.0
        total_violations = 0
        total_fatigue = 0.0
        tasks_done = 0

        for step in range(env.max_steps):
            global_state = env._get_global_obs()
            actions, log_probs, values = agent.get_actions(obs, global_state)

            obs, rewards, terms, truncs, infos = env.step(actions)

            ep_reward = sum(rewards.values())
            episode_reward += ep_reward

            for agent_id, info in infos.items():
                total_violations += info.get("safety_violations", 0)
                fatigue_vals = info.get("fatigue", {})
                total_fatigue += sum(fatigue_vals.values())
                if info.get("task", "rest") != "rest":
                    tasks_done += 1

            if all(terms.values()):
                break

        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_violations"].append(total_violations)
        metrics["episode_fatigue"].append(total_fatigue / max(1, env.max_steps * env.n_workers))
        metrics["episode_tasks_completed"].append(tasks_done)

        if verbose and (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1}/{n_episodes}: "
                f"reward={episode_reward:.1f}, "
                f"violations={total_violations}, "
                f"avg_fatigue={metrics['episode_fatigue'][-1]:.3f}, "
                f"tasks={tasks_done}"
            )

    return metrics
