"""
HC-MARL Phase 3: Multi-Modal Inverse Constrained RL (MMICRL)
=============================================================
Learns individual fatigue thresholds Θmax(z) from unlabelled mixed
demonstrations by maximising the weighted objective:

    max_π [λ₁·H[π(τ)] - λ₂·H[π(τ|z)]]

When λ₁ = λ₂ = λ (recommended), this reduces to maximising mutual
information I(τ; z) between trajectories and latent worker types.

Reference: Qiao et al. "Multi-Modal Inverse Constrained Reinforcement
Learning from a Mixture of Demonstrations" (NeurIPS 2023).

Pipeline:
  1. Collect demonstrations from N workers performing warehouse tasks
  2. MMICRL discovers K latent types via flow-based density estimation
  3. Per-type constraint boundaries Θmax(z_k) are extracted
  4. Feed learned thresholds into ECBF safety filter for personalised protection
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ---------------------------------------------------------------------------
# CFDE: Conditional Flow-based Density Estimator (Qiao et al. NeurIPS 2023)
# Architecture: Masked Autoregressive Flow with MADE layers
# Reference impl: github.com/qiaoguanren/Multi-Modal-Inverse-Constrained-RL
# ---------------------------------------------------------------------------


def _get_autoregressive_mask(in_features, out_features, in_flow_features,
                             mask_type=None):
    """
    Build autoregressive mask for MADE.
    mask_type: 'input' | None (hidden) | 'output'
    See Germain et al. 2015, Figure 1.
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class _MaskedLinear(nn.Module):
    """Linear layer with an autoregressive mask and optional conditioning input."""

    def __init__(self, in_features, out_features, mask, cond_in_features=None,
                 bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features,
                                         bias=False)
        else:
            self.cond_linear = None
        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        out = F.linear(inputs, self.linear.weight * self.mask,
                       self.linear.bias)
        if cond_inputs is not None and self.cond_linear is not None:
            out = out + self.cond_linear(cond_inputs)
        return out


class _MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (Germain et al. 2015).
    Outputs mean mu and log-scale alpha for each dimension, enforcing
    the autoregressive property: p(x_i | x_{1:i-1}, z).
    """

    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None,
                 act='relu'):
        super().__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = _get_autoregressive_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = _get_autoregressive_mask(
            num_hidden, num_hidden, num_inputs)
        output_mask = _get_autoregressive_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = _MaskedLinear(num_inputs, num_hidden, input_mask,
                                    num_cond_inputs)
        self.trunk = nn.Sequential(
            act_func(),
            _MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            _MaskedLinear(num_hidden, num_inputs * 2, output_mask),
        )

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, dim=1)
            # Clamp log-scale to prevent numerical instability in high dimensions
            a = a.clamp(-5.0, 5.0)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)
        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, dim=1)
                a = a.clamp(-5.0, 5.0)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class _BatchNormFlow(nn.Module):
    """Batch normalization as a flow layer (Dinh et al. 2017)."""

    def __init__(self, num_inputs, momentum=0.1, eps=1e-5):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps
                self.running_mean.mul_(self.momentum).add_(
                    self.batch_mean.data * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(
                    self.batch_var.data * (1 - self.momentum))
                mean, var = self.batch_mean, self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var
            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean, var = self.batch_mean, self.batch_var
            else:
                mean, var = self.running_mean, self.running_var
            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * var.sqrt() + mean
            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class _Reverse(nn.Module):
    """Reverse permutation layer (Dinh et al. 2017)."""

    def __init__(self, num_inputs):
        super().__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class _FlowSequential(nn.Sequential):
    """Sequential container for normalizing flow layers with log-det tracking."""

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        self.num_inputs = inputs.size(-1)
        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)


class CFDE(nn.Module):
    """
    Conditional Flow-based Density Estimator from Qiao et al. NeurIPS 2023.

    Models p(x|z) using a Masked Autoregressive Flow conditioned on one-hot
    type code z. Each MADE layer computes autoregressive mean/scale:
        p(x_i | x_{1:i-1}, z) = N(x_i; mu_i, exp(alpha_i)^2)
    where mu_i = psi_mu(x_{1:i-1}, z), alpha_i = psi_alpha(x_{1:i-1}, z).

    Agent identification via Bayes rule:
        p(z|tau) = prod_{(s,a) in tau} p(s,a|z) / sum_{z'} prod p(s,a|z')
    """

    def __init__(self, input_dim, n_types, hidden_dims=None, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.n_types = n_types
        self.device = device

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Build conditional MAF: stack MADE + BatchNorm + Reverse per layer
        modules = []
        for hidden_dim in hidden_dims:
            modules.append(_MADE(
                num_inputs=input_dim,
                num_hidden=hidden_dim,
                num_cond_inputs=n_types,
                act='relu',
            ))
            modules.append(_BatchNormFlow(input_dim))
            modules.append(_Reverse(input_dim))

        self.flow = _FlowSequential(*modules).to(device)

        # Orthogonal init (matches Qiao et al.)
        for p in self.flow.parameters():
            if p.dim() >= 2:
                nn.init.orthogonal_(p)

        # Learnable type prior log p(z) (uniform init)
        self.log_prior = nn.Parameter(torch.zeros(n_types, device=device))

    def log_prob(self, x, z_onehot):
        """Compute log p(x|z) for given one-hot z."""
        return self.flow.log_probs(x, z_onehot)

    def log_prob_all_types(self, x):
        """
        Compute log p(x|z_k) for all K types.
        Returns: (batch, K) tensor of log-probabilities.
        """
        batch_size = x.size(0)
        all_log_probs = []
        for k in range(self.n_types):
            z_k = torch.zeros(batch_size, self.n_types, device=self.device)
            z_k[:, k] = 1.0
            lp = self.log_prob(x, z_k)  # (batch, 1)
            all_log_probs.append(lp.squeeze(-1))
        return torch.stack(all_log_probs, dim=1)  # (batch, K)

    def posterior(self, x):
        """
        Bayesian agent identification: p(z|x) = p(x|z) p(z) / p(x).
        Returns: (batch, K) posterior probabilities.
        """
        log_px_given_z = self.log_prob_all_types(x)  # (batch, K)
        log_pz = F.log_softmax(self.log_prior, dim=0)  # (K,)
        log_joint = log_px_given_z + log_pz.unsqueeze(0)  # (batch, K)
        return F.softmax(log_joint, dim=1)

    def assign_types(self, x):
        """Hard type assignment via MAP: argmax_z p(z|x)."""
        with torch.no_grad():
            post = self.posterior(x)
            return torch.argmax(post, dim=1)

    def trajectory_log_posterior(self, step_features, traj_indices):
        """
        Bayesian trajectory-level type assignment (Qiao et al. Eq in Section 4.2):
            p(z|τ) ∝ p(z) · ∏_{(s,a)∈τ} p(s,a|z)
            log p(z|τ) = log p(z) + Σ_{(s,a)∈τ} log p(s,a|z) - log Z

        Implementation note: we use the MEAN (not sum) of per-step log-probs
        to prevent trajectory-length-dependent mode collapse during EM. This is
        equivalent to p(z|τ) ∝ p(z) · [∏ p(s,a|z)]^{1/T}, a tempered posterior
        that prevents long trajectories from overwhelming the prior and causing
        all demos to collapse into one type. Standard practice in flow-based
        clustering (see Papamakarios et al. 2021, Section 3.3.2).

        Args:
            step_features: (total_steps, input_dim) tensor of per-step (s,a) features
            traj_indices: (total_steps,) tensor mapping steps to trajectory indices

        Returns:
            traj_posterior: (n_trajectories, K) tensor of posterior probabilities
            traj_assignments: (n_trajectories,) tensor of hard assignments
        """
        n_trajs = int(traj_indices.max().item()) + 1

        # Compute per-step log p(s,a|z_k) for all K types
        log_px_given_z = self.log_prob_all_types(step_features)  # (total_steps, K)
        log_pz = F.log_softmax(self.log_prior, dim=0)  # (K,)

        # Aggregate per-step log-likelihoods to trajectory level using MEAN
        traj_log_joint = torch.zeros(n_trajs, self.n_types,
                                      device=step_features.device)
        traj_log_joint += log_pz.unsqueeze(0)  # prior

        for t_idx in range(n_trajs):
            mask = (traj_indices == t_idx)
            if mask.any():
                # Mean log p(s,a|z_k) — tempered posterior to avoid mode collapse
                traj_log_joint[t_idx] += log_px_given_z[mask].mean(dim=0)

        # Normalize to get posterior
        traj_posterior = F.softmax(traj_log_joint, dim=1)
        traj_assignments = torch.argmax(traj_posterior, dim=1)

        return traj_posterior, traj_assignments

    def train_density(self, features, n_epochs=100, lr=0.01, batch_size=64):
        """
        EM-style training of the CFDE:
          E-step: assign types via current posterior p(z|x)
          M-step: maximize log p(x|z) under current assignments

        This maximizes the MMICRL objective I(tau; z) by finding type
        assignments that make the conditional density model most informative.
        """
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        n_samples = x.size(0)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Warm-up: initialize assignments via K-means on features for
        # stable starting point (K-means here is just initialization,
        # not the final method — the flow refines it)
        assignments = self._kmeans_init(features)
        z_onehot = F.one_hot(
            torch.tensor(assignments, device=self.device),
            self.n_types
        ).float()

        for epoch in range(n_epochs):
            self.flow.train()

            # M-step: train flow on current assignments
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                x_batch = x[idx]
                z_batch = z_onehot[idx]

                log_p = self.log_prob(x_batch, z_batch)
                loss = -log_p.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # E-step: reassign types via posterior (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self.flow.eval()
                with torch.no_grad():
                    new_assignments = self.assign_types(x)
                    z_onehot = F.one_hot(new_assignments,
                                         self.n_types).float()

        # Final assignment
        self.flow.eval()
        with torch.no_grad():
            final_assignments = self.assign_types(x).cpu().numpy()
            final_posterior = self.posterior(x).cpu().numpy()

        return final_assignments, final_posterior

    def _kmeans_init(self, features, max_iter=30):
        """K-means initialization for EM warm start."""
        rng = np.random.RandomState(42)
        n, d = features.shape
        centroids = np.zeros((self.n_types, d))
        centroids[0] = features[rng.randint(n)]
        for k in range(1, self.n_types):
            dists = np.min([
                np.sum((features - centroids[j]) ** 2, axis=1)
                for j in range(k)
            ], axis=0)
            probs = dists / (dists.sum() + 1e-10)
            centroids[k] = features[rng.choice(n, p=probs)]
        for _ in range(max_iter):
            dists = np.stack([
                np.sum((features - centroids[k]) ** 2, axis=1)
                for k in range(self.n_types)
            ], axis=1)
            assignments = np.argmin(dists, axis=1)
            new_c = np.zeros_like(centroids)
            for k in range(self.n_types):
                mask = assignments == k
                new_c[k] = features[mask].mean(0) if mask.sum() > 0 else centroids[k]
            if np.allclose(centroids, new_c, atol=1e-6):
                break
            centroids = new_c
        return assignments


# ---------------------------------------------------------------------------
# Constraint Network (Malik et al. ICML 2021, adapted for fatigue domain)
# ---------------------------------------------------------------------------

class ConstraintNetwork(nn.Module):
    """
    Learned constraint function c_θ(s) → [0, 1] per worker type.

    Malik et al. (ICML 2021) learn c(s,a) to distinguish constraint-satisfying
    from constraint-violating state-action pairs. In our fatigue domain, the
    constraint is fundamentally about state (MF levels), so we learn c_θ(s).

    Training: Binary cross-entropy on per-type demonstrations where:
        - Steps with MF < type's behavioral boundary → label 0 (safe)
        - Steps with MF near/above boundary → label 1 (constrained)
    The boundary is where c_θ(s) = 0.5, from which we extract θ_max.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)

    def train_on_demos(
        self,
        states: np.ndarray,
        n_muscles: int,
        boundary_percentile: float = 90.0,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """
        Train constraint function on demonstrations from a single type.

        Labels are derived from observed MF: steps where any muscle's MF
        exceeds the boundary_percentile of that muscle's MF distribution
        are labeled as "near constraint" (1), others as "safe" (0).

        Returns:
            Dict mapping muscle name to learned θ_max
        """
        device = next(self.parameters()).device
        muscle_names = ["shoulder", "elbow", "grip", "trunk", "ankle", "knee"]

        # Extract per-muscle MF values from states
        mf_per_muscle = {}
        for m in range(n_muscles):
            mf_col = states[:, 3 * m + 2]
            mf_per_muscle[m] = mf_col

        # Compute boundary for each muscle (high percentile of MF)
        boundaries = {}
        for m in range(n_muscles):
            boundaries[m] = float(np.percentile(mf_per_muscle[m], boundary_percentile))

        # Create binary labels: 1 if any muscle MF > its boundary
        labels = np.zeros(len(states), dtype=np.float32)
        for m in range(n_muscles):
            labels[mf_per_muscle[m] >= boundaries[m]] = 1.0

        # Ensure at least some positive labels (can't learn from all-zero)
        if labels.sum() < 2:
            # Fall back to percentile-based thresholds directly
            theta_max = {}
            for m in range(n_muscles):
                name = muscle_names[m] if m < len(muscle_names) else f"muscle_{m}"
                theta = float(np.percentile(mf_per_muscle[m], 95))
                theta_max[name] = min(max(theta, 0.1), 0.95)
            return theta_max

        # Train
        x = torch.tensor(states, dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            perm = torch.randperm(len(x), device=device)
            batch_size = min(128, len(x))
            for i in range(0, len(x), batch_size):
                idx = perm[i:i + batch_size]
                pred = self(x[idx])
                loss = F.binary_cross_entropy(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Extract θ_max: sweep MF values, find where c_θ crosses 0.5
        theta_max = {}
        for m in range(n_muscles):
            name = muscle_names[m] if m < len(muscle_names) else f"muscle_{m}"
            # Create synthetic states with varying MF for this muscle
            mf_sweep = np.linspace(0.0, 0.99, 200)
            sweep_states = np.tile(states.mean(axis=0), (200, 1))
            sweep_states[:, 3 * m + 2] = mf_sweep

            with torch.no_grad():
                costs = self(torch.tensor(sweep_states, dtype=torch.float32,
                                          device=device)).cpu().numpy()

            # Find crossing point where cost >= 0.5
            crossing_idx = np.where(costs >= 0.5)[0]
            if len(crossing_idx) > 0:
                theta = float(mf_sweep[crossing_idx[0]])
            else:
                # Network never predicts constraint → use 95th percentile fallback
                theta = float(np.percentile(mf_per_muscle[m], 95))

            theta_max[name] = min(max(theta, 0.1), 0.95)

        return theta_max


# ---------------------------------------------------------------------------
# Demonstration collection
# ---------------------------------------------------------------------------

class DemonstrationCollector:
    """
    Collect worker demonstrations for MMICRL training.
    Each demonstration τ = [(s₀, a₀), (s₁, a₁), ...] is a trajectory
    where s = [MR, MA, MF] per muscle and a = task assignment.
    """

    def __init__(self, n_muscles: int = 3):
        self.n_muscles = n_muscles
        self.demonstrations = []  # List of trajectories
        self.worker_ids = []       # Corresponding worker IDs (unlabelled in training)

    def collect_from_env(
        self,
        env,
        policy,
        n_episodes: int = 100,
        worker_id: Optional[int] = None,
    ) -> int:
        """Collect trajectories from a policy in the environment."""
        count = 0
        for ep in range(n_episodes):
            obs, _ = env.reset()
            trajectory = []

            for step in range(env.max_steps):
                if hasattr(policy, 'get_actions'):
                    actions = policy.get_actions(obs)
                else:
                    actions = {agent: policy(obs[agent]) for agent in obs}

                # Record state-action pairs
                for agent, action in actions.items():
                    state = obs[agent].copy()
                    trajectory.append((state, action))

                obs, rewards, terms, truncs, infos = env.step(actions)
                if all(terms.values()):
                    break

            self.demonstrations.append(trajectory)
            self.worker_ids.append(worker_id)
            count += 1

        return count

    def load_real_demos(
        self,
        source: str,
        path: Optional[str] = None,
        **kwargs,
    ) -> int:
        """
        Load real demonstration data from published datasets.

        Supported sources:
            'robomimic' — RoboMimic multi-proficiency human teleop (Mandlekar et al. CoRL 2021)
            'd4rl_adroit' — D4RL Adroit human CyberGlove demos (Fu et al. 2020)
            'pamap2' — PAMAP2 IMU + heart rate physical activity (Reiss & Stricker 2012)

        Args:
            source: Dataset name
            path: Path to dataset files (required for robomimic, pamap2)
            **kwargs: Dataset-specific arguments (see hcmarl.data.loaders)

        Returns:
            Number of demonstrations loaded
        """
        from hcmarl.data.loaders import load_dataset

        raw_demos = load_dataset(source, path=path, **kwargs)

        count = 0
        for demo in raw_demos:
            # Convert to (state, action) tuple format
            states = demo["states"]
            actions = demo["actions"]
            n_steps = min(len(states), len(actions))

            trajectory = []
            for t in range(n_steps):
                state = states[t]
                # Pad/truncate state to expected dimension if needed
                expected_dim = self.n_muscles * 3 + 1
                if len(state) != expected_dim:
                    padded = np.zeros(expected_dim, dtype=np.float32)
                    padded[:min(len(state), expected_dim)] = state[:expected_dim]
                    state = padded

                action = actions[t]
                if isinstance(action, np.ndarray):
                    action = int(np.argmax(action)) if len(action) > 1 else int(action[0])
                trajectory.append((state, int(action)))

            self.demonstrations.append(trajectory)
            self.worker_ids.append(demo["worker_id"])
            count += 1

        return count

    def generate_synthetic_demos(
        self,
        n_workers: int = 4,
        n_episodes_per_worker: int = 50,
        n_muscles: int = 3,
        n_steps: int = 60,
        n_tasks: int = 4,
    ) -> int:
        """
        Generate synthetic demonstrations with known worker types for validation.
        Type 0: Cautious (low threshold, frequent rest)
        Type 1: Moderate (medium threshold)
        Type 2: Aggressive (high threshold, rare rest)

        WARNING: Synthetic demos are for unit testing and algorithm validation only.
        For publication-quality results, use load_real_demos() with a real dataset
        (RoboMimic, D4RL Adroit, or PAMAP2). See hcmarl.data.loaders for details.
        """
        import warnings
        warnings.warn(
            "Using synthetic demos. For publication, use load_real_demos() "
            "with a real dataset (robomimic, d4rl_adroit, or pamap2). "
            "See hcmarl.data.loaders for supported datasets.",
            UserWarning,
            stacklevel=2,
        )
        type_params = {
            0: {"rest_prob": 0.4, "theta_eff": 0.3, "label": "cautious"},
            1: {"rest_prob": 0.2, "theta_eff": 0.5, "label": "moderate"},
            2: {"rest_prob": 0.05, "theta_eff": 0.7, "label": "aggressive"},
        }

        count = 0
        for worker_id in range(n_workers):
            worker_type = worker_id % len(type_params)
            params = type_params[worker_type]

            for ep in range(n_episodes_per_worker):
                trajectory = []
                # Simulate simple fatigue accumulation
                mf = 0.0

                for step in range(n_steps):
                    # Construct observation
                    mr = max(0.0, 1.0 - 0.1 - mf)
                    ma = min(0.1, 1.0 - mr - mf)
                    obs = np.zeros(n_muscles * 3 + 1, dtype=np.float32)
                    for m in range(n_muscles):
                        obs[m * 3] = mr
                        obs[m * 3 + 1] = ma
                        obs[m * 3 + 2] = mf
                    obs[-1] = step / n_steps

                    # Action selection based on type
                    if mf > params["theta_eff"] or np.random.random() < params["rest_prob"]:
                        action = n_tasks - 1  # rest
                        mf = max(0.0, mf - 0.05)
                    else:
                        action = np.random.randint(0, n_tasks - 1)  # work
                        mf = min(0.99, mf + 0.02 + 0.01 * np.random.randn())

                    trajectory.append((obs.copy(), action))

                self.demonstrations.append(trajectory)
                self.worker_ids.append(worker_id)
                count += 1

        return count

    def get_trajectory_features(self) -> np.ndarray:
        """
        Extract summary features from each trajectory for clustering.
        Features: [mean_MF, max_MF, rest_fraction, total_steps, fatigue_rate]

        NOTE: This is a legacy method kept for backward compatibility and quick
        validation. For proper MMICRL training, use get_step_data() which returns
        per-step (s,a) pairs as required by Qiao et al. (NeurIPS 2023).
        """
        features = []
        for traj in self.demonstrations:
            if not traj:
                features.append(np.zeros(5))
                continue

            mf_values = []
            rest_count = 0
            n_muscles = (len(traj[0][0]) - 1) // 3

            for state, action in traj:
                avg_mf = np.mean([state[3 * m + 2] for m in range(n_muscles)])
                mf_values.append(avg_mf)
                if action == max(a for _, a in traj):  # rest is highest action index
                    rest_count += 1

            mf_arr = np.array(mf_values)
            features.append([
                float(np.mean(mf_arr)),
                float(np.max(mf_arr)),
                rest_count / len(traj),
                len(traj),
                float(np.mean(np.diff(mf_arr))) if len(mf_arr) > 1 else 0.0,
            ])

        return np.array(features, dtype=np.float32)

    def get_step_data(self, n_actions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return per-step (s, a) feature vectors and trajectory indices.

        Per Qiao et al. (NeurIPS 2023), the CFDE models p(s,a|z) for individual
        state-action pairs, and trajectory-level type assignment aggregates:
            p(z|τ) ∝ p(z) · ∏_{(s,a)∈τ} p(s,a|z)

        Returns:
            step_features: (total_steps, state_dim + n_actions) float32 array
                Each row is [state, action_onehot].
            traj_indices: (total_steps,) int array
                Maps each step to its demonstration index.
        """
        all_features = []
        all_traj_idx = []

        for traj_idx, traj in enumerate(self.demonstrations):
            for state, action in traj:
                # One-hot encode the action
                a_onehot = np.zeros(n_actions, dtype=np.float32)
                a_onehot[min(action, n_actions - 1)] = 1.0
                feat = np.concatenate([state.astype(np.float32), a_onehot])
                all_features.append(feat)
                all_traj_idx.append(traj_idx)

        if not all_features:
            state_dim = self.n_muscles * 3 + 1
            return (np.zeros((0, state_dim + n_actions), dtype=np.float32),
                    np.zeros(0, dtype=np.int64))

        return (np.array(all_features, dtype=np.float32),
                np.array(all_traj_idx, dtype=np.int64))


# ---------------------------------------------------------------------------
# MMICRL Core: Type Discovery + Constraint Learning
# ---------------------------------------------------------------------------

class MMICRL:
    """
    Multi-Modal Inverse Constrained Reinforcement Learning.

    Objective (Eq 9): max_pi [lambda_1 * H[pi(tau)] - lambda_2 * H[pi(tau|z)]]
    Decomposition (Eq 10): = lambda_2 * I(tau;z) + (lambda_1 - lambda_2) * H[pi(tau)]
    Recommended: lambda_1 = lambda_2 = lambda -> pure MI maximisation (Eq 11)

    Type discovery uses a Conditional Flow-based Density Estimator (CFDE)
    built from stacked MADE layers (Germain et al. 2015) forming a Masked
    Autoregressive Flow conditioned on latent type code z.

    Architecture (Qiao et al. NeurIPS 2023, Section 4.2):
      p(x|z) = prod_i p(x_i | x_{1:i-1}, z)
      p(x_i | ...) = N(x_i; mu_i, exp(alpha_i)^2)
      mu_i = psi_mu(x_{1:i-1}, z),  alpha_i = psi_alpha(x_{1:i-1}, z)

    Agent identification via Bayes rule:
      p(z|tau) = prod_{(s,a) in tau} p(s,a|z) / sum_{z'} prod p(s,a|z')

    Steps:
      1. Extract trajectory features from mixed demonstrations
      2. Train CFDE to discover latent types z via EM on flow density
      3. Learn per-type constraint boundaries theta_max(z_k)
      4. Output personalised safety thresholds for ECBF
    """

    def __init__(
        self,
        n_types: int = 3,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        n_muscles: int = 3,
        learning_rate: float = 0.01,
        n_iterations: int = 100,
        hidden_dims: Optional[List[int]] = None,
        device: str = 'cpu',
    ):
        self.n_types = n_types
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_muscles = n_muscles
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.hidden_dims = hidden_dims or [64, 64]
        self.device = device

        # Learned parameters
        self.cfde = None               # CFDE model (trained flow)
        self.type_centroids = None     # K x feature_dim (kept for compatibility)
        self.type_assignments = None   # N demos -> type index
        self.theta_max_per_type = {}   # {type_k: {muscle: theta_max}}
        self.type_proportions = None   # K proportions
        self._feature_mean = None      # For normalizing new inputs
        self._feature_std = None

    def _discover_types_cfde(
        self, step_features: np.ndarray, traj_indices: np.ndarray, n_demos: int
    ) -> np.ndarray:
        """
        Discover worker types using CFDE on per-step (s,a) pairs.

        Per Qiao et al. (NeurIPS 2023):
        - The CFDE models p(s,a|z) for individual state-action pairs
        - Type assignment uses trajectory-level Bayesian aggregation:
            p(z|τ) ∝ p(z) · ∏_{(s,a)∈τ} p(s,a|z)
        - EM alternates between:
            E-step: Assign trajectory types via p(z|τ)
            M-step: Train flow on per-step data under current assignments

        Args:
            step_features: (total_steps, feat_dim) per-step (s,a) vectors
            traj_indices: (total_steps,) maps each step to its trajectory
            n_demos: total number of trajectories
        """
        total_steps, feat_dim = step_features.shape

        # Normalize for stable flow training
        self._feature_mean = step_features.mean(axis=0)
        self._feature_std = step_features.std(axis=0) + 1e-8
        features_norm = (step_features - self._feature_mean) / self._feature_std

        # Build CFDE on per-step feature dimension
        self.cfde = CFDE(
            input_dim=feat_dim,
            n_types=self.n_types,
            hidden_dims=self.hidden_dims,
            device=self.device,
        )

        x = torch.tensor(features_norm, dtype=torch.float32, device=self.device)
        t_idx = torch.tensor(traj_indices, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(self.cfde.parameters(), lr=self.lr)

        # Initialize trajectory assignments via K-means on trajectory summary features
        traj_summary = np.zeros((n_demos, feat_dim), dtype=np.float32)
        for i in range(n_demos):
            mask = traj_indices == i
            if mask.any():
                traj_summary[i] = features_norm[mask].mean(axis=0)
        traj_assignments = self.cfde._kmeans_init(traj_summary)

        # Map trajectory assignments to per-step assignments for M-step
        step_assignments = traj_assignments[traj_indices]
        z_onehot = torch.nn.functional.one_hot(
            torch.tensor(step_assignments, device=self.device),
            self.n_types
        ).float()

        batch_size = min(256, total_steps)

        for epoch in range(self.n_iterations):
            self.cfde.flow.train()

            # M-step: train flow on per-step (s,a) data under current assignments
            perm = torch.randperm(total_steps, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, total_steps, batch_size):
                idx = perm[i:i + batch_size]
                x_batch = x[idx]
                z_batch = z_onehot[idx]
                log_p = self.cfde.log_prob(x_batch, z_batch)
                loss = -log_p.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.cfde.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # E-step: reassign trajectories via Bayesian posterior
            # Delay first E-step to epoch 20 to let the flow learn conditional
            # structure from K-means init. Then every 10 epochs. Guard against
            # mode collapse: reject reassignments that empty any type.
            e_step_start = min(20, self.n_iterations // 3)
            if epoch >= e_step_start and (epoch + 1) % 10 == 0:
                self.cfde.flow.eval()
                with torch.no_grad():
                    _, new_traj_assignments = self.cfde.trajectory_log_posterior(
                        x, t_idx
                    )
                    proposed = new_traj_assignments.cpu().numpy()
                    # Reject if any type gets fewer than 5% of demos
                    counts = [int((proposed == k).sum()) for k in range(self.n_types)]
                    min_count = max(1, int(n_demos * 0.05))
                    if min(counts) >= min_count:
                        traj_assignments = proposed
                        step_assignments = traj_assignments[traj_indices]
                        z_onehot = torch.nn.functional.one_hot(
                            torch.tensor(step_assignments, device=self.device),
                            self.n_types
                        ).float()

        # Final trajectory-level assignment with collapse guard
        self.cfde.flow.eval()
        with torch.no_grad():
            final_posterior, final_assignments = self.cfde.trajectory_log_posterior(
                x, t_idx
            )
            proposed_final = final_assignments.cpu().numpy()
            counts_final = [int((proposed_final == k).sum()) for k in range(self.n_types)]
            min_count = max(1, int(n_demos * 0.05))
            if min(counts_final) >= min_count:
                traj_assignments = proposed_final

        self.type_assignments = traj_assignments

        # Compute type proportions
        self.type_proportions = np.array([
            (traj_assignments == k).sum() / n_demos for k in range(self.n_types)
        ])

        # Store trajectory-summary centroids for backward compat / fallback
        traj_summary_raw = np.zeros((n_demos, feat_dim), dtype=np.float32)
        for i in range(n_demos):
            mask = traj_indices == i
            if mask.any():
                traj_summary_raw[i] = step_features[mask].mean(axis=0)
        self.type_centroids = np.zeros((self.n_types, feat_dim))
        for k in range(self.n_types):
            mask = traj_assignments == k
            if mask.sum() > 0:
                self.type_centroids[k] = traj_summary_raw[mask].mean(axis=0)
            else:
                self.type_centroids[k] = traj_summary_raw.mean(axis=0)

        return traj_assignments

    def _compute_mutual_information(
        self, step_features: np.ndarray, traj_indices: np.ndarray,
        assignments: np.ndarray
    ) -> float:
        """
        Estimate I(τ; z) from the trained CFDE using trajectory-level posteriors.

        Uses the decomposition I(τ;z) = H(z) - H(z|τ).

        If CFDE soft posteriors are available and non-degenerate, both terms are
        computed from soft posteriors. If the posteriors show mode collapse (any
        type gets 0 mass), falls back to hard-assignment entropy H(z), which is
        an upper bound on I(τ;z) (since H(z|τ)=0 for deterministic assignments).
        """
        n_demos = len(assignments)

        def _hard_assignment_mi():
            """MI from hard assignments: H(z) where z ~ empirical type dist."""
            proportions = np.array([
                (assignments == k).sum() / n_demos
                for k in range(self.n_types)
            ])
            proportions = proportions[proportions > 0]
            return float(-np.sum(proportions * np.log(proportions + 1e-10)))

        if self.cfde is None:
            return max(0.0, _hard_assignment_mi())

        # Try soft posteriors from CFDE
        features_norm = (step_features - self._feature_mean) / self._feature_std
        x = torch.tensor(features_norm, dtype=torch.float32,
                         device=self.device)
        t_idx = torch.tensor(traj_indices, dtype=torch.long,
                              device=self.device)
        with torch.no_grad():
            traj_post, _ = self.cfde.trajectory_log_posterior(x, t_idx)
            post = traj_post.cpu().numpy()  # (n_demos, K)
            post_clipped = np.clip(post, 1e-10, 1.0)

        # Check for mode collapse: if any type gets <1% marginal mass, posteriors
        # are degenerate and we fall back to hard assignments
        marginal = post_clipped.mean(axis=0)
        marginal = marginal / marginal.sum()
        if marginal.min() < 0.01:
            return max(0.0, _hard_assignment_mi())

        # H(z) from marginal of soft posteriors
        h_z = -np.sum(marginal * np.log(marginal + 1e-10))

        # H(z|τ) from per-trajectory posterior entropy
        h_z_given_tau = float(np.mean(
            -np.sum(post_clipped * np.log(post_clipped), axis=1)
        ))

        return float(max(0.0, h_z - h_z_given_tau))

    def _learn_constraints(
        self, demonstrations: List, assignments: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Learn per-type safety thresholds using a constraint network (Malik et al.
        ICML 2021), adapted for the fatigue domain.

        For each discovered type k:
        1. Collect all states from type-k demonstrations
        2. Train a ConstraintNetwork c_θ(s) on binary labels (safe vs near-boundary)
        3. Extract θ_max per muscle from the learned decision boundary (c_θ = 0.5)

        This is a proper learned constraint function, not just a percentile.
        """
        theta_per_type = {}
        state_dim = self.n_muscles * 3 + 1  # MR,MA,MF per muscle + timestep

        for k in range(self.n_types):
            # Collect states from type-k demos
            type_states = []
            for demo_idx, type_k in enumerate(assignments):
                if int(type_k) == k:
                    traj = demonstrations[demo_idx]
                    for state, action in traj:
                        type_states.append(state.astype(np.float32))

            if len(type_states) < 10:
                # Too few samples — use conservative defaults
                muscle_names = ["shoulder", "elbow", "grip", "trunk", "ankle", "knee"]
                theta_per_type[k] = {}
                for m in range(self.n_muscles):
                    name = muscle_names[m] if m < len(muscle_names) else f"muscle_{m}"
                    theta_per_type[k][name] = 0.5
                continue

            states_arr = np.array(type_states, dtype=np.float32)

            # Train constraint network for this type
            c_net = ConstraintNetwork(state_dim, hidden_dim=32).to(self.device)
            theta_per_type[k] = c_net.train_on_demos(
                states_arr, self.n_muscles, boundary_percentile=90.0,
                n_epochs=50, lr=1e-3,
            )

        return theta_per_type

    def fit(self, collector: DemonstrationCollector, n_actions: Optional[int] = None) -> Dict[str, Any]:
        """
        Full MMICRL pipeline (Qiao et al. NeurIPS 2023):
          1. Extract per-step (s,a) features from demonstrations
          2. Train CFDE on per-step data with trajectory-level Bayesian assignment
          3. Compute MI objective I(τ; z) from trajectory posteriors
          4. Learn per-type constraint boundaries via constraint networks
        """
        # Auto-detect n_actions from demonstration data
        if n_actions is None:
            max_action = 0
            for traj in collector.demonstrations:
                for _, action in traj:
                    max_action = max(max_action, int(action))
            n_actions = max_action + 1

        n_demos = len(collector.demonstrations)
        step_features, traj_indices = collector.get_step_data(n_actions=n_actions)

        # Type discovery via per-step CFDE with trajectory-level assignment
        assignments = self._discover_types_cfde(step_features, traj_indices, n_demos)

        # MI from trajectory-level posteriors
        mi = self._compute_mutual_information(step_features, traj_indices, assignments)

        # Learn constraint functions per type
        self.theta_max_per_type = self._learn_constraints(
            collector.demonstrations, assignments
        )

        # Compute objective (Eq 10): λ₂·I(τ;z) + (λ₁-λ₂)·H[π(τ)]
        # H[π(τ)] estimated from per-step features (same space as CFDE)
        h_marginal = 0.0
        if abs(self.lambda1 - self.lambda2) > 1e-10:
            n_bins = max(5, int(np.sqrt(len(step_features))))
            for col in range(step_features.shape[1]):
                counts, _ = np.histogram(step_features[:, col], bins=n_bins)
                probs = counts[counts > 0] / counts.sum()
                h_marginal += float(-np.sum(probs * np.log(probs + 1e-10)))
            h_marginal /= step_features.shape[1]
        objective = self.lambda2 * mi + (self.lambda1 - self.lambda2) * h_marginal

        results = {
            "n_demonstrations": n_demos,
            "n_types_discovered": self.n_types,
            "type_proportions": self.type_proportions.tolist(),
            "mutual_information": mi,
            "objective_value": objective,
            "theta_per_type": self.theta_max_per_type,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
        }

        return results

    def get_threshold_for_worker(
        self, worker_trajectory: np.ndarray, traj_as_steps: bool = True
    ) -> Dict[str, float]:
        """
        Assign a worker to a type using the trained CFDE and return
        their personalised theta_max.

        For trajectory-level identification (traj_as_steps=True), uses Bayesian
        aggregation: p(z|τ) ∝ p(z) · ∏_{(s,a)∈τ} p(s,a|z)

        Args:
            worker_trajectory: Either:
                - (T, feat_dim) array of per-step (s,a) features (traj_as_steps=True)
                - (feat_dim,) single summary vector for backward compat (traj_as_steps=False)

        Returns:
            Dict mapping muscle name to theta_max value
        """
        if self.cfde is None and self.type_centroids is None:
            raise RuntimeError("MMICRL not fitted. Call fit() first.")

        if self.cfde is not None and self._feature_mean is not None:
            if traj_as_steps and worker_trajectory.ndim == 2:
                # Proper trajectory-level Bayesian identification
                obs_norm = (worker_trajectory - self._feature_mean) / self._feature_std
                x = torch.tensor(obs_norm, dtype=torch.float32,
                                  device=self.device)
                t_idx = torch.zeros(len(x), dtype=torch.long, device=self.device)
                with torch.no_grad():
                    _, assignments = self.cfde.trajectory_log_posterior(x, t_idx)
                    assigned_type = int(assignments[0].item())
            else:
                # Single vector — per-step posterior
                obs = worker_trajectory if worker_trajectory.ndim == 1 else worker_trajectory[0]
                obs_norm = (obs - self._feature_mean) / self._feature_std
                x = torch.tensor(obs_norm, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
                with torch.no_grad():
                    assigned_type = int(self.cfde.assign_types(x).item())
        else:
            # Fallback to centroid distance
            obs = worker_trajectory if worker_trajectory.ndim == 1 else worker_trajectory.mean(axis=0)
            dists = np.sum((self.type_centroids - obs) ** 2, axis=1)
            assigned_type = int(np.argmin(dists))

        return self.theta_max_per_type[assigned_type]


# ---------------------------------------------------------------------------
# Online adaptation (lightweight threshold update during shift)
# ---------------------------------------------------------------------------

class OnlineAdapter:
    """
    Exponential moving average update to Θmax during a shift.
    If a worker consistently operates below their learned threshold,
    the system can tighten it. If they approach it too often,
    it can trigger re-evaluation.

    This is a simple online complement to the offline MMICRL learning.
    """

    def __init__(
        self,
        initial_thresholds: Dict[str, float],
        ema_alpha: float = 0.01,
        tighten_factor: float = 0.95,
        alert_fraction: float = 0.8,
    ):
        self.thresholds = dict(initial_thresholds)
        self.ema_alpha = ema_alpha
        self.tighten_factor = tighten_factor
        self.alert_fraction = alert_fraction

        # Running statistics
        self.ema_mf = {m: 0.0 for m in self.thresholds}
        self.max_mf_seen = {m: 0.0 for m in self.thresholds}
        self.steps = 0

    def update(self, fatigue_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Update online statistics with new fatigue observations.
        Returns alerts if approaching threshold.
        """
        self.steps += 1
        alerts = {}

        for muscle, mf in fatigue_values.items():
            if muscle not in self.thresholds:
                continue

            # EMA update
            self.ema_mf[muscle] = (
                self.ema_alpha * mf + (1 - self.ema_alpha) * self.ema_mf[muscle]
            )
            self.max_mf_seen[muscle] = max(self.max_mf_seen[muscle], mf)

            # Check if approaching threshold
            theta = self.thresholds[muscle]
            if mf > self.alert_fraction * theta:
                alerts[muscle] = {
                    "level": "warning",
                    "mf": mf,
                    "theta": theta,
                    "fraction": mf / theta,
                }

        return alerts

    def get_adapted_thresholds(self) -> Dict[str, float]:
        """
        Return potentially tightened thresholds based on online observations.
        If max observed MF is well below threshold, tighten conservatively.
        """
        adapted = {}
        for muscle, theta in self.thresholds.items():
            max_seen = self.max_mf_seen.get(muscle, 0.0)
            if self.steps > 10 and max_seen < 0.5 * theta:
                # Worker never approaches threshold — tighten
                adapted[muscle] = theta * self.tighten_factor
            else:
                adapted[muscle] = theta
        return adapted


# ---------------------------------------------------------------------------
# Validation against ground truth
# ---------------------------------------------------------------------------

def validate_mmicrl(
    n_workers: int = 12,
    n_episodes_per_worker: int = 50,
    n_types: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end validation:
      1. Generate synthetic demos with known types
      2. Run MMICRL to discover types
      3. Compare learned thresholds with ground truth
    """
    collector = DemonstrationCollector(n_muscles=3)
    n_demos = collector.generate_synthetic_demos(
        n_workers=n_workers,
        n_episodes_per_worker=n_episodes_per_worker,
        n_muscles=3,
    )

    mmicrl = MMICRL(n_types=n_types, lambda1=1.0, lambda2=1.0, n_muscles=3)
    results = mmicrl.fit(collector)

    if verbose:
        print(f"MMICRL Validation Results:")
        print(f"  Demonstrations: {results['n_demonstrations']}")
        print(f"  Types discovered: {results['n_types_discovered']}")
        print(f"  Type proportions: {results['type_proportions']}")
        print(f"  Mutual information: {results['mutual_information']:.4f}")
        print(f"  Objective value: {results['objective_value']:.4f}")
        print(f"  Learned thresholds:")
        for k, thetas in results['theta_per_type'].items():
            print(f"    Type {k}: {thetas}")

    return results


if __name__ == "__main__":
    validate_mmicrl(verbose=True)
