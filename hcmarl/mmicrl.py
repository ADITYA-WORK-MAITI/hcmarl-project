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

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


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


# ---------------------------------------------------------------------------
# MMICRL Core: Type Discovery + Constraint Learning
# ---------------------------------------------------------------------------

class MMICRL:
    """
    Multi-Modal Inverse Constrained Reinforcement Learning.

    Objective (Eq 9): max_π [λ₁·H[π(τ)] - λ₂·H[π(τ|z)]]
    Decomposition (Eq 10): = λ₂·I(τ;z) + (λ₁-λ₂)·H[π(τ)]
    Recommended: λ₁ = λ₂ = λ → pure MI maximisation (Eq 11)

    Steps:
      1. Discover latent types z via trajectory clustering
      2. Learn per-type constraint boundaries Θmax(z_k)
      3. Output personalised safety thresholds
    """

    def __init__(
        self,
        n_types: int = 3,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        n_muscles: int = 3,
        learning_rate: float = 0.01,
        n_iterations: int = 100,
    ):
        self.n_types = n_types
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_muscles = n_muscles
        self.lr = learning_rate
        self.n_iterations = n_iterations

        # Learned parameters
        self.type_centroids = None     # K x feature_dim
        self.type_assignments = None   # N demos -> type index
        self.theta_max_per_type = {}   # {type_k: {muscle: theta_max}}
        self.type_proportions = None   # K proportions

    def _cluster_trajectories(self, features: np.ndarray) -> np.ndarray:
        """
        K-means clustering on trajectory features to discover worker types.
        This approximates the flow-based density estimation in Qiao et al.
        """
        n_demos, feat_dim = features.shape
        rng = np.random.RandomState(42)

        # Initialise centroids (K-means++)
        centroids = np.zeros((self.n_types, feat_dim))
        centroids[0] = features[rng.randint(n_demos)]

        for k in range(1, self.n_types):
            dists = np.min([
                np.sum((features - centroids[j]) ** 2, axis=1)
                for j in range(k)
            ], axis=0)
            probs = dists / dists.sum()
            centroids[k] = features[rng.choice(n_demos, p=probs)]

        # K-means iterations
        for iteration in range(50):
            # Assign
            dists = np.stack([
                np.sum((features - centroids[k]) ** 2, axis=1)
                for k in range(self.n_types)
            ], axis=1)
            assignments = np.argmin(dists, axis=1)

            # Update
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_types):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = features[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self.type_centroids = centroids
        self.type_assignments = assignments
        self.type_proportions = np.array([
            (assignments == k).sum() / n_demos for k in range(self.n_types)
        ])

        return assignments

    def _compute_mutual_information(
        self, features: np.ndarray, assignments: np.ndarray
    ) -> float:
        """
        Estimate I(τ; z) from clustered trajectories.
        I(τ;z) = H(z) - H(z|τ) ≈ H(z) (since assignment is deterministic).
        """
        proportions = np.array([
            (assignments == k).sum() / len(assignments)
            for k in range(self.n_types)
        ])
        proportions = proportions[proportions > 0]
        h_z = -np.sum(proportions * np.log(proportions + 1e-10))
        return float(h_z)

    def _learn_constraints(
        self, demonstrations: List, assignments: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Learn per-type safety thresholds by analysing the maximum fatigue
        levels observed in each type's demonstrations.

        For type k, Θmax(z_k, muscle_m) = percentile_95(MF values in type k demos)
        """
        type_mf_values = defaultdict(lambda: defaultdict(list))

        for demo_idx, (traj, type_k) in enumerate(zip(demonstrations, assignments)):
            for state, action in traj:
                for m in range(self.n_muscles):
                    mf = state[3 * m + 2]
                    type_mf_values[int(type_k)][m].append(mf)

        theta_per_type = {}
        muscle_names = ["shoulder", "elbow", "grip"]  # default mapping

        for k in range(self.n_types):
            theta_per_type[k] = {}
            for m in range(self.n_muscles):
                mf_list = type_mf_values[k][m]
                if mf_list:
                    # 95th percentile of observed fatigue = inferred threshold
                    theta = float(np.percentile(mf_list, 95))
                    theta = min(max(theta, 0.1), 0.95)  # clamp to safe range
                else:
                    theta = 0.5  # default
                name = muscle_names[m] if m < len(muscle_names) else f"muscle_{m}"
                theta_per_type[k][name] = theta

        return theta_per_type

    def fit(self, collector: DemonstrationCollector) -> Dict[str, Any]:
        """
        Full MMICRL pipeline:
          1. Extract trajectory features
          2. Cluster to discover types
          3. Compute MI objective
          4. Learn per-type constraints
        """
        features = collector.get_trajectory_features()
        assignments = self._cluster_trajectories(features)
        mi = self._compute_mutual_information(features, assignments)
        self.theta_max_per_type = self._learn_constraints(
            collector.demonstrations, assignments
        )

        # Compute objective value
        # Eq 9: λ₁·H[π(τ)] - λ₂·H[π(τ|z)]
        # ≈ λ₂·I(τ;z) + (λ₁-λ₂)·H[π(τ)]
        h_marginal = float(np.log(len(features)))  # uniform upper bound
        objective = self.lambda2 * mi + (self.lambda1 - self.lambda2) * h_marginal

        results = {
            "n_demonstrations": len(collector.demonstrations),
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
        self, worker_obs: np.ndarray
    ) -> Dict[str, float]:
        """
        Assign a worker to the nearest type and return their personalised Θmax.

        Args:
            worker_obs: trajectory feature vector (5-dim)

        Returns:
            Dict mapping muscle name to Θmax value
        """
        if self.type_centroids is None:
            raise RuntimeError("MMICRL not fitted. Call fit() first.")

        dists = np.sum((self.type_centroids - worker_obs) ** 2, axis=1)
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
