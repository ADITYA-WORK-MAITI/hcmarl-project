"""
HC-MARL Phase 2 (#25): Safety-Gymnasium ECBF Wrapper
Wraps standard Safety-Gymnasium envs with our ECBF filter post-hoc.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any


class ECBFSafetyWrapper:
    """Post-hoc ECBF wrapper for Safety-Gymnasium environments."""

    def __init__(self, env, safe_distance=0.5, alpha1=2.0, alpha2=2.0):
        self.env = env
        self.safe_distance = safe_distance
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def _compute_barrier(self, pos, hazard_pos):
        dist_sq = np.sum((pos - hazard_pos)**2)
        return self.safe_distance**2 - dist_sq

    def _filter_action(self, action, obs):
        """Filter action through ECBF. Requires env to expose pos/vel/hazards."""
        if not hasattr(self.env, 'robot_pos'):
            return action
        pos = self.env.robot_pos[:2] if hasattr(self.env, 'robot_pos') else obs[:2]
        vel = self.env.robot_vel[:2] if hasattr(self.env, 'robot_vel') else obs[2:4]
        hazards = getattr(self.env, 'hazards_pos', [])
        filtered = action.copy()
        for h_pos in hazards:
            h = self._compute_barrier(pos, h_pos[:2])
            h_dot = -2.0 * np.dot(pos - h_pos[:2], vel)
            psi1 = h_dot + self.alpha1 * h
            if psi1 < 0:
                diff = pos - h_pos[:2]
                direction = diff / (np.linalg.norm(diff) + 1e-8)
                correction = -psi1 * np.concatenate([direction, np.zeros(len(action)-2)])
                filtered = filtered + correction[:len(action)] * 0.5
        return np.clip(filtered, -1.0, 1.0)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs = getattr(self, '_last_obs', None)
        safe_action = self._filter_action(action, obs) if obs is not None else action
        result = self.env.step(safe_action)
        self._last_obs = result[0]
        return result

    def __getattr__(self, name):
        return getattr(self.env, name)
