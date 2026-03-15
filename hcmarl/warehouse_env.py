"""
HC-MARL Phase 2: Warehouse Multi-Agent Environment
====================================================
PettingZoo-compatible parallel environment wrapping the HC-MARL pipeline.
Each worker agent observes its own physiological state x_i(t) = [MR, MA, MF]
per muscle group and receives task assignments from the NSWF allocator.

Integrates: 3CC-r fatigue model, ECBF safety filter, NSWF task allocation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import yaml
import copy


# ---------------------------------------------------------------------------
# Single-agent wrapper (Gymnasium API) for debugging / single-worker tests
# ---------------------------------------------------------------------------

class SingleWorkerWarehouseEnv(gym.Env):
    """Single-worker warehouse env for unit testing and baselines."""

    metadata = {"render_modes": ["human", "ansi"], "name": "SingleWorkerWarehouse-v0"}

    def __init__(
        self,
        muscle_groups: Optional[Dict] = None,
        tasks: Optional[Dict] = None,
        theta_max: Optional[Dict] = None,
        dt: float = 1.0,
        max_steps: int = 60,
        kappa: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.max_steps = max_steps
        self.kappa = kappa

        # Default muscle groups: shoulder, elbow, grip
        self.muscle_groups = muscle_groups or {
            "shoulder": {"F": 0.0146, "R": 0.00058, "r": 15},
            "elbow":    {"F": 0.00912, "R": 0.00094, "r": 15},
            "grip":     {"F": 0.00794, "R": 0.00109, "r": 30},
        }
        self.muscle_names = list(self.muscle_groups.keys())
        self.n_muscles = len(self.muscle_names)

        # Default tasks with per-muscle target loads
        self.tasks = tasks or {
            "heavy_lift":   {"shoulder": 0.50, "elbow": 0.30, "grip": 0.60},
            "light_sort":   {"shoulder": 0.10, "elbow": 0.15, "grip": 0.20},
            "carry":        {"shoulder": 0.30, "elbow": 0.20, "grip": 0.40},
            "rest":         {"shoulder": 0.00, "elbow": 0.00, "grip": 0.00},
        }
        self.task_names = list(self.tasks.keys())
        self.n_tasks = len(self.task_names)

        # Safety thresholds per muscle
        self.theta_max = theta_max or {
            "shoulder": 0.70, "elbow": 0.45, "grip": 0.25,
        }

        # Observation: [MR, MA, MF] per muscle + current_step_normalised
        obs_dim = self.n_muscles * 3 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: choose a task index (discrete)
        self.action_space = spaces.Discrete(self.n_tasks)

        # State
        self.state = None
        self.current_step = 0

    def _get_obs(self) -> np.ndarray:
        obs = []
        for m in self.muscle_names:
            obs.extend([self.state[m]["MR"], self.state[m]["MA"], self.state[m]["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, task_name: str) -> float:
        """
        Reward = productivity - fatigue penalty.
        Productivity = sum of target loads (higher load = more work done).
        Fatigue penalty = sum of MF across muscles.
        """
        task = self.tasks[task_name]
        productivity = sum(task[m] for m in self.muscle_names)
        fatigue_penalty = sum(self.state[m]["MF"] for m in self.muscle_names)
        safety_violation = 0.0
        for m in self.muscle_names:
            if self.state[m]["MF"] > self.theta_max[m]:
                safety_violation += 10.0 * (self.state[m]["MF"] - self.theta_max[m])
        return productivity - 0.5 * fatigue_penalty - safety_violation

    def _integrate_3cc_r(self, task_name: str):
        """Euler integration of 3CC-r ODEs for one timestep."""
        task = self.tasks[task_name]
        kp = 10.0  # proportional gain for baseline controller

        for m in self.muscle_names:
            params = self.muscle_groups[m]
            F = params["F"]
            R_base = params["R"]
            r = params["r"]
            TL = task[m]

            MR = self.state[m]["MR"]
            MA = self.state[m]["MA"]
            MF = self.state[m]["MF"]

            # Neural drive (proportional controller)
            C = kp * max(TL - MA, 0.0) if TL > 0 else 0.0

            # ECBF safety clipping (simplified analytical bound)
            R_eff = R_base if TL > 0 else R_base * r
            # Fatigue ceiling bound (Eq 19 simplified)
            alpha1, alpha2 = 0.5, 0.5
            h = self.theta_max[m] - MF
            h_dot = -F * MA + R_eff * MF
            psi1 = h_dot + alpha1 * h
            ecbf_bound = (1.0 / F) * (
                F**2 * MA + R_eff * F * MA - R_eff**2 * MF
                + alpha1 * (-F * MA + R_eff * MF)
                + alpha2 * psi1
            )
            # Resting floor bound (Eq 23)
            alpha3 = 0.5
            cbf_bound = R_eff * MF + alpha3 * MR

            C_max = min(ecbf_bound, cbf_bound)
            C = max(0.0, min(C, C_max))

            # ODEs (Eqs 2-4)
            dMA = C - F * MA
            dMF = F * MA - R_eff * MF
            dMR = R_eff * MF - C

            # Euler step
            MA_new = MA + dMA * self.dt
            MF_new = MF + dMF * self.dt
            MR_new = MR + dMR * self.dt

            # Clamp to [0, 1] and renormalise for conservation
            MA_new = max(0.0, min(1.0, MA_new))
            MF_new = max(0.0, min(1.0, MF_new))
            MR_new = max(0.0, min(1.0, MR_new))
            total = MA_new + MF_new + MR_new
            if total > 0:
                MA_new /= total
                MF_new /= total
                MR_new /= total

            self.state[m] = {"MR": MR_new, "MA": MA_new, "MF": MF_new}

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.state = {}
        for m in self.muscle_names:
            self.state[m] = {"MR": 1.0, "MA": 0.0, "MF": 0.0}
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        task_name = self.task_names[action]
        self._integrate_3cc_r(task_name)
        self.current_step += 1

        obs = self._get_obs()
        reward = self._compute_reward(task_name)
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "task": task_name,
            "fatigue": {m: self.state[m]["MF"] for m in self.muscle_names},
            "safety_violations": sum(
                1 for m in self.muscle_names
                if self.state[m]["MF"] > self.theta_max[m]
            ),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            lines = [f"Step {self.current_step}/{self.max_steps}"]
            for m in self.muscle_names:
                s = self.state[m]
                lines.append(
                    f"  {m}: MR={s['MR']:.3f} MA={s['MA']:.3f} MF={s['MF']:.3f} "
                    f"(Θmax={self.theta_max[m]:.2f})"
                )
            return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-agent wrapper (PettingZoo Parallel API)
# ---------------------------------------------------------------------------

class WarehouseMultiAgentEnv:
    """
    PettingZoo-style parallel environment for N warehouse workers.
    Each worker independently observes its physiological state and
    receives task assignments. The NSWF allocator handles centralised
    coordination (used in the centralised critic for MAPPO).
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "WarehouseMA-v0"}

    def __init__(
        self,
        n_workers: int = 4,
        muscle_groups: Optional[Dict] = None,
        tasks: Optional[Dict] = None,
        theta_max: Optional[Dict] = None,
        dt: float = 1.0,
        max_steps: int = 60,
        kappa: float = 1.0,
    ):
        self.n_workers = n_workers
        self.dt = dt
        self.max_steps = max_steps
        self.kappa = kappa

        self.muscle_groups = muscle_groups or {
            "shoulder": {"F": 0.0146, "R": 0.00058, "r": 15},
            "elbow":    {"F": 0.00912, "R": 0.00094, "r": 15},
            "grip":     {"F": 0.00794, "R": 0.00109, "r": 30},
        }
        self.muscle_names = list(self.muscle_groups.keys())
        self.n_muscles = len(self.muscle_names)

        self.tasks = tasks or {
            "heavy_lift":   {"shoulder": 0.50, "elbow": 0.30, "grip": 0.60},
            "light_sort":   {"shoulder": 0.10, "elbow": 0.15, "grip": 0.20},
            "carry":        {"shoulder": 0.30, "elbow": 0.20, "grip": 0.40},
            "rest":         {"shoulder": 0.00, "elbow": 0.00, "grip": 0.00},
        }
        self.task_names = list(self.tasks.keys())
        self.n_tasks = len(self.task_names)

        self.theta_max = theta_max or {
            "shoulder": 0.70, "elbow": 0.45, "grip": 0.25,
        }

        # Agent IDs
        self.possible_agents = [f"worker_{i}" for i in range(n_workers)]
        self.agents = list(self.possible_agents)

        # Spaces
        obs_dim = self.n_muscles * 3 + 1  # [MR,MA,MF]*muscles + normalised_step
        self.observation_spaces = {
            agent: spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(self.n_tasks)
            for agent in self.possible_agents
        }

        # Global state for centralised critic: all workers' states concatenated
        global_obs_dim = n_workers * self.n_muscles * 3 + 1
        self.global_observation_space = spaces.Box(
            0.0, 1.0, (global_obs_dim,), np.float32
        )

        self.states = {}
        self.current_step = 0

    def _get_obs(self, agent: str) -> np.ndarray:
        idx = int(agent.split("_")[1])
        obs = []
        for m in self.muscle_names:
            s = self.states[idx][m]
            obs.extend([s["MR"], s["MA"], s["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _get_global_obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_workers):
            for m in self.muscle_names:
                s = self.states[i][m]
                obs.extend([s["MR"], s["MA"], s["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _disagreement_utility(self, mf: float) -> float:
        """Di(MF) = κ · MF² / (1 - MF), Eq 32."""
        if mf >= 0.999:
            return 1e6
        return self.kappa * (mf ** 2) / (1.0 - mf)

    def _compute_reward(self, worker_idx: int, task_name: str) -> float:
        task = self.tasks[task_name]
        productivity = sum(task[m] for m in self.muscle_names)
        avg_mf = np.mean([self.states[worker_idx][m]["MF"] for m in self.muscle_names])
        di = self._disagreement_utility(avg_mf)
        surplus = max(productivity - di, 0.0)
        violation = sum(
            1.0 for m in self.muscle_names
            if self.states[worker_idx][m]["MF"] > self.theta_max[m]
        )
        return surplus - 5.0 * violation

    def _integrate_worker(self, worker_idx: int, task_name: str):
        """Integrate 3CC-r for one worker, one timestep, with ECBF filtering."""
        task = self.tasks[task_name]
        kp = 10.0

        for m in self.muscle_names:
            params = self.muscle_groups[m]
            F = params["F"]
            R_base = params["R"]
            r_mult = params["r"]
            TL = task[m]

            s = self.states[worker_idx][m]
            MR, MA, MF = s["MR"], s["MA"], s["MF"]

            C = kp * max(TL - MA, 0.0) if TL > 0 else 0.0
            R_eff = R_base if TL > 0 else R_base * r_mult

            # ECBF analytical bound
            alpha1, alpha2, alpha3 = 0.5, 0.5, 0.5
            h = self.theta_max[m] - MF
            h_dot = -F * MA + R_eff * MF
            psi1 = h_dot + alpha1 * h
            ecbf_bound = (1.0 / F) * (
                F**2 * MA + R_eff * F * MA - R_eff**2 * MF
                + alpha1 * (-F * MA + R_eff * MF)
                + alpha2 * psi1
            )
            cbf_bound = R_eff * MF + alpha3 * MR
            C = max(0.0, min(C, max(0.0, ecbf_bound), cbf_bound))

            dMA = C - F * MA
            dMF = F * MA - R_eff * MF
            dMR = R_eff * MF - C

            MA_new = max(0.0, MA + dMA * self.dt)
            MF_new = max(0.0, MF + dMF * self.dt)
            MR_new = max(0.0, MR + dMR * self.dt)
            total = MA_new + MF_new + MR_new
            if total > 0:
                MA_new /= total
                MF_new /= total
                MR_new /= total

            self.states[worker_idx][m] = {"MR": MR_new, "MA": MA_new, "MF": MF_new}

    def reset(self, seed=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        self.agents = list(self.possible_agents)
        self.current_step = 0
        self.states = {}
        for i in range(self.n_workers):
            self.states[i] = {}
            for m in self.muscle_names:
                self.states[i][m] = {"MR": 1.0, "MA": 0.0, "MF": 0.0}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        rewards = {}
        infos = {}

        for agent in self.agents:
            idx = int(agent.split("_")[1])
            task_idx = actions[agent]
            task_name = self.task_names[task_idx]
            self._integrate_worker(idx, task_name)
            rewards[agent] = self._compute_reward(idx, task_name)
            infos[agent] = {
                "task": task_name,
                "fatigue": {m: self.states[idx][m]["MF"] for m in self.muscle_names},
            }

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        observations = {agent: self._get_obs(agent) for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Global state for centralised critic."""
        return self._get_global_obs()


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def make_single_env(**kwargs) -> SingleWorkerWarehouseEnv:
    return SingleWorkerWarehouseEnv(**kwargs)

def make_multi_env(**kwargs) -> WarehouseMultiAgentEnv:
    return WarehouseMultiAgentEnv(**kwargs)
