"""
HC-MARL Phase 2 (#23): PettingZoo ParallelEnv Wrapper
N workers with parallel observations and actions.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from hcmarl.envs.reward_functions import nswf_reward, safety_cost


class WarehousePettingZoo:
    """PettingZoo-style parallel environment for N warehouse workers."""

    metadata = {"name": "WarehouseHCMARL-v0"}

    def __init__(self, n_workers=4, muscle_groups=None, tasks=None, theta_max=None,
                 dt=1.0, max_steps=480, kappa=1.0, ecbf_mode="on"):
        from hcmarl.envs.task_profiles import TaskProfileManager
        if ecbf_mode not in ("on", "off"):
            raise ValueError(f"ecbf_mode must be 'on' or 'off', got '{ecbf_mode}'")
        self.ecbf_mode = ecbf_mode
        self.n_workers = n_workers
        self.dt = dt
        self.max_steps = max_steps
        self.kappa = kappa

        self.task_mgr = TaskProfileManager()
        self.muscle_names = self.task_mgr.muscle_names
        self.task_names = self.task_mgr.task_names
        self.n_muscles = len(self.muscle_names)
        self.n_tasks = len(self.task_names)

        from hcmarl.three_cc_r import get_muscle
        self.muscle_params = {m: {"F": get_muscle(m).F, "R": get_muscle(m).R, "r": get_muscle(m).r} for m in self.muscle_names}

        default_theta = {
            "shoulder": 0.70, "ankle": 0.80, "knee": 0.60,
            "elbow": 0.45, "trunk": 0.65, "grip": 0.25,
        }
        if theta_max is None:
            self.theta_max_per_worker = {i: dict(default_theta) for i in range(n_workers)}
        elif any(isinstance(v, dict) for v in theta_max.values()):
            # Per-worker thresholds: {"worker_0": {"shoulder": 0.7, ...}, ...}
            self.theta_max_per_worker = {}
            for i in range(n_workers):
                key = f"worker_{i}"
                if key in theta_max:
                    merged = dict(default_theta)
                    merged.update(theta_max[key])
                    self.theta_max_per_worker[i] = merged
                else:
                    self.theta_max_per_worker[i] = dict(default_theta)
        else:
            # Flat dict: {"shoulder": 0.7, ...} — same for all workers
            self.theta_max_per_worker = {i: dict(theta_max) for i in range(n_workers)}
        # Keep a flat reference for backward compatibility
        self.theta_max = self.theta_max_per_worker[0]

        self.possible_agents = [f"worker_{i}" for i in range(n_workers)]
        self.agents = list(self.possible_agents)
        self.obs_dim = self.n_muscles * 3 + 1
        self.global_obs_dim = n_workers * self.n_muscles * 3 + 1

        self.states = {}
        self.current_step = 0

    def _init_worker(self):
        return {m: {"MR": 1.0, "MA": 0.0, "MF": 0.0} for m in self.muscle_names}

    def _get_obs(self, worker_idx):
        obs = []
        for m in self.muscle_names:
            s = self.states[worker_idx][m]
            obs.extend([s["MR"], s["MA"], s["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _get_global_obs(self):
        obs = []
        for i in range(self.n_workers):
            for m in self.muscle_names:
                s = self.states[i][m]
                obs.extend([s["MR"], s["MA"], s["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _integrate(self, worker_idx, task_name):
        """Integrate 3CC-r ODEs for one worker, one timestep.

        Returns:
            (ecbf_interventions, ecbf_clip_total): number of muscles where
            the ECBF clipped the neural drive, and the total clipped magnitude.
        """
        demands = self.task_mgr.get_demand_vector(task_name)
        kp = 10.0
        ecbf_interventions = 0
        ecbf_clip_total = 0.0
        for mi, m in enumerate(self.muscle_names):
            p = self.muscle_params[m]
            F, R_base, r = p["F"], p["R"], p["r"]
            TL = demands[mi]
            s = self.states[worker_idx][m]
            MR, MA, MF = s["MR"], s["MA"], s["MF"]
            C_nominal = kp * max(TL - MA, 0.0) if TL > 0 else 0.0
            R_eff = R_base if TL > 0 else R_base * r
            if self.ecbf_mode == "on":
                # ECBF analytical clipping (match ecbf_filter.py defaults and proofs)
                alpha1, alpha2, alpha3 = 0.05, 0.05, 0.1
                h = self.theta_max_per_worker[worker_idx][m] - MF
                h_dot = -F * MA + R_eff * MF
                psi1 = h_dot + alpha1 * h
                ecbf_bound = (1.0/F)*(F**2*MA + R_eff*F*MA - R_eff**2*MF + alpha1*(-F*MA+R_eff*MF) + alpha2*psi1)
                cbf_bound = R_eff * MF + alpha3 * MR
                C = max(0.0, min(C_nominal, max(0.0, ecbf_bound), cbf_bound))
            else:
                # ecbf_mode == "off": no safety filtering
                C = max(0.0, C_nominal)
            # Track ECBF interventions
            if C_nominal > 1e-9 and (C_nominal - C) > 1e-9:
                ecbf_interventions += 1
                ecbf_clip_total += C_nominal - C
            dMA = C - F*MA
            dMF = F*MA - R_eff*MF
            dMR = R_eff*MF - C
            MA_n = max(0.0, MA + dMA*self.dt)
            MF_n = max(0.0, MF + dMF*self.dt)
            MR_n = max(0.0, MR + dMR*self.dt)
            total = MA_n + MF_n + MR_n
            if total > 0:
                MA_n /= total; MF_n /= total; MR_n /= total
            self.states[worker_idx][m] = {"MR": MR_n, "MA": MA_n, "MF": MF_n}
        return ecbf_interventions, float(ecbf_clip_total)

    def reset(self, seed=None):
        self.agents = list(self.possible_agents)
        self.current_step = 0
        self.states = {i: self._init_worker() for i in range(self.n_workers)}
        obs = {a: self._get_obs(int(a.split("_")[1])) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions):
        rewards, infos = {}, {}
        for agent in self.agents:
            idx = int(agent.split("_")[1])
            task_name = self.task_names[actions[agent]]
            ecbf_interventions, ecbf_clip_total = self._integrate(idx, task_name)
            fatigue = {m: self.states[idx][m]["MF"] for m in self.muscle_names}
            prod = self.task_mgr.task_intensity(task_name)
            worker_theta = self.theta_max_per_worker[idx]
            rewards[agent] = nswf_reward(prod, fatigue, worker_theta, kappa=self.kappa)
            cost = safety_cost(fatigue, worker_theta)
            violations = int(cost > 0)
            infos[agent] = {
                "task": task_name, "fatigue": fatigue,
                "violations": violations, "cost": cost,
                "ecbf_interventions": ecbf_interventions,
                "ecbf_clip_total": ecbf_clip_total,
            }
        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = {a: self._get_obs(int(a.split("_")[1])) for a in self.agents}
        terms = {a: done for a in self.agents}
        truncs = {a: False for a in self.agents}
        return obs, rewards, terms, truncs, infos
