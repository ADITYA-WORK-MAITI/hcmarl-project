"""
HC-MARL Phase 2 (#23): PettingZoo ParallelEnv Wrapper
N workers with parallel observations and actions.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class WarehousePettingZoo:
    """PettingZoo-style parallel environment for N warehouse workers."""

    metadata = {"name": "WarehouseHCMARL-v0"}

    def __init__(self, n_workers=4, muscle_groups=None, tasks=None, theta_max=None,
                 dt=1.0, max_steps=480, kappa=1.0):
        from hcmarl.envs.task_profiles import TaskProfileManager
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

        self.theta_max = theta_max or {
            "shoulder": 0.70, "ankle": 0.80, "knee": 0.60,
            "elbow": 0.45, "trunk": 0.65, "grip": 0.25,
        }

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
        demands = self.task_mgr.get_demand_vector(task_name)
        kp = 10.0
        for mi, m in enumerate(self.muscle_names):
            p = self.muscle_params[m]
            F, R_base, r = p["F"], p["R"], p["r"]
            TL = demands[mi]
            s = self.states[worker_idx][m]
            MR, MA, MF = s["MR"], s["MA"], s["MF"]
            C = kp * max(TL - MA, 0.0) if TL > 0 else 0.0
            R_eff = R_base if TL > 0 else R_base * r
            # ECBF analytical clipping
            alpha1, alpha2, alpha3 = 0.5, 0.5, 0.5
            h = self.theta_max[m] - MF
            h_dot = -F * MA + R_eff * MF
            psi1 = h_dot + alpha1 * h
            ecbf_bound = (1.0/F)*(F**2*MA + R_eff*F*MA - R_eff**2*MF + alpha1*(-F*MA+R_eff*MF) + alpha2*psi1)
            cbf_bound = R_eff * MF + alpha3 * MR
            C = max(0.0, min(C, max(0.0, ecbf_bound), cbf_bound))
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
            self._integrate(idx, task_name)
            fatigue = {m: self.states[idx][m]["MF"] for m in self.muscle_names}
            prod = self.task_mgr.task_intensity(task_name)
            avg_mf = np.mean(list(fatigue.values()))
            violations = sum(1 for m in self.muscle_names if fatigue[m] > self.theta_max[m])
            rewards[agent] = prod - 0.5*avg_mf - 5.0*violations
            infos[agent] = {"task": task_name, "fatigue": fatigue, "violations": violations}
        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = {a: self._get_obs(int(a.split("_")[1])) for a in self.agents}
        terms = {a: done for a in self.agents}
        truncs = {a: False for a in self.agents}
        return obs, rewards, terms, truncs, infos
