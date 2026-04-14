"""
HC-MARL Phase 2 (#23): PettingZoo ParallelEnv Wrapper
N workers with parallel observations and actions.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from hcmarl.envs.reward_functions import nswf_reward, safety_cost
from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
from hcmarl.three_cc_r import ThreeCCrState, get_muscle


class WarehousePettingZoo:
    """PettingZoo-style parallel environment for N warehouse workers."""

    metadata = {"name": "WarehouseHCMARL-v0"}

    def __init__(self, n_workers=4, muscle_groups=None, tasks=None, theta_max=None,
                 dt=1.0, max_steps=480, kappa=1.0, ecbf_mode="on",
                 action_mode="discrete", disagreement_type="divergent",
                 muscle_params_override=None,
                 ecbf_alpha1=0.05, ecbf_alpha2=0.05, ecbf_alpha3=0.1):
        """
        Args:
            action_mode: "discrete" (default) — agent selects a task index.
                "continuous" — agent outputs C_nom per muscle (Remark 7.2).
                In continuous mode, task assignment comes from the NSWF
                allocator at the outer loop (C-7.A), not from the agent.
            disagreement_type: "divergent" (Eq 32, default) or "constant"
                (D_i = kappa). Used for the no_divergent ablation (C-17).
            muscle_params_override: Optional dict {muscle_name: {"F": ..., "R": ..., "r": ...}}
                to override MUSCLE_REGISTRY values. Used for no_reperfusion
                ablation (C-17) where r=1 for all muscles.
        """
        from hcmarl.envs.task_profiles import TaskProfileManager
        if ecbf_mode not in ("on", "off"):
            raise ValueError(f"ecbf_mode must be 'on' or 'off', got '{ecbf_mode}'")
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(f"action_mode must be 'discrete' or 'continuous', got '{action_mode}'")
        self.ecbf_mode = ecbf_mode
        self.action_mode = action_mode
        self.disagreement_type = disagreement_type
        self.n_workers = n_workers
        self.dt = dt
        self.max_steps = max_steps
        self.kappa = kappa

        self.task_mgr = TaskProfileManager()
        self.muscle_names = self.task_mgr.muscle_names
        self.task_names = self.task_mgr.task_names
        self.n_muscles = len(self.muscle_names)
        self.n_tasks = len(self.task_names)

        # Muscle params: use override if provided (e.g., no_reperfusion sets r=1)
        if muscle_params_override:
            self.muscle_params = {}
            for m in self.muscle_names:
                base = {"F": get_muscle(m).F, "R": get_muscle(m).R, "r": get_muscle(m).r}
                if m in muscle_params_override:
                    base.update(muscle_params_override[m])
                self.muscle_params[m] = base
        else:
            self.muscle_params = {m: {"F": get_muscle(m).F, "R": get_muscle(m).R, "r": get_muscle(m).r} for m in self.muscle_names}

        # S-17: grip raised from 0.25 to 0.35 to match hcmarl_full_config.yaml
        # and warehouse_env.py defaults. With theta_min_max=19.5% (r=30),
        # the old 0.25 gave only 5.5pp margin -- ECBF was excessively
        # aggressive on grip, creating an artificial bottleneck. 0.35 gives
        # 15.5pp margin, consistent with all training configs.
        #
        # S-18: Ankle (0.80) and trunk (0.65) have wide margins above their
        # theta_min_max values (2.1% and 11.0% respectively). This is
        # biologically correct, not a calibration oversight:
        #   - Ankle: Rr/F = 46.35 (Table 2) -- recovers 46x faster than it
        #     fatigues. Fatigue is self-limiting; MF never approaches 0.80
        #     under any realistic task demand. The ECBF serves as a backup.
        #   - Trunk: Rr/F = 8.08 -- fast recovery means the ECBF rarely
        #     binds. The filter is most critical for shoulder (Rr/F = 0.596,
        #     the only muscle where rest-phase overshoot is a genuine concern
        #     per Theorem 5.7 and Table 2 of the math doc).
        # Artificially lowering these thresholds would impose constraints
        # that the physiology does not warrant.
        default_theta = {
            "shoulder": 0.70, "ankle": 0.80, "knee": 0.60,
            "elbow": 0.45, "trunk": 0.65, "grip": 0.35,
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

        # C-6.A: Build per-worker, per-muscle ECBFFilter instances
        ecbf_alpha1 = ecbf_alpha1
        ecbf_alpha2 = ecbf_alpha2
        ecbf_alpha3 = ecbf_alpha3
        self.ecbf_filters = {}  # {worker_idx: {muscle_name: ECBFFilter}}
        for i in range(n_workers):
            self.ecbf_filters[i] = {}
            for m in self.muscle_names:
                mp = get_muscle(m)
                theta = self.theta_max_per_worker[i].get(m, default_theta.get(m, 0.7))
                # Ensure theta_max satisfies Assumption 5.5
                theta = max(theta, mp.theta_min_max + 0.01)
                params = ECBFParams(
                    theta_max=theta,
                    alpha1=ecbf_alpha1,
                    alpha2=ecbf_alpha2,
                    alpha3=ecbf_alpha3,
                )
                self.ecbf_filters[i][m] = ECBFFilter(muscle=mp, ecbf_params=params)

        self.possible_agents = [f"worker_{i}" for i in range(n_workers)]
        self.agents = list(self.possible_agents)

        # C-8.A: In continuous mode, obs includes task assignment one-hot
        if self.action_mode == "continuous":
            # obs = [MR,MA,MF]*muscles + task_one_hot(n_tasks) + step_norm
            self.obs_dim = self.n_muscles * 3 + self.n_tasks + 1
        else:
            self.obs_dim = self.n_muscles * 3 + 1
        self.global_obs_dim = n_workers * (self.n_muscles * 3) + 1

        # Per-worker current task assignment (for continuous mode, set by allocator)
        self._task_assignments = {i: 0 for i in range(n_workers)}  # 0 = rest

        self.states = {}
        self.current_step = 0

    def _init_worker(self):
        return {m: {"MR": 1.0, "MA": 0.0, "MF": 0.0} for m in self.muscle_names}

    def _get_obs(self, worker_idx):
        obs = []
        for m in self.muscle_names:
            s = self.states[worker_idx][m]
            obs.extend([s["MR"], s["MA"], s["MF"]])
        if self.action_mode == "continuous":
            # Append task assignment one-hot (conditioning for the actor)
            task_oh = np.zeros(self.n_tasks, dtype=np.float32)
            task_idx = self._task_assignments.get(worker_idx, 0)
            task_oh[min(task_idx, self.n_tasks - 1)] = 1.0
            obs.extend(task_oh.tolist())
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
            (ecbf_interventions, ecbf_clip_total, barrier_violations):
            - ecbf_interventions: muscles where ECBF clipped C_nom
            - ecbf_clip_total: total clipped magnitude
            - barrier_violations: muscles where discrete-time step crossed
              the barrier (MF > theta_max or MR < 0) despite ECBF filtering.
              S-4: empirical verification that the continuous-time ECBF
              guarantee holds under Euler discretization.
        """
        demands = self.task_mgr.get_demand_vector(task_name)
        kp = 1.0  # proportional gain (Eq 35)
        ecbf_interventions = 0
        ecbf_clip_total = 0.0
        barrier_violations = 0
        worker_theta = self.theta_max_per_worker[worker_idx]
        for mi, m in enumerate(self.muscle_names):
            p = self.muscle_params[m]
            F, R_base, r = p["F"], p["R"], p["r"]
            TL = demands[mi]
            s = self.states[worker_idx][m]
            MR, MA, MF = s["MR"], s["MA"], s["MF"]
            C_nominal = kp * max(TL - MA, 0.0) if TL > 0 else 0.0
            R_eff = R_base if TL > 0 else R_base * r
            if self.ecbf_mode == "on":
                # C-6.A: Use canonical ECBFFilter instead of inlined bounds
                state = ThreeCCrState(MR=MR, MA=MA, MF=MF)
                C, _infeasible = self.ecbf_filters[worker_idx][m].filter_analytical(
                    state, C_nominal, TL
                )
            else:
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
            MR_n = 1.0 - MA_n - MF_n
            if MR_n < 0.0:
                s = MA_n + MF_n
                if s > 0:
                    MA_n /= s; MF_n /= s
                MR_n = 0.0
            # S-4: post-step barrier verification (discrete-time check)
            theta_m = worker_theta.get(m, 0.7)
            if MF_n > theta_m + 1e-6:
                barrier_violations += 1
            if MR_n < -1e-6:
                barrier_violations += 1
            self.states[worker_idx][m] = {"MR": MR_n, "MA": MA_n, "MF": MF_n}
        return ecbf_interventions, float(ecbf_clip_total), barrier_violations

    def _integrate_continuous(self, worker_idx, c_nom_per_muscle):
        """Integrate 3CC-r with agent-provided C_nom per muscle (Remark 7.2).

        In continuous mode, the RL policy outputs C_nom directly and the
        ECBF filter clips it. The task assignment determines R_eff only
        (work vs rest recovery rate).

        Args:
            worker_idx: Worker index.
            c_nom_per_muscle: np.array of shape (n_muscles,) with C_nom in [0, 1].

        Returns:
            (ecbf_interventions, ecbf_clip_total, barrier_violations)
        """
        task_idx = self._task_assignments.get(worker_idx, 0)
        task_name = self.task_names[task_idx]
        demands = self.task_mgr.get_demand_vector(task_name)
        ecbf_interventions = 0
        ecbf_clip_total = 0.0
        barrier_violations = 0
        worker_theta = self.theta_max_per_worker[worker_idx]
        for mi, m in enumerate(self.muscle_names):
            p = self.muscle_params[m]
            F, R_base, r = p["F"], p["R"], p["r"]
            TL = demands[mi]
            s = self.states[worker_idx][m]
            MR, MA, MF = s["MR"], s["MA"], s["MF"]
            C_nominal = float(c_nom_per_muscle[mi])
            R_eff = R_base if TL > 0 else R_base * r
            if self.ecbf_mode == "on":
                state = ThreeCCrState(MR=MR, MA=MA, MF=MF)
                C, _infeasible = self.ecbf_filters[worker_idx][m].filter_analytical(
                    state, C_nominal, TL
                )
            else:
                C = max(0.0, C_nominal)
            if C_nominal > 1e-9 and (C_nominal - C) > 1e-9:
                ecbf_interventions += 1
                ecbf_clip_total += C_nominal - C
            dMA = C - F * MA
            dMF = F * MA - R_eff * MF
            dMR = R_eff * MF - C
            MA_n = max(0.0, MA + dMA * self.dt)
            MF_n = max(0.0, MF + dMF * self.dt)
            MR_n = 1.0 - MA_n - MF_n
            if MR_n < 0.0:
                s = MA_n + MF_n
                if s > 0:
                    MA_n /= s; MF_n /= s
                MR_n = 0.0
            # S-4: post-step barrier verification
            theta_m = worker_theta.get(m, 0.7)
            if MF_n > theta_m + 1e-6:
                barrier_violations += 1
            if MR_n < -1e-6:
                barrier_violations += 1
            self.states[worker_idx][m] = {"MR": MR_n, "MA": MA_n, "MF": MF_n}
        return ecbf_interventions, float(ecbf_clip_total), barrier_violations

    def set_task_assignments(self, assignments):
        """Set task assignments from the NSWF allocator (C-7.A).

        Args:
            assignments: dict {worker_idx: task_idx} or {agent_name: task_idx}
        """
        for key, task_idx in assignments.items():
            if isinstance(key, str):
                idx = int(key.split("_")[1])
            else:
                idx = int(key)
            self._task_assignments[idx] = task_idx

    def reset(self, seed=None):
        self.agents = list(self.possible_agents)
        self.current_step = 0
        self.states = {i: self._init_worker() for i in range(self.n_workers)}
        self._task_assignments = {i: 0 for i in range(self.n_workers)}
        obs = {a: self._get_obs(int(a.split("_")[1])) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions):
        rewards, infos = {}, {}
        for agent in self.agents:
            idx = int(agent.split("_")[1])

            if self.action_mode == "continuous":
                # C-8.A: actions[agent] is np.array of C_nom per muscle
                c_nom = np.asarray(actions[agent], dtype=np.float32)
                ecbf_interventions, ecbf_clip_total, barrier_violations = self._integrate_continuous(idx, c_nom)
                task_idx = self._task_assignments.get(idx, 0)
                task_name = self.task_names[task_idx]
            else:
                # Discrete: actions[agent] is task index
                task_name = self.task_names[actions[agent]]
                ecbf_interventions, ecbf_clip_total, barrier_violations = self._integrate(idx, task_name)

            fatigue = {m: self.states[idx][m]["MF"] for m in self.muscle_names}
            prod = self.task_mgr.task_intensity(task_name)
            worker_theta = self.theta_max_per_worker[idx]
            rewards[agent] = nswf_reward(prod, fatigue, worker_theta, kappa=self.kappa,
                                         disagreement_type=self.disagreement_type)
            cost = safety_cost(fatigue, worker_theta)
            violations = int(cost > 0)
            infos[agent] = {
                "task": task_name, "fatigue": fatigue,
                "violations": violations, "cost": cost,
                "ecbf_interventions": ecbf_interventions,
                "ecbf_clip_total": ecbf_clip_total,
                "barrier_violations": barrier_violations,  # S-4: discrete-time barrier check
            }
        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = {a: self._get_obs(int(a.split("_")[1])) for a in self.agents}
        terms = {a: done for a in self.agents}
        truncs = {a: False for a in self.agents}
        return obs, rewards, terms, truncs, infos
