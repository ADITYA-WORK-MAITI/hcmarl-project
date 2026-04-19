"""
Generate code flowcharts for hcmarl/ sub-packages: envs/, agents/, baselines/.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from flowchart_framework import FlowchartBuilder, HDR, CLS, FUNC, CONST, PROP, UTIL

OUT = "diagrams/code_flowcharts"
os.makedirs(OUT, exist_ok=True)


# =====================================================================
# envs/__init__.py
# =====================================================================
def gen_envs_init():
    fb = FlowchartBuilder("hcmarl/envs/__init__.py", "Environments package re-exports", lines="19",
        imports_desc="pettingzoo_wrapper, task_profiles, reward_functions")
    fb.make_node("imports", "Package re-exports", CONST, [
        ("WarehousePettingZoo", "from pettingzoo_wrapper"),
        ("TaskProfileManager", "from task_profiles"),
        ("nswf_reward", "from reward_functions"),
        ("safety_cost", "from reward_functions"),
        ("disagreement_utility", "from reward_functions"),
    ])
    fb.make_node("lazy", "__getattr__('WarehouseEnv')", FUNC, [
        ("Logic", "lazy import SingleWorkerWarehouseEnv"),
        ("Avoids", "circular dependency"),
    ])
    fb.edge("file_hdr", "imports", "", style="bold")
    fb.edge("file_hdr", "lazy", "", style="bold")
    fb.dangling_in("imports", "WarehousePettingZoo", "pettingzoo_wrapper.py")
    fb.dangling_in("imports", "TaskProfileManager", "task_profiles.py")
    fb.dangling_in("imports", "nswf_reward, safety_cost", "reward_functions.py")
    fb.dangling_out("imports", "public API", "scripts/train.py, tests/")
    fb.add_legend_entry("WarehouseEnv", "lazy alias for SingleWorkerWarehouseEnv", "gym.Env subclass")
    fb.render(OUT, stem_override="envs_init")


# =====================================================================
# envs/pettingzoo_wrapper.py
# =====================================================================
def gen_pettingzoo_wrapper():
    fb = FlowchartBuilder("hcmarl/envs/pettingzoo_wrapper.py",
        "PettingZoo ParallelEnv for N workers", lines="164",
        equations="Eqs 2-4, 19, 23, 32-33, 35",
        imports_desc="numpy, reward_functions, task_profiles, three_cc_r")
    fb.make_node("init", "WarehousePettingZoo.__init__", CLS, [
        ("n_workers", "4 (default)"), ("task_mgr", "TaskProfileManager()"),
        ("muscle_params", "from get_muscle() per muscle"), ("theta_max_per_worker", "per-worker thresholds"),
        ("ecbf_mode", "'on'|'off'"), ("obs_dim", "n_muscles*3 + 1"),
    ])
    fb.make_node("integrate", "_integrate(worker_idx, task_name)", FUNC, [
        ("Per muscle", "C_nom, ECBF clip, Euler ODE step"),
        ("Return", "(ecbf_interventions, ecbf_clip_total)"),
    ])
    fb.make_node("reset", "reset()", FUNC, [("Sets", "all workers MR=1,MA=0,MF=0"), ("Return", "(obs_dict, info_dict)")])
    fb.make_node("step", "step(actions)", FUNC, [
        ("Input", "dict[agent, action_int]"), ("Per agent", "_integrate + nswf_reward + safety_cost"),
        ("Return", "(obs, rewards, terms, truncs, infos)"),
    ])
    fb.make_node("obs_fns", "_get_obs(idx) / _get_global_obs()", FUNC, [
        ("Local", "[MR,MA,MF]*muscles + step_norm"), ("Global", "all workers concatenated"),
    ])
    fb.edge("file_hdr", "init", "", style="bold")
    fb.edge("init", "integrate", "self.muscle_params, theta_max", style="dashed")
    fb.edge("init", "obs_fns", "self.states", style="dashed")
    fb.edge("integrate", "step", "ecbf_interventions")
    fb.edge("obs_fns", "step", "obs per agent")
    fb.edge("reset", "step", "initialises state", style="dashed")
    fb.dangling_in("init", "TaskProfileManager", "task_profiles.py")
    fb.dangling_in("init", "get_muscle()", "three_cc_r.py")
    fb.dangling_in("step", "nswf_reward, safety_cost", "reward_functions.py")
    fb.dangling_out("step", "obs, rewards, infos", "agents/mappo.py, train.py")
    fb.add_legend_entry("actions", "dict agent->task_index", "{'worker_0':2, 'worker_1':0}")
    fb.add_legend_entry("obs", "[MR,MA,MF]*6 muscles + step_norm", "(19,) float32")
    fb.add_legend_entry("theta_max_per_worker", "per-worker per-muscle thresholds", "{0:{'shoulder':0.7}}")
    fb.add_legend_entry("ecbf_interventions", "ECBF clip applied per muscle per step", "{'shoulder':0.02}")
    fb.add_legend_entry("obs per agent", "local observation vector per worker", "(19,) float32")
    fb.add_legend_entry("self.muscle_params, theta_max", "muscle ODE params and safety limits", "MuscleParams + 0.7")
    fb.add_legend_entry("self.states", "array of MR, MA, MF per worker per muscle", "(4,6,3) float32")
    fb.add_legend_entry("initialises state", "resets all muscle states to MR=1, MA=0, MF=0", "ndarray zeros")
    fb.render(OUT)


# =====================================================================
# envs/reward_functions.py
# =====================================================================
def gen_reward_functions():
    fb = FlowchartBuilder("hcmarl/envs/reward_functions.py",
        "Canonical reward & cost functions (Eqs 31-33)", lines="94",
        imports_desc="math")
    fb.make_node("disagreement", "disagreement_utility(avg_mf, kappa)", FUNC, [
        ("Formula", "kappa * MF^2 / (1-MF) [Eq 32]"),
        ("Return", "float D_i"),
    ])
    fb.make_node("nswf_rew", "nswf_reward(productivity, fatigue, theta_max, ...)", FUNC, [
        ("surplus", "productivity - D_i(max_MF)"),
        ("reward", "log(max(surplus, eps)) - weight*violations"),
        ("Return", "float scalar reward"),
    ])
    fb.make_node("safety_c", "safety_cost(fatigue, theta_max)", FUNC, [
        ("Logic", "1.0 if any MF > theta_max, else 0.0"),
        ("Return", "float (binary cost signal)"),
    ])
    fb.edge("file_hdr", "disagreement", "", style="bold")
    fb.edge("file_hdr", "nswf_rew", "", style="bold")
    fb.edge("file_hdr", "safety_c", "", style="bold")
    fb.edge("disagreement", "nswf_rew", "D_i")
    fb.dangling_in("nswf_rew", "productivity (sum of task loads)", "warehouse_env.py / pettingzoo_wrapper.py")
    fb.dangling_in("nswf_rew", "fatigue dict {muscle: MF}", "warehouse_env.py")
    fb.dangling_out("nswf_rew", "reward float", "warehouse_env.py, pettingzoo_wrapper.py")
    fb.dangling_out("safety_c", "cost float", "warehouse_env.py, mappo_lag.py")
    fb.add_legend_entry("D_i", "disagreement utility kappa*MF^2/(1-MF)", "0.143")
    fb.add_legend_entry("productivity", "sum of task demands across muscles (U(i,j))", "1.2")
    fb.add_legend_entry("surplus", "productivity - D_i", "1.057")
    fb.add_legend_entry("violations", "count of muscles where MF > theta_max", "0")
    fb.render(OUT)


# =====================================================================
# envs/task_profiles.py
# =====================================================================
def gen_task_profiles():
    fb = FlowchartBuilder("hcmarl/envs/task_profiles.py",
        "Task demand profiles T_L,g (Def 7.1, Eq 34)", lines="67",
        imports_desc="numpy, yaml, pathlib")
    fb.make_node("profiles", "DEFAULT_PROFILES (class var)", CONST, [
        ("heavy_lift", "shoulder:0.45 knee:0.40 trunk:0.50 grip:0.55"),
        ("light_sort", "shoulder:0.10 elbow:0.15 grip:0.20"),
        ("carry", "shoulder:0.25 trunk:0.30 grip:0.45"),
        ("overhead_reach", "shoulder:0.55 elbow:0.35"),
        ("push_cart", "shoulder:0.20 trunk:0.25 grip:0.40"),
        ("rest", "all 0.00"),
    ])
    fb.make_node("mgr_init", "TaskProfileManager.__init__", CLS, [
        ("profiles", "dict or from YAML"), ("task_names", "list of 6 tasks"),
        ("muscle_names", "6 muscles"), ("n_tasks, n_muscles", "6, 6"),
    ])
    fb.make_node("get_demand", "get_demand(task, muscle)", FUNC, [("Return", "float TL")])
    fb.make_node("get_vec", "get_demand_vector(task)", FUNC, [("Return", "np.ndarray (6,)")])
    fb.make_node("get_mat", "get_demand_matrix()", FUNC, [("Return", "np.ndarray (M,G)")])
    fb.make_node("intensity", "task_intensity(task)", FUNC, [("Return", "sum of all muscle loads")])
    fb.edge("file_hdr", "profiles", "", style="bold")
    fb.edge("file_hdr", "mgr_init", "", style="bold")
    fb.edge("profiles", "mgr_init", "DEFAULT_PROFILES")
    fb.edge("mgr_init", "get_demand", "self.profiles", style="dashed")
    fb.edge("mgr_init", "get_vec", "self.profiles", style="dashed")
    fb.edge("mgr_init", "get_mat", "self.profiles", style="dashed")
    fb.edge("mgr_init", "intensity", "self.profiles", style="dashed")
    fb.dangling_in("mgr_init", "config YAML (optional)", "config/task_profiles.yaml")
    fb.dangling_out("get_vec", "demand_vector", "pettingzoo_wrapper.py")
    fb.dangling_out("intensity", "task_intensity", "pettingzoo_wrapper.py")
    fb.add_legend_entry("DEFAULT_PROFILES", "dict of task->muscle->TL load values", "{'heavy_lift':{'shoulder':0.45}}")
    fb.add_legend_entry("self.profiles", "active task demand profile dict", "{'heavy_lift':{'shoulder':0.45}}")
    fb.add_legend_entry("TL", "target load fraction of MVC [0,1]", "0.45")
    fb.add_legend_entry("demand_vector", "per-muscle TL for one task", "[0.45,0.10,0.40,0.30,0.50,0.55]")
    fb.add_legend_entry("demand_matrix", "all tasks x all muscles", "(6,6) float32")
    fb.render(OUT)


# =====================================================================
# agents/__init__.py
# =====================================================================
def gen_agents_init():
    fb = FlowchartBuilder("hcmarl/agents/__init__.py", "Agents package re-exports", lines="7",
        imports_desc="networks, mappo, mappo_lag, ippo, hcmarl_agent")
    fb.make_node("exports", "Package re-exports", CONST, [
        ("ActorNetwork", "from networks"), ("CriticNetwork", "from networks"),
        ("MAPPO", "from mappo"), ("MAPPOLagrangian", "from mappo_lag"),
        ("IPPO", "from ippo"), ("HCMARLAgent", "from hcmarl_agent"),
    ])
    fb.edge("file_hdr", "exports", "", style="bold")
    fb.dangling_in("exports", "all agent classes", "networks.py, mappo.py, etc.")
    fb.dangling_out("exports", "public API", "scripts/train.py")
    fb.render(OUT, stem_override="agents_init")


# =====================================================================
# agents/networks.py
# =====================================================================
def gen_networks():
    fb = FlowchartBuilder("hcmarl/agents/networks.py",
        "Actor-Critic neural networks (MLP)", lines="72",
        imports_desc="torch, torch.nn, numpy")
    fb.make_node("init_weights", "init_weights(module, gain)", FUNC, [
        ("Logic", "orthogonal init for Linear layers"),
    ])
    fb.make_node("actor", "class ActorNetwork (nn.Module)", CLS, [
        ("Architecture", "Linear(obs,64)->Tanh->Linear(64,64)->Tanh->Linear(64,n_actions)"),
        ("forward(obs)", "returns logits"),
        ("get_action(obs)", "Categorical sample -> (action, log_prob, entropy)"),
        ("evaluate(obs, action)", "-> (log_prob, entropy)"),
    ])
    fb.make_node("critic", "class CriticNetwork (nn.Module)", CLS, [
        ("Architecture", "Linear(global_obs,128)->Tanh->...->Linear(128,1)"),
        ("forward(state)", "returns V(s) scalar"),
    ])
    fb.make_node("cost_critic", "class CostCriticNetwork (nn.Module)", CLS, [
        ("Architecture", "same as CriticNetwork"),
        ("forward(state)", "returns V_c(s) scalar (cost value)"),
    ])
    fb.edge("file_hdr", "init_weights", "", style="bold")
    fb.edge("file_hdr", "actor", "", style="bold")
    fb.edge("file_hdr", "critic", "", style="bold")
    fb.edge("file_hdr", "cost_critic", "", style="bold")
    fb.edge("init_weights", "actor", "gain=sqrt(2)")
    fb.edge("init_weights", "critic", "gain=sqrt(2)")
    fb.dangling_in("actor", "obs tensor (obs_dim,)", "warehouse_env.py")
    fb.dangling_in("critic", "global_state tensor", "pettingzoo_wrapper.py")
    fb.dangling_out("actor", "ActorNetwork", "mappo.py, ippo.py, mappo_lag.py")
    fb.dangling_out("critic", "CriticNetwork", "mappo.py, ippo.py")
    fb.dangling_out("cost_critic", "CostCriticNetwork", "mappo_lag.py")
    fb.add_legend_entry("gain=sqrt(2)", "orthogonal init gain for hidden layers", "1.4142")
    fb.add_legend_entry("obs", "local observation vector", "(19,) float32")
    fb.add_legend_entry("global_state", "concatenated all-agent state", "(73,) float32")
    fb.add_legend_entry("logits", "unnormalized action log-probs", "(6,) float32")
    fb.add_legend_entry("V(s)", "centralised state value", "0.85")
    fb.add_legend_entry("V_c(s)", "cost value prediction", "0.12")
    fb.render(OUT)


# =====================================================================
# agents/mappo.py
# =====================================================================
def gen_mappo():
    fb = FlowchartBuilder("hcmarl/agents/mappo.py",
        "MAPPO: Multi-Agent PPO (Yu et al. NeurIPS 2022)", lines="278",
        imports_desc="torch, numpy, networks.ActorNetwork/CriticNetwork")
    fb.make_node("buffer", "class RolloutBuffer", CLS, [
        ("Storage", "per-agent: obs, actions, log_probs, rewards"),
        ("Shared", "global_states, dones, values"),
        ("store_step()", "store one timestep for ALL agents"),
        ("compute_returns()", "per-agent GAE -> flatten (T*N,)"),
        ("get_flat_tensors()", "obs, gs, acts, lps as tensors"),
    ])
    fb.make_node("mappo_init", "class MAPPO.__init__", CLS, [
        ("actor", "ActorNetwork (shared weights)"),
        ("critic", "CriticNetwork (centralised)"),
        ("buffer", "RolloutBuffer"),
        ("Hyperparams", "gamma=0.99, clip_eps=0.2, entropy=0.01"),
    ])
    fb.make_node("get_actions", "get_actions(observations, global_state)", FUNC, [
        ("Per agent", "actor.get_action(obs) -> action, log_prob"),
        ("Critic", "critic(global_state) -> value"),
        ("Return", "(actions_dict, log_probs_dict, value)"),
    ])
    fb.make_node("update", "update() — PPO training step", FUNC, [
        ("1", "get_flat_tensors from buffer"),
        ("2", "compute_returns (GAE)"),
        ("3", "n_epochs PPO loops: ratio, surr1, surr2"),
        ("Actor loss", "-min(surr1,surr2) - entropy_coeff*H"),
        ("Critic loss", "MSE(V(s), returns)"),
        ("Return", "{'actor_loss', 'critic_loss'}"),
    ])
    fb.make_node("save_load", "save(path) / load(path)", FUNC, [
        ("Saves", "actor + critic state_dicts"),
    ])
    fb.edge("file_hdr", "buffer", "", style="bold")
    fb.edge("file_hdr", "mappo_init", "", style="bold")
    fb.edge("buffer", "mappo_init", "self.buffer")
    fb.edge("mappo_init", "get_actions", "self.actor, self.critic", style="dashed")
    fb.edge("get_actions", "update", "obs, actions, log_probs, rewards")
    fb.edge("buffer", "update", "flat_obs, flat_acts, advantages, returns")
    fb.edge("mappo_init", "save_load", "state_dicts", style="dashed")
    fb.dangling_in("mappo_init", "ActorNetwork, CriticNetwork", "networks.py")
    fb.dangling_in("get_actions", "observations dict, global_state", "pettingzoo_wrapper.py")
    fb.dangling_out("get_actions", "actions dict", "pettingzoo_wrapper.py")
    fb.dangling_out("update", "loss metrics", "scripts/train.py")
    fb.dangling_out("save_load", "checkpoint .pt file", "checkpoints/")
    fb.add_legend_entry("obs, actions, log_probs, rewards", "transition tuple stored per step", "ndarray per agent")
    fb.add_legend_entry("flat_obs, flat_acts, advantages, returns", "flattened tensors for PPO loss", "(T*N,) float")
    fb.add_legend_entry("self.buffer", "RolloutBuffer instance on MAPPO", "RolloutBuffer(T=2048)")
    fb.add_legend_entry("self.actor, self.critic", "shared actor and centralised critic refs", "ActorNetwork, CriticNetwork")
    fb.add_legend_entry("state_dicts", "PyTorch network weight dicts", "OrderedDict")
    fb.add_legend_entry("advantages", "GAE-computed per agent, flattened", "(T*N,) float")
    fb.add_legend_entry("returns", "advantages + values", "(T*N,) float")
    fb.add_legend_entry("ratio", "exp(new_lp - old_lp)", "tensor ~1.0")
    fb.add_legend_entry("clip_eps", "PPO clipping epsilon", "0.2")
    fb.add_legend_entry("gamma", "discount factor", "0.99")
    fb.render(OUT)


# =====================================================================
# agents/ippo.py
# =====================================================================
def gen_ippo():
    fb = FlowchartBuilder("hcmarl/agents/ippo.py",
        "IPPO: Independent PPO (per-agent actor+critic)", lines="121",
        imports_desc="torch, numpy, networks, mappo.RolloutBuffer")
    fb.make_node("ippo_init", "class IPPO.__init__", CLS, [
        ("actors", "list[ActorNetwork] per agent"),
        ("critics", "list[CriticNetwork] per agent (LOCAL obs)"),
        ("buffers", "list[RolloutBuffer] per agent"),
    ])
    fb.make_node("get_act", "get_actions(observations)", FUNC, [
        ("Per agent i", "actors[i].get_action(obs)"),
        ("Return", "(actions_dict, log_probs_dict, 0.0)"),
    ])
    fb.make_node("store_tr", "store_transition(agent_idx, ...)", FUNC, [
        ("Stores", "obs, action, log_prob, reward, done per agent"),
    ])
    fb.make_node("update_ippo", "update() — per-agent PPO", FUNC, [
        ("Per agent", "compute_returns, PPO loop"),
        ("Return", "mean actor_loss, critic_loss"),
    ])
    fb.edge("file_hdr", "ippo_init", "", style="bold")
    fb.edge("ippo_init", "get_act", "self.actors", style="dashed")
    fb.edge("get_act", "store_tr", "obs, action, log_prob, reward")
    fb.edge("store_tr", "update_ippo", "buffer data")
    fb.dangling_in("ippo_init", "ActorNetwork, CriticNetwork", "networks.py")
    fb.dangling_in("ippo_init", "RolloutBuffer", "mappo.py")
    fb.dangling_out("get_act", "actions dict", "pettingzoo_wrapper.py")
    fb.add_legend_entry("obs, action, log_prob, reward", "per-agent transition stored each step", "ndarray, int, float, float")
    fb.add_legend_entry("buffer data", "full rollout tensors for one agent", "RolloutBuffer contents")
    fb.add_legend_entry("self.actors", "list of per-agent ActorNetwork instances", "list[ActorNetwork]")
    fb.add_legend_entry("actors[i]", "per-agent actor network", "ActorNetwork(19,6)")
    fb.add_legend_entry("critics[i]", "per-agent LOCAL critic", "CriticNetwork(19)")
    fb.render(OUT)


# =====================================================================
# agents/mappo_lag.py
# =====================================================================
def gen_mappo_lag():
    fb = FlowchartBuilder("hcmarl/agents/mappo_lag.py",
        "MAPPO-Lagrangian: MAPPO + cost critic + dual lambda", lines="285",
        imports_desc="torch, numpy, networks.*")
    fb.make_node("lag_buffer", "class LagrangianRolloutBuffer", CLS, [
        ("Extra field", "_costs per agent"),
        ("Extra field", "_cost_values"),
        ("compute_returns()", "reward GAE + cost GAE -> flatten"),
    ])
    fb.make_node("lag_init", "class MAPPOLagrangian.__init__", CLS, [
        ("actor", "ActorNetwork (shared)"), ("critic", "CriticNetwork"),
        ("cost_critic", "CostCriticNetwork"), ("log_lambda", "nn.Parameter (dual var)"),
        ("cost_limit", "0.1 (constraint threshold)"),
    ])
    fb.make_node("lag_actions", "get_actions(obs, global_state)", FUNC, [
        ("Return", "(actions, log_probs, value, cost_value)"),
    ])
    fb.make_node("lag_update", "update() — PPO + Lagrangian", FUNC, [
        ("Actor loss", "reward_clip + lambda * cost_clip - entropy"),
        ("Critic loss", "MSE(V, returns)"),
        ("Cost critic", "MSE(V_c, cost_returns)"),
    ])
    fb.make_node("update_lam", "update_lambda(mean_cost)", FUNC, [
        ("Dual ascent", "lambda += lr * (cost - limit)"),
    ])
    fb.edge("file_hdr", "lag_buffer", "", style="bold")
    fb.edge("file_hdr", "lag_init", "", style="bold")
    fb.edge("lag_buffer", "lag_init", "self.buffer")
    fb.edge("lag_init", "lag_actions", "actor, critic, cost_critic", style="dashed")
    fb.edge("lag_actions", "lag_update", "obs, actions, log_probs, rewards, costs")
    fb.edge("lag_update", "update_lam", "mean_episode_cost")
    fb.dangling_in("lag_init", "ActorNetwork, CriticNetwork, CostCriticNetwork", "networks.py")
    fb.dangling_out("lag_actions", "actions, value, cost_value", "pettingzoo_wrapper.py")
    fb.dangling_out("lag_update", "loss + lambda metrics", "scripts/train.py")
    fb.add_legend_entry("obs, actions, log_probs, rewards, costs", "transition + cost stored per step", "ndarray per agent")
    fb.add_legend_entry("mean_episode_cost", "average cost signal over episode", "0.07")
    fb.add_legend_entry("self.buffer", "LagrangianRolloutBuffer instance", "LagrangianRolloutBuffer(T=2048)")
    fb.add_legend_entry("actor, critic, cost_critic", "shared network refs on MAPPOLagrangian", "ActorNetwork, CriticNetwork, CostCriticNetwork")
    fb.add_legend_entry("log_lambda", "log of Lagrangian dual variable", "ln(0.5) = -0.69")
    fb.add_legend_entry("cost_limit", "max acceptable mean cost per step", "0.1")
    fb.add_legend_entry("cost_advantages", "GAE on cost stream", "(T*N,) float")
    fb.render(OUT)


# =====================================================================
# agents/hcmarl_agent.py
# =====================================================================
def gen_hcmarl_agent():
    fb = FlowchartBuilder("hcmarl/agents/hcmarl_agent.py",
        "Full HC-MARL agent: MAPPO+ECBF+NSWF", lines="35",
        imports_desc="torch, numpy, agents.mappo.MAPPO")
    fb.make_node("agent_init", "class HCMARLAgent.__init__", CLS, [
        ("mappo", "MAPPO instance"), ("theta_max", "per-muscle thresholds"),
        ("ecbf_alpha1,2,3", "ECBF gains"), ("use_nswf", "bool"),
    ])
    fb.make_node("agent_act", "get_actions(observations, global_state)", FUNC, [
        ("Delegates", "self.mappo.get_actions()"),
        ("Note", "ECBF filtering happens in env step"),
        ("Return", "(actions, log_probs, value)"),
    ])
    fb.make_node("agent_io", "save(path) / load(path)", FUNC, [("Delegates", "self.mappo.save/load")])
    fb.edge("file_hdr", "agent_init", "", style="bold")
    fb.edge("agent_init", "agent_act", "self.mappo", style="dashed")
    fb.edge("agent_init", "agent_io", "self.mappo", style="dashed")
    fb.dangling_in("agent_init", "MAPPO", "mappo.py")
    fb.dangling_out("agent_act", "actions dict", "pettingzoo_wrapper.py")
    fb.add_legend_entry("self.mappo", "wrapped MAPPO instance inside HCMARLAgent", "MAPPO(obs_dim=19)")
    fb.add_legend_entry("theta_max", "per-muscle safety thresholds", "{'shoulder':0.70}")
    fb.render(OUT)


# =====================================================================
# baselines/__init__.py
# =====================================================================
def gen_baselines_init():
    fb = FlowchartBuilder("hcmarl/baselines/__init__.py",
        "Baselines package: OmniSafe + SafePO + 10 legacy", lines="12",
        imports_desc="omnisafe_wrapper, safepo_wrapper, _legacy")
    fb.make_node("exports", "Package re-exports", CONST, [
        ("OmniSafeWrapper", "PPOLag/CPO/FOCOPS/CUP"), ("SafePOWrapper", "MACPO"),
        ("10 legacy baselines", "Random, RoundRobin, Greedy, GreedySafe, ..."),
        ("create_all_baselines()", "factory for all 10"),
    ])
    fb.edge("file_hdr", "exports", "", style="bold")
    fb.dangling_in("exports", "OmniSafeWrapper", "omnisafe_wrapper.py")
    fb.dangling_in("exports", "SafePOWrapper", "safepo_wrapper.py")
    fb.dangling_in("exports", "10 baselines", "_legacy.py")
    fb.dangling_out("exports", "all baseline classes", "scripts/train.py")
    fb.render(OUT, stem_override="baselines_init")


# =====================================================================
# baselines/_legacy.py
# =====================================================================
def gen_baselines_legacy():
    fb = FlowchartBuilder("hcmarl/baselines/_legacy.py",
        "10 baseline methods for comparison", lines="324",
        imports_desc="numpy")
    classes = [
        ("Random", "uniform random task selection"),
        ("RoundRobin", "cyclic task assignment"),
        ("Greedy", "always highest-productivity task"),
        ("GreedySafe", "greedy + rest if MF > threshold"),
        ("PPOUnconstrained", "single-agent PPO, no safety"),
        ("PPOLagrangian", "PPO + lambda * cost penalty"),
        ("MAPPONoFilter", "MAPPO without ECBF"),
        ("MAPPOLagrangian", "extends PPOLagrangian"),
        ("MACPO", "trust-region + cost budget"),
        ("FixedSchedule", "work/rest cycle (industry std)"),
    ]
    rows = [(name, desc) for name, desc in classes]
    fb.make_node("baselines", "10 Baseline Classes", CLS, rows)
    fb.make_node("factory", "create_all_baselines(obs_dim, n_actions, n_muscles)", FUNC, [
        ("Return", "list of all 10 instantiated baselines"),
    ])
    fb.make_node("interface", "Common interface: get_actions(observations)", FUNC, [
        ("Input", "dict[agent, obs_array]"),
        ("Return", "dict[agent, action_int]"),
    ])
    fb.edge("file_hdr", "baselines", "", style="bold")
    fb.edge("file_hdr", "factory", "", style="bold")
    fb.edge("baselines", "factory", "obs_dim, n_actions, n_muscles")
    fb.edge("baselines", "interface", "observations")
    fb.dangling_in("interface", "observations from env", "pettingzoo_wrapper.py")
    fb.dangling_out("interface", "actions dict", "pettingzoo_wrapper.py")
    fb.dangling_out("factory", "baseline instances", "scripts/train.py")
    fb.add_legend_entry("obs_dim, n_actions, n_muscles", "constructor args passed to all 10 baselines", "19, 6, 6")
    fb.add_legend_entry("observations", "dict of agent->obs_array passed to get_actions", "{'worker_0': ndarray(19,)}")
    fb.add_legend_entry("get_actions()", "common method across all baselines", "{'worker_0':2,...}")
    fb.add_legend_entry("mf_threshold", "GreedySafe rest trigger", "0.5")
    fb.add_legend_entry("cost_budget", "MACPO per-step constraint", "0.1")
    fb.render(OUT)


# =====================================================================
# baselines/omnisafe_wrapper.py
# =====================================================================
def gen_omnisafe_wrapper():
    fb = FlowchartBuilder("hcmarl/baselines/omnisafe_wrapper.py",
        "OmniSafe wrapper: PPOLag/CPO/FOCOPS/CUP", lines="242",
        imports_desc="numpy, json, pathlib, omnisafe (optional)")
    fb.make_node("wrapper_init", "class OmniSafeWrapper.__init__", CLS, [
        ("algo_name", "'PPOLag' (or CPO/FOCOPS/CUP)"),
        ("agent", "omnisafe.Agent (if installed)"),
        ("Fallback", "random policy if omnisafe missing"),
    ])
    fb.make_node("get_act_os", "get_actions(observations)", FUNC, [
        ("If omnisafe", "agent.predict(obs)"),
        ("Fallback", "random action"),
    ])
    fb.make_node("train_os", "train(total_steps)", FUNC, [("Calls", "agent.learn()")])
    fb.make_node("benchmark", "run_omnisafe_benchmark(algo, env_id, seeds)", FUNC, [
        ("Multi-seed", "train + evaluate per seed"),
        ("Saves", "results.json"),
    ])
    fb.edge("file_hdr", "wrapper_init", "", style="bold")
    fb.edge("file_hdr", "benchmark", "", style="bold")
    fb.edge("wrapper_init", "get_act_os", "self.agent", style="dashed")
    fb.edge("wrapper_init", "train_os", "self.agent", style="dashed")
    fb.dangling_in("wrapper_init", "omnisafe library (optional)", "pip install omnisafe")
    fb.dangling_out("get_act_os", "actions", "scripts/train.py")
    fb.add_legend_entry("self.agent", "omnisafe.Agent instance or None", "omnisafe.Agent('PPOLag')")
    fb.add_legend_entry("algo_name", "OmniSafe algorithm identifier", "'PPOLag'")
    fb.render(OUT)


# =====================================================================
# baselines/safepo_wrapper.py
# =====================================================================
def gen_safepo_wrapper():
    fb = FlowchartBuilder("hcmarl/baselines/safepo_wrapper.py",
        "SafePO MACPO wrapper (fallback: MAPPOLagrangian)", lines="116",
        imports_desc="numpy, torch, agents.mappo_lag")
    fb.make_node("sp_init", "class SafePOWrapper.__init__", CLS, [
        ("Try", "import safepo -> native MACPO"),
        ("Fallback", "MAPPOLagrangian (same algo family)"),
    ])
    fb.make_node("sp_act", "get_actions(observations, global_state)", FUNC, [
        ("If safepo", "native predict"), ("Fallback", "MAPPOLag.get_actions()"),
    ])
    fb.make_node("sp_update", "update() / update_lambda()", FUNC, [
        ("Delegates", "to fallback MAPPOLagrangian"),
    ])
    fb.edge("file_hdr", "sp_init", "", style="bold")
    fb.edge("sp_init", "sp_act", "self._fallback or _safepo_agent", style="dashed")
    fb.edge("sp_act", "sp_update", "obs, actions, rewards, costs")
    fb.dangling_in("sp_init", "MAPPOLagrangian", "agents/mappo_lag.py")
    fb.dangling_in("sp_init", "safepo (optional)", "pip install safepo")
    fb.dangling_out("sp_act", "actions, value, cost_value", "scripts/train.py")
    fb.add_legend_entry("obs, actions, rewards, costs", "transition data buffered after each step", "ndarray per agent")
    fb.add_legend_entry("self._fallback or _safepo_agent", "active backend: MAPPOLagrangian or native SafePO", "MAPPOLagrangian instance")
    fb.add_legend_entry("_safepo_available", "whether native SafePO is installed", "True/False")
    fb.render(OUT)


# =====================================================================
# hcmarl/__init__.py
# =====================================================================
def gen_hcmarl_init():
    fb = FlowchartBuilder("hcmarl/__init__.py", "Package root re-exports", lines="~10",
        imports_desc="three_cc_r, ecbf_filter, nswf_allocator, pipeline, etc.")
    fb.make_node("root_exports", "Package public API", CONST, [
        ("Core", "ThreeCCr, MuscleParams, ThreeCCrState"),
        ("Safety", "ECBFFilter, ECBFParams"),
        ("Allocation", "NSWFAllocator, NSWFParams"),
        ("Pipeline", "HCMARLPipeline"),
        ("MMICRL", "MMICRL, CFDE, DemonstrationCollector"),
    ])
    fb.edge("file_hdr", "root_exports", "", style="bold")
    fb.dangling_in("root_exports", "all core modules", "three_cc_r.py, ecbf_filter.py, ...")
    fb.dangling_out("root_exports", "import hcmarl", "scripts/, tests/, notebooks/")
    fb.render(OUT, stem_override="hcmarl_init")


if __name__ == "__main__":
    generators = [
        gen_envs_init, gen_pettingzoo_wrapper, gen_reward_functions, gen_task_profiles,
        gen_agents_init, gen_networks, gen_mappo, gen_ippo, gen_mappo_lag, gen_hcmarl_agent,
        gen_baselines_init, gen_baselines_legacy, gen_omnisafe_wrapper, gen_safepo_wrapper,
        gen_hcmarl_init,
    ]
    for fn in generators:
        try:
            fn()
        except Exception as e:
            print(f"FAIL: {fn.__name__}: {e}")
