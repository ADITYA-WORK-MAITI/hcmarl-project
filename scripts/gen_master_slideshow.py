"""
Generate an interactive HTML slideshow of the master inter-file data flow.

Slide 0: Just the 28 boxes (no arrows)
Slide 1: boxes + arrow #1
Slide 2: boxes + arrows #1-2
...
Slide N: the full diagram (all arrows visible)

Left/Right arrow keys navigate. Current arrow highlights in red.
Legend panel on the right shows all arrows, highlighting the current one.
"""

import json, os

OUT = "diagrams/code_flowcharts"
os.makedirs(OUT, exist_ok=True)

# ==================================================================
# DATA (same as gen_master_variants.py)
# ==================================================================

BOXES = [
    {"id": "utils", "title": "utils.py", "sub": "Logging, YAML, seeding, safe_log, clip", "color": "#1E3A5F", "group": "core"},
    {"id": "three_cc_r", "title": "three_cc_r.py", "sub": "3CC-r fatigue ODE (Eqs 1-8)", "color": "#1E3A5F", "group": "core"},
    {"id": "ecbf", "title": "ecbf_filter.py", "sub": "ECBF safety filter CBF-QP (Eqs 12-23)", "color": "#1E3A5F", "group": "core"},
    {"id": "nswf", "title": "nswf_allocator.py", "sub": "NSWF allocator (Eqs 31-33)", "color": "#1E3A5F", "group": "core"},
    {"id": "pipeline", "title": "pipeline.py", "sub": "End-to-end HC-MARL pipeline", "color": "#1E3A5F", "group": "core"},
    {"id": "mmicrl", "title": "mmicrl.py", "sub": "MM-ICRL CFDE flows (Eqs 9-11)", "color": "#1E3A5F", "group": "core"},
    {"id": "real_data", "title": "real_data_calibration.py", "sub": "Path G: WSD4FEDSRM calibration", "color": "#1E3A5F", "group": "core"},
    {"id": "warehouse", "title": "warehouse_env.py", "sub": "Single/Multi-agent warehouse env", "color": "#1E3A5F", "group": "env"},
    {"id": "logger", "title": "logger.py", "sub": "HCMARLLogger + W&B", "color": "#1E3A5F", "group": "core"},
    {"id": "pettingzoo", "title": "pettingzoo_wrapper.py", "sub": "PettingZoo parallel env (N workers)", "color": "#2D4A22", "group": "env"},
    {"id": "reward_fn", "title": "reward_functions.py", "sub": "nswf_reward() + safety_cost()", "color": "#2D4A22", "group": "env"},
    {"id": "task_prof", "title": "task_profiles.py", "sub": "TaskProfileManager", "color": "#2D4A22", "group": "env"},
    {"id": "networks", "title": "networks.py", "sub": "Actor, Critic, CostCritic nets", "color": "#2D4A22", "group": "agents"},
    {"id": "mappo", "title": "mappo.py", "sub": "MAPPO + RolloutBuffer", "color": "#2D4A22", "group": "agents"},
    {"id": "ippo", "title": "ippo.py", "sub": "IPPO (independent PPO)", "color": "#2D4A22", "group": "agents"},
    {"id": "mappo_lag", "title": "mappo_lag.py", "sub": "MAPPO-Lagrangian", "color": "#2D4A22", "group": "agents"},
    {"id": "hcmarl_agent", "title": "hcmarl_agent.py", "sub": "HCMARLAgent (pipeline-aware)", "color": "#2D4A22", "group": "agents"},
    {"id": "legacy", "title": "_legacy.py", "sub": "10 baseline classes", "color": "#2D4A22", "group": "agents"},
    {"id": "omnisafe_w", "title": "omnisafe_wrapper.py", "sub": "OmniSafe wrapper", "color": "#2D4A22", "group": "agents"},
    {"id": "safepo_w", "title": "safepo_wrapper.py", "sub": "SafePO wrapper", "color": "#2D4A22", "group": "agents"},
    {"id": "train", "title": "scripts/train.py", "sub": "Main training: all 7 methods", "color": "#6B3A1E", "group": "scripts"},
    {"id": "evaluate", "title": "scripts/evaluate.py", "sub": "Evaluation: 9 metrics", "color": "#6B3A1E", "group": "scripts"},
    {"id": "run_abl", "title": "scripts/run_ablations.py", "sub": "Ablation launcher", "color": "#6B3A1E", "group": "scripts"},
    {"id": "run_base", "title": "scripts/run_baselines.py", "sub": "Baseline launcher", "color": "#6B3A1E", "group": "scripts"},
    {"id": "run_scale", "title": "scripts/run_scaling.py", "sub": "Scaling launcher", "color": "#6B3A1E", "group": "scripts"},
    {"id": "configs", "title": "config/*.yaml (20 files)", "sub": "Training, method, ablation, scaling configs", "color": "#4A2A5C", "group": "infra"},
    {"id": "notebooks", "title": "notebooks/*.ipynb (4)", "sub": "Colab training notebooks", "color": "#4A2A5C", "group": "infra"},
    {"id": "tests", "title": "tests/*.py (13 files)", "sub": "223 pytest tests", "color": "#4A2A5C", "group": "infra"},
]

# Chronological order: foundation → derived → integration → training → experiments → tests
EDGES = [
    # Phase A Layer 1: Foundation utilities
    {"from": "EXT_logging", "to": "utils", "label": "logging, yaml.safe_load", "desc": "Python logging + YAML", "ex": "yaml.safe_load(f)", "type": "dangling_in"},
    {"from": "EXT_numpy", "to": "three_cc_r", "label": "np.array, solve_ivp, dataclass", "desc": "NumPy, SciPy ODE, dataclass", "ex": "np.array([1,3,5])", "type": "dangling_in"},
    # Phase A Layer 2: Core math modules
    {"from": "utils", "to": "nswf", "label": "safe_log", "desc": "Numerically stable log utility (nswf_allocator.py:26)", "ex": "safe_log(0.0) = -46.05", "type": "internal"},
    {"from": "utils", "to": "pipeline", "label": "load_yaml, get_logger", "desc": "YAML loader + logger factory (pipeline.py:33)", "ex": "load_yaml('config/default_config.yaml')", "type": "internal"},
    {"from": "EXT_cvxpy", "to": "ecbf", "label": "cp.Variable, cp.Problem", "desc": "CVXPY optimisation primitives", "ex": "cp.Variable(3)", "type": "dangling_in"},
    {"from": "three_cc_r", "to": "ecbf", "label": "MuscleParams, ThreeCCrState", "desc": "Muscle dataclass + ODE state", "ex": "MuscleParams(F=0.0146, R=0.00058)  # SHOULDER", "type": "internal"},
    {"from": "three_cc_r", "to": "pipeline", "label": "ThreeCCr, get_muscle, SHOULDER", "desc": "ODE solver + factory + preset", "ex": "ThreeCCr(SHOULDER).simulate(60)", "type": "internal"},
    # Phase A Layer 3: Pipeline integration
    {"from": "ecbf", "to": "pipeline", "label": "ECBFFilter, ECBFParams", "desc": "CBF-QP filter class + params dataclass (pipeline.py:24)", "ex": "ECBFFilter(params, muscle, state)", "type": "internal"},
    {"from": "pipeline", "to": "ecbf", "label": "C_nominal", "desc": "Unfiltered RL/baseline command", "ex": "array([1.0, 0.9, 0.7])", "type": "internal"},
    {"from": "nswf", "to": "pipeline", "label": "NSWFAllocator, NSWFParams, AllocationResult", "desc": "Allocator class + params + result dataclass (pipeline.py:25)", "ex": "AllocationResult(assignments=[0,1,2])", "type": "internal"},
    {"from": "pipeline", "to": "nswf", "label": "utility_matrix, fatigue_levels", "desc": "Agent-task utils + fatigue", "ex": "U:(3,5), f:(3,) arrays", "type": "internal"},
    # Phase A Layer 4: MMICRL + Real data
    {"from": "EXT_torch", "to": "mmicrl", "label": "nn.Module, nn.Linear", "desc": "PyTorch NN base classes", "ex": "nn.Linear(19, 64)", "type": "dangling_in"},
    {"from": "EXT_wsd", "to": "real_data", "label": "WSD4FEDSRM/ CSV files", "desc": "Real fatigue dataset (34 subjects)", "ex": "demographic.csv + Borg/MVIC/EMG/IMU/PPG/KSS folders", "type": "dangling_in"},
    {"from": "three_cc_r", "to": "real_data", "label": "MuscleParams, ThreeCCr, SHOULDER", "desc": "Muscle params + ODE + preset", "ex": "SHOULDER.F = 0.0146", "type": "internal"},
    {"from": "real_data", "to": "mmicrl", "label": "demo trajectories", "desc": "Path G calibrated demos injected via real_data.load_path_g_into_collector() into mmicrl.DemonstrationCollector (lazy import, real_data_calibration.py:962)", "ex": "[(state=[MR,MA,MF], action=task_idx), ...]", "type": "internal"},
    {"from": "mmicrl", "to": "pettingzoo", "label": "theta_max dict (per worker, per muscle)", "desc": "MMICRL learned per-type fatigue thresholds; routed via train.py config into WarehousePettingZoo constructor (train.py:201-243)", "ex": "{'worker_0': {'shoulder': 0.72, 'ankle': 0.80, ...}}", "type": "internal"},
    # Phase A Layer 5: Environment
    {"from": "EXT_gym", "to": "warehouse", "label": "gym.Env, spaces", "desc": "Gymnasium env + spaces", "ex": "spaces.Discrete(4)  # SingleWorkerWarehouseEnv", "type": "dangling_in"},
    {"from": "warehouse", "to": "reward_fn", "label": "productivity, fatigue dict", "desc": "Task load + per-muscle MF passed to nswf_reward()/safety_cost() (warehouse_env.py:17)", "ex": "prod=5.2, {'shoulder':0.6}", "type": "internal"},
    {"from": "reward_fn", "to": "warehouse", "label": "reward float, cost float", "desc": "NSWF reward + safety cost", "ex": "r=3.14, c=0.02", "type": "internal"},
    {"from": "reward_fn", "to": "pettingzoo", "label": "nswf_reward, safety_cost", "desc": "Reward/cost callables", "ex": "nswf_reward(prod, fatigue)", "type": "internal"},
    {"from": "configs", "to": "task_prof", "label": "config YAML", "desc": "Parsed YAML config dict", "ex": "task_profiles.yaml", "type": "internal"},
    {"from": "task_prof", "to": "pettingzoo", "label": "demand_vector, task_intensity", "desc": "Per-task demands + intensity", "ex": "array([0.3,0.6,0.1]), 0.8", "type": "internal"},
    {"from": "three_cc_r", "to": "pettingzoo", "label": "get_muscle()", "desc": "Factory returning MuscleParams", "ex": "get_muscle('shoulder')", "type": "internal"},
    {"from": "warehouse", "to": "mmicrl", "label": "env, policy", "desc": "Env + policy for demos", "ex": "(gym.Env, nn.Module)", "type": "internal"},
    # Phase A Layer 6: Neural networks + Agents
    {"from": "pettingzoo", "to": "networks", "label": "obs + global_state tensors (via MAPPO/IPPO/MAPPO-Lag)", "desc": "Per-worker obs shape (n_muscles*3+1,) + centralised global_state shape (n_workers*n_muscles*3+1,); flow through agent classes that instantiate networks", "ex": "obs (19,), global_state (55,)  # n=3, n_muscles=6", "type": "internal"},
    {"from": "networks", "to": "mappo", "label": "ActorNetwork, CriticNetwork", "desc": "Actor + Critic nets", "ex": "ActorNetwork(19, 6)", "type": "internal"},
    {"from": "networks", "to": "ippo", "label": "ActorNetwork, CriticNetwork", "desc": "Actor + Critic nets", "ex": "ActorNetwork(19, 6)", "type": "internal"},
    {"from": "networks", "to": "mappo_lag", "label": "ActorNetwork, CriticNetwork, CostCriticNetwork", "desc": "All 3 nets", "ex": "CostCriticNetwork(55)", "type": "internal"},
    {"from": "pettingzoo", "to": "mappo", "label": "observations dict, global_state", "desc": "Per-agent obs + shared state", "ex": "{'worker_0': tensor, ...}", "type": "internal"},
    {"from": "mappo", "to": "pettingzoo", "label": "actions dict", "desc": "Per-agent actions", "ex": "{'worker_0':2, 'worker_1':0, 'worker_2':5}", "type": "internal"},
    {"from": "ippo", "to": "pettingzoo", "label": "actions dict", "desc": "Per-agent actions (IPPO)", "ex": "{'worker_0':2}", "type": "internal"},
    {"from": "mappo_lag", "to": "pettingzoo", "label": "actions, value, cost_value", "desc": "Actions + V(s) + Vc(s)", "ex": "acts, 3.14, 0.02", "type": "internal"},
    {"from": "mappo", "to": "ippo", "label": "RolloutBuffer", "desc": "Shared rollout buffer", "ex": "RolloutBuffer(capacity=2048)", "type": "internal"},
    {"from": "mappo", "to": "hcmarl_agent", "label": "MAPPO", "desc": "MAPPO agent class", "ex": "MAPPO(obs_dim=19, global_obs_dim=55, n_actions=6, n_agents=3)", "type": "internal"},
    {"from": "hcmarl_agent", "to": "pettingzoo", "label": "actions dict", "desc": "Per-agent actions (HCMARLAgent)", "ex": "{'worker_0':2}", "type": "internal"},
    {"from": "pettingzoo", "to": "legacy", "label": "observations", "desc": "Raw obs for baselines", "ex": "{'worker_0': array([...])}", "type": "internal"},
    {"from": "legacy", "to": "pettingzoo", "label": "actions dict", "desc": "Per-agent actions (baselines)", "ex": "{'worker_0':2}", "type": "internal"},
    {"from": "EXT_omnisafe", "to": "omnisafe_w", "label": "omnisafe library", "desc": "OmniSafe safe RL lib", "ex": "omnisafe.Agent('PPOLag')", "type": "dangling_in"},
    {"from": "EXT_safepo", "to": "safepo_w", "label": "safepo library", "desc": "SafePO safe MARL lib", "ex": "safepo.MAPPO_Lagrangian()", "type": "dangling_in"},
    {"from": "mappo_lag", "to": "safepo_w", "label": "MAPPOLagrangian", "desc": "Lagrangian agent class", "ex": "MAPPOLagrangian(...)", "type": "internal"},
    # Phase B Layer 7: Training scripts
    {"from": "EXT_wandb", "to": "logger", "label": "wandb (lazy)", "desc": "Weights & Biases tracking (lazy-imported inside HCMARLLogger.__init__)", "ex": "wandb.init(project='hcmarl')", "type": "dangling_in"},
    {"from": "configs", "to": "train", "label": "env params, PPO hyperparams, algo settings", "desc": "Full training config parsed via yaml.safe_load", "ex": "lr=3e-4, n_workers=3, method='hcmarl'", "type": "internal"},
    {"from": "configs", "to": "pipeline", "label": "ECBF params, NSWF params", "desc": "Default safety/alloc params (pipeline.from_config uses load_yaml)", "ex": "gamma_1=1.0, alpha=0.5", "type": "internal"},
    {"from": "pettingzoo", "to": "train", "label": "WarehousePettingZoo", "desc": "PettingZoo parallel env class (train.py:25)", "ex": "WarehousePettingZoo(n_workers=3)", "type": "internal"},
    {"from": "hcmarl_agent", "to": "train", "label": "HCMARLAgent", "desc": "Pipeline-aware agent", "ex": "HCMARLAgent(obs_dim=19, global_obs_dim=55, n_actions=6, n_agents=3, theta_max={...}, ecbf_params={...})", "type": "internal"},
    {"from": "mappo", "to": "train", "label": "MAPPO, loss metrics", "desc": "MAPPO + training losses", "ex": "policy_loss=0.03", "type": "internal"},
    {"from": "ippo", "to": "train", "label": "IPPO", "desc": "Independent PPO agent", "ex": "IPPO(obs_dim=19, n_actions=6, n_agents=3)", "type": "internal"},
    {"from": "mappo_lag", "to": "train", "label": "MAPPOLagrangian, loss + lambda", "desc": "Lagrangian + dual var", "ex": "lambda=0.5", "type": "internal"},
    {"from": "omnisafe_w", "to": "train", "label": "OmniSafeWrapper, actions", "desc": "OmniSafe wrapper + acts", "ex": "OmniSafeWrapper('PPOLag')", "type": "internal"},
    {"from": "safepo_w", "to": "train", "label": "SafePOWrapper, actions", "desc": "SafePO wrapper + acts", "ex": "SafePOWrapper(mappo_lag)", "type": "internal"},
    {"from": "mmicrl", "to": "train", "label": "DemonstrationCollector, MMICRL (lazy)", "desc": "Demo collector + CFDE model (lazy-imported inside run_mmicrl_pretrain, train.py:120)", "ex": "MMICRL(obs_dim=19)", "type": "internal"},
    {"from": "logger", "to": "train", "label": "HCMARLLogger", "desc": "Logger class (train.py:32)", "ex": "HCMARLLogger(log_dir)", "type": "internal"},
    {"from": "train", "to": "logger", "label": "episode_data", "desc": "Per-episode metrics via logger.log_episode()", "ex": "{'reward':4.2, 'cost':0.01}", "type": "internal"},
    # Phase B Layer 8: Evaluation
    {"from": "train", "to": "evaluate", "label": "create_agent()", "desc": "Agent factory re-imported by evaluate (evaluate.py:22)", "ex": "create_agent('hcmarl', obs_dim, ...)", "type": "internal"},
    {"from": "pettingzoo", "to": "evaluate", "label": "WarehousePettingZoo", "desc": "PettingZoo env class (evaluate.py:20)", "ex": "WarehousePettingZoo(n_workers=3)", "type": "internal"},
    {"from": "logger", "to": "evaluate", "label": "HCMARLLogger.METRIC_NAMES", "desc": "9 eval metric names defined on HCMARLLogger (evaluate.py:21)", "ex": "['violation_rate','cumulative_cost',...]", "type": "internal"},
    # Phase B Layer 9: Experiment launchers
    {"from": "configs", "to": "run_abl", "label": "ablation configs", "desc": "5 ablation YAML files", "ex": "ablation_no_ecbf.yaml", "type": "internal"},
    {"from": "configs", "to": "run_base", "label": "per-method configs", "desc": "6 per-method YAML files", "ex": "mappo_config.yaml", "type": "internal"},
    {"from": "configs", "to": "run_scale", "label": "scaling configs", "desc": "5 scaling YAML files", "ex": "scaling_n3.yaml", "type": "internal"},
    {"from": "run_abl", "to": "train", "label": "ablation_name, config_path", "desc": "Ablation ID + path", "ex": "'no_ecbf', 'config/...'", "type": "internal"},
    {"from": "run_base", "to": "train", "label": "method, config_path", "desc": "Method + path", "ex": "'mappo', 'config/...'", "type": "internal"},
    {"from": "run_scale", "to": "train", "label": "n_workers, config_path", "desc": "N + path", "ex": "6, 'config/scaling_n6.yaml'", "type": "internal"},
    # Phase B Layer 10: Notebooks
    {"from": "configs", "to": "notebooks", "label": "YAML configs", "desc": "Configs consumed by notebooks", "ex": "hcmarl_full_config.yaml", "type": "internal"},
    {"from": "train", "to": "notebooks", "label": "scripts/train.py", "desc": "Training script path", "ex": "!python scripts/train.py", "type": "internal"},
    {"from": "run_base", "to": "notebooks", "label": "scripts/run_baselines.py", "desc": "Baseline launcher path", "ex": "!python scripts/run_baselines.py", "type": "internal"},
    {"from": "run_abl", "to": "notebooks", "label": "scripts/run_ablations.py", "desc": "Ablation launcher path", "ex": "!python scripts/run_ablations.py", "type": "internal"},
    {"from": "run_scale", "to": "notebooks", "label": "scripts/run_scaling.py", "desc": "Scaling launcher path", "ex": "!python scripts/run_scaling.py", "type": "internal"},
    # Dangling outputs
    {"from": "train", "to": "EXT_ckpt1", "label": "checkpoints .pt", "desc": "Saved model weights", "ex": "checkpoint_final.pt", "type": "dangling_out"},
    {"from": "train", "to": "EXT_logs1", "label": "summary.json, config.yaml", "desc": "Training summary + config", "ex": "{'best_reward':8.3}", "type": "dangling_out"},
    {"from": "evaluate", "to": "EXT_results", "label": "results JSON", "desc": "Eval metrics per method", "ex": "hcmarl_eval.json", "type": "dangling_out"},
    {"from": "mappo", "to": "EXT_ckpt2", "label": "checkpoint .pt file", "desc": "MAPPO saved weights", "ex": "mappo_seed0.pt", "type": "dangling_out"},
    {"from": "logger", "to": "EXT_logcsv", "label": "training_log.csv", "desc": "Per-episode metric log", "ex": "episode,reward,cost,...", "type": "dangling_out"},
    {"from": "logger", "to": "EXT_wandb2", "label": "W&B dashboard", "desc": "Live W&B dashboard", "ex": "wandb.ai/run/hcmarl-001", "type": "dangling_out"},
    {"from": "notebooks", "to": "EXT_drive", "label": "checkpoints + logs", "desc": "Colab -> Google Drive", "ex": "Drive/hcmarl/checkpoints/", "type": "dangling_out"},
    {"from": "notebooks", "to": "EXT_evalj", "label": "eval JSONs", "desc": "Eval result files", "ex": "hcmarl_eval.json", "type": "dangling_out"},
    # Tests (last — they consume from everything)
    {"from": "three_cc_r", "to": "tests", "label": "SHOULDER, ThreeCCr, MuscleParams, ...", "desc": "Core 3CC-r types", "ex": "SHOULDER.F = 0.0146", "type": "test"},
    {"from": "ecbf", "to": "tests", "label": "ECBFParams, ECBFFilter", "desc": "Filter classes", "ex": "ECBFFilter(params, state)", "type": "test"},
    {"from": "nswf", "to": "tests", "label": "NSWFAllocator, NSWFParams, AllocationResult", "desc": "Allocator classes", "ex": "NSWFAllocator(NSWFParams())", "type": "test"},
    {"from": "pipeline", "to": "tests", "label": "HCMARLPipeline, TaskProfile, WorkerState", "desc": "Pipeline + data classes", "ex": "HCMARLPipeline.from_config()", "type": "test"},
    {"from": "mmicrl", "to": "tests", "label": "CFDE, _MADE, MMICRL, ...", "desc": "Flow model + collector", "ex": "CFDE(input_dim=18, n_types=3)", "type": "test"},
    {"from": "real_data", "to": "tests", "label": "predict_endurance_time, POPULATION_FR, ...", "desc": "Calibration fn + constants", "ex": "POPULATION_FR['shoulder']=(0.0146, 0.00058)", "type": "test"},
    {"from": "warehouse", "to": "tests", "label": "SingleWorkerWarehouseEnv, ...", "desc": "Env classes", "ex": "SingleWorkerWarehouseEnv()", "type": "test"},
    {"from": "pettingzoo", "to": "tests", "label": "WarehousePettingZoo", "desc": "PettingZoo env", "ex": "WarehousePettingZoo(n_workers=3)", "type": "test"},
    {"from": "hcmarl_agent", "to": "tests", "label": "HCMARLAgent", "desc": "Pipeline-aware agent", "ex": "HCMARLAgent(obs_dim=19, global_obs_dim=55, n_actions=6, n_agents=3, theta_max={...}, ecbf_params={...})", "type": "test"},
    {"from": "logger", "to": "tests", "label": "HCMARLLogger", "desc": "Logger class", "ex": "HCMARLLogger(log_dir)", "type": "test"},
]


def build():
    EXT_LABELS = {
        "EXT_logging": "Python stdlib (logging, yaml)",
        "EXT_numpy": "NumPy / SciPy",
        "EXT_cvxpy": "CVXPY",
        "EXT_torch": "PyTorch",
        "EXT_wsd": "WSD4FEDSRM dataset",
        "EXT_gym": "Gymnasium",
        "EXT_omnisafe": "OmniSafe",
        "EXT_safepo": "SafePO",
        "EXT_wandb": "W&B",
        "EXT_ckpt1": "checkpoints/ (.pt)",
        "EXT_logs1": "logs/ (summary.json)",
        "EXT_results": "results/ (eval JSON)",
        "EXT_ckpt2": "checkpoints/ (MAPPO .pt)",
        "EXT_logcsv": "training_log.csv",
        "EXT_wandb2": "W&B dashboard",
        "EXT_drive": "Google Drive",
        "EXT_evalj": "eval JSONs",
    }

    html = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>HC-MARL Inter-File Data Flow</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:#FAFBFC;color:#1a1a2e;display:flex;height:100vh;overflow:hidden}

/* ---- SVG graph area ---- */
#graph-area{flex:1;position:relative;background:#FFFFFF;overflow:hidden;border-right:1px solid #E5E7EB}
#graph-area svg{width:100%;height:100%;display:block}

/* Subtle dot grid */
#graph-area::before{content:'';position:absolute;inset:0;background-image:radial-gradient(circle,#E5E7EB 1px,transparent 1px);background-size:24px 24px;opacity:0.5;pointer-events:none;z-index:0}

/* ---- Right panel ---- */
#panel{width:480px;min-width:480px;background:#F8FAFC;display:flex;flex-direction:column;overflow:hidden}
#header{padding:20px 24px 16px;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:white;text-align:center}
#header h1{font-size:15px;font-weight:600;letter-spacing:3px;color:#94A3B8;margin-bottom:8px}
#step-display{font-size:36px;font-weight:700;color:#F59E0B;font-family:'JetBrains Mono',monospace;transition:all 0.3s ease}
#phase-label{font-size:13px;color:#94A3B8;margin-top:6px;font-weight:500;min-height:18px;transition:opacity 0.3s ease}

/* Navigation buttons */
#nav{padding:16px 24px;background:#1a1a2e;display:flex;justify-content:center;gap:8px;border-bottom:1px solid #E5E7EB}
.btn{background:#334155;color:white;border:none;padding:12px 28px;cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;border-radius:8px;transition:all 0.2s ease;user-select:none}
.btn:hover{background:#475569;transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,0.3)}
.btn:active{transform:translateY(0)}
.btn:disabled{opacity:0.25;cursor:default;transform:none;box-shadow:none}
.btn-primary{background:#F59E0B;color:#1a1a2e}
.btn-primary:hover{background:#FBBF24}
#kbd-hint{text-align:center;padding:8px;font-size:11px;color:#64748B;background:#F1F5F9;border-bottom:1px solid #E5E7EB}
kbd{background:white;padding:2px 8px;border-radius:4px;font-size:11px;border:1px solid #D1D5DB;font-family:'JetBrains Mono',monospace;box-shadow:0 1px 2px rgba(0,0,0,0.05)}

/* Current arrow info */
#info-box{margin:16px 20px;padding:20px;background:white;border-radius:12px;border:1px solid #E5E7EB;box-shadow:0 2px 8px rgba(0,0,0,0.04);min-height:120px;transition:all 0.4s ease}
#info-box.pulse{border-color:#F59E0B;box-shadow:0 0 0 3px rgba(245,158,11,0.15)}
#info-box .num{display:inline-block;background:#F59E0B;color:#1a1a2e;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:13px;padding:3px 10px;border-radius:6px;margin-bottom:8px}
#info-box .arrow-label{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:700;color:#1a1a2e;margin-bottom:10px;line-height:1.3}
#info-box .path-line{font-size:14px;color:#3B82F6;font-weight:600;margin-bottom:8px;padding:8px 12px;background:#EFF6FF;border-radius:8px;display:flex;align-items:center;gap:8px}
#info-box .path-line .arr{color:#F59E0B;font-size:18px}
#info-box .desc{font-size:13px;color:#64748B;line-height:1.5;margin-bottom:6px}
#info-box .ex{font-family:'JetBrains Mono',monospace;font-size:12px;color:#9CA3AF;background:#F9FAFB;padding:6px 10px;border-radius:6px;border:1px solid #F3F4F6}

/* Legend table */
#legend-wrap{flex:1;overflow-y:auto;padding:0 16px 16px}
#legend-wrap h3{font-size:12px;font-weight:600;color:#9CA3AF;letter-spacing:2px;padding:12px 8px 8px;position:sticky;top:0;background:#F8FAFC;z-index:1}
#legend-wrap table{width:100%;border-collapse:separate;border-spacing:0;font-size:11px}
#legend-wrap thead th{background:#F1F5F9;color:#64748B;padding:8px 6px;text-align:left;position:sticky;top:32px;z-index:1;font-weight:600;font-size:10px;letter-spacing:0.5px;text-transform:uppercase}
#legend-wrap thead th:first-child{border-radius:6px 0 0 6px}
#legend-wrap thead th:last-child{border-radius:0 6px 6px 0}
#legend-wrap td{padding:6px;color:#9CA3AF;border-bottom:1px solid #F3F4F6;transition:all 0.35s ease}
#legend-wrap tr.active{background:#FEF3C7!important;border-radius:6px}
#legend-wrap tr.active td{color:#92400E;font-weight:600}
#legend-wrap tr.past td{color:#6B7280}
#legend-wrap tr.future td{color:#D1D5DB}
.badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:9px;color:white;margin-left:4px;font-weight:600;letter-spacing:0.3px}
.badge-internal{background:#6B7280}
.badge-dangling_in{background:#EF4444}
.badge-dangling_out{background:#3B82F6}
.badge-test{background:#10B981}

/* ---- SVG styles ---- */
.node-box{cursor:default;transition:opacity 0.3s ease}
.node-title{font-family:'JetBrains Mono',monospace;font-weight:600;fill:white;pointer-events:none}
.node-sub{font-family:'Inter',sans-serif;font-weight:400;pointer-events:none}
.ext-box{rx:16;ry:16}
.edge-line{fill:none;stroke-width:2;transition:opacity 0.4s ease,stroke 0.3s ease}
.edge-label{font-family:'JetBrains Mono',monospace;font-weight:600;text-anchor:middle;pointer-events:none}
.edge-label-bg{fill:white;opacity:0.9;rx:4;ry:4}
.arrowhead{transition:fill 0.3s ease}
.level-label{font-family:'Inter',sans-serif;font-size:12px;fill:#9CA3AF;font-weight:600;letter-spacing:1px}

/* Blink animation for active edge */
@keyframes pulse-edge{
  0%,100%{opacity:1;stroke-width:3.5}
  50%{opacity:0.4;stroke-width:2}
}
.edge-active .edge-line{animation:pulse-edge 1.2s ease-in-out infinite;stroke:#F59E0B!important;stroke-width:3.5}
.edge-active .arrowhead{fill:#F59E0B!important;animation:pulse-edge 1.2s ease-in-out infinite}
.edge-active .edge-label{fill:#B45309!important;font-size:13px}
.edge-active .edge-label-bg{fill:#FEF3C7!important;stroke:#F59E0B;stroke-width:1;opacity:1}
</style>
</head><body>

<div id="graph-area">
  <svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg>
</div>

<div id="panel">
  <div id="header">
    <h1>HC-MARL INTER-FILE DATA FLOW</h1>
    <div id="step-display">Step 0 / TOTAL</div>
    <div id="phase-label">28 boxes, no arrows yet</div>
  </div>
  <div id="nav">
    <button class="btn" id="btn-start" onclick="goTo(0)">|&laquo;</button>
    <button class="btn" id="btn-prev" onclick="prev()">&lsaquo; Prev</button>
    <button class="btn btn-primary" id="btn-next" onclick="next()">Next &rsaquo;</button>
    <button class="btn" id="btn-end" onclick="goTo(edges.length)">&raquo;|</button>
  </div>
  <div id="kbd-hint"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> arrow keys &nbsp;&nbsp; <kbd>Home</kbd> <kbd>End</kbd> &nbsp;&nbsp; <kbd>Space</kbd> = next</div>
  <div id="info-box">
    <div class="arrow-label">Ready</div>
    <div class="desc">Press <kbd>&rarr;</kbd> or click <b>Next</b> to reveal the first data-flow arrow.</div>
  </div>
  <div id="legend-wrap">
    <h3>ARROW LEGEND</h3>
    <table><thead><tr><th>#</th><th>Arrow Label</th><th>What It Is</th><th>Example</th></tr></thead>
    <tbody id="legend-body"></tbody></table>
  </div>
</div>

<script>
const boxes = BOXES_JSON;
const edges = EDGES_JSON;
const extLabels = EXT_LABELS_JSON;

// ===== LAYOUT CONSTANTS =====
const NW = 170, NH = 54, EW = 130, EH = 38;
const PAD_X = 24, PAD_Y = 80;
const MARGIN_TOP = 60, MARGIN_LEFT = 180;

// Group colours (light-theme friendly, saturated)
const groupFill = {
  core:'#1E3A5F', env:'#166534', agents:'#7C2D12', scripts:'#92400E', infra:'#581C87'
};
const groupFillExt = '#6B7280';
const groupFillExtOut = '#374151';

// ===== BUILD NODE MAP =====
const levelOrder = ['ext_in','core','env','agents','scripts','infra','ext_out'];
const levelRows = {ext_in:[],core:[],env:[],agents:[],scripts:[],infra:[],ext_out:[]};
const nodeMap = {};

// Collect externals
const extIn = new Set(), extOut = new Set();
for (const e of edges) {
    if (e.from.startsWith('EXT_')) extIn.add(e.from);
    if (e.to.startsWith('EXT_')) extOut.add(e.to);
}
for (const ext of extIn) {
    levelRows.ext_in.push(ext);
    nodeMap[ext] = {title:extLabels[ext]||ext.replace('EXT_',''),sub:'',fill:groupFillExt,isExt:true};
}
for (const b of boxes) {
    levelRows[b.group].push(b.id);
    nodeMap[b.id] = {title:b.title,sub:b.sub,fill:groupFill[b.group]||'#1E3A5F',isExt:false};
}
for (const ext of extOut) {
    levelRows.ext_out.push(ext);
    nodeMap[ext] = {title:extLabels[ext]||ext.replace('EXT_',''),sub:'',fill:groupFillExtOut,isExt:true};
}

// Assign x,y positions
const lvlLabels = {ext_in:'EXTERNAL INPUTS',core:'CORE MODULES  (hcmarl/)',env:'ENVIRONMENTS  (hcmarl/envs/)',agents:'AGENTS  (hcmarl/agents/)',scripts:'SCRIPTS  (scripts/)',infra:'INFRASTRUCTURE',ext_out:'EXTERNAL OUTPUTS'};
let curY = MARGIN_TOP;
const levelYMap = {};

for (const lvl of levelOrder) {
    const row = levelRows[lvl];
    if (!row.length) continue;
    const isExt = lvl==='ext_in'||lvl==='ext_out';
    const nw = isExt?EW:NW, nh = isExt?EH:NH;
    const rowW = row.length*(nw+PAD_X)-PAD_X;
    const startX = MARGIN_LEFT + (2000 - rowW)/2; // center in a 2000px-wide logical space
    levelYMap[lvl] = curY + nh/2;
    for (let i=0;i<row.length;i++) {
        const id = row[i];
        nodeMap[id].x = startX + i*(nw+PAD_X);
        nodeMap[id].y = curY;
        nodeMap[id].w = nw;
        nodeMap[id].h = nh;
    }
    curY += nh + PAD_Y;
}
const totalW = MARGIN_LEFT + 2000 + 40;
const totalH = curY + 40;

// ===== SVG CONSTRUCTION =====
const svg = document.getElementById('svg');
svg.setAttribute('viewBox', '0 0 '+totalW+' '+totalH);
svg.setAttribute('preserveAspectRatio','xMidYMid meet');

// Defs: arrowhead markers
const defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
const markerColors = {internal:'#6B7280',dangling_in:'#EF4444',dangling_out:'#3B82F6',test:'#10B981',active:'#F59E0B'};
for (const [key,col] of Object.entries(markerColors)) {
    const m = document.createElementNS('http://www.w3.org/2000/svg','marker');
    m.setAttribute('id','arrow-'+key);m.setAttribute('viewBox','0 0 10 10');
    m.setAttribute('refX','10');m.setAttribute('refY','5');
    m.setAttribute('markerWidth','8');m.setAttribute('markerHeight','8');
    m.setAttribute('orient','auto-start-reverse');
    const p = document.createElementNS('http://www.w3.org/2000/svg','path');
    p.setAttribute('d','M 0 0 L 10 5 L 0 10 z');p.setAttribute('fill',col);
    p.classList.add('arrowhead');
    m.appendChild(p); defs.appendChild(m);
}
svg.appendChild(defs);

// Level labels
for (const lvl of levelOrder) {
    if (!levelYMap[lvl]) continue;
    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x','30');t.setAttribute('y',levelYMap[lvl]+4);
    t.classList.add('level-label');t.textContent=lvlLabels[lvl]||lvl;
    svg.appendChild(t);
}

// Draw nodes (always visible, FIXED, never move)
for (const id in nodeMap) {
    const n = nodeMap[id];
    const g = document.createElementNS('http://www.w3.org/2000/svg','g');
    g.classList.add('node-box');
    const r = document.createElementNS('http://www.w3.org/2000/svg','rect');
    r.setAttribute('x',n.x);r.setAttribute('y',n.y);r.setAttribute('width',n.w);r.setAttribute('height',n.h);
    r.setAttribute('rx',n.isExt?n.h/2:8);r.setAttribute('ry',n.isExt?n.h/2:8);
    r.setAttribute('fill',n.fill);
    r.setAttribute('stroke',n.isExt?'#9CA3AF':'#D1D5DB');r.setAttribute('stroke-width','1.5');
    // Drop shadow via filter
    r.setAttribute('filter','drop-shadow(0 2px 4px rgba(0,0,0,0.1))');
    g.appendChild(r);

    const titleEl = document.createElementNS('http://www.w3.org/2000/svg','text');
    titleEl.setAttribute('x',n.x+n.w/2);
    titleEl.setAttribute('y', n.isExt ? n.y+n.h/2+4 : n.y+n.h/2 - (n.sub?3:4));
    titleEl.setAttribute('text-anchor','middle');
    titleEl.classList.add('node-title');
    titleEl.setAttribute('font-size', n.isExt?'11':'13');
    titleEl.setAttribute('fill','white');
    titleEl.textContent = n.title;
    g.appendChild(titleEl);

    if (n.sub && !n.isExt) {
        const subEl = document.createElementNS('http://www.w3.org/2000/svg','text');
        subEl.setAttribute('x',n.x+n.w/2);subEl.setAttribute('y',n.y+n.h/2+14);
        subEl.setAttribute('text-anchor','middle');subEl.classList.add('node-sub');
        subEl.setAttribute('font-size','9');subEl.setAttribute('fill','#CBD5E1');
        // Truncate long subtitles
        subEl.textContent = n.sub.length>28 ? n.sub.slice(0,26)+'..' : n.sub;
        g.appendChild(subEl);
    }
    svg.appendChild(g);
}

// ===== EDGE ELEMENTS (all created hidden) =====
const edgeTypeColor = {internal:'#6B7280',dangling_in:'#EF4444',dangling_out:'#3B82F6',test:'#10B981'};
const edgeTypeDash = {dangling_in:'6,4',dangling_out:'4,4'};
const edgeGroups = []; // one <g> per edge

function clipEdge(n, otherX, otherY) {
    const cx=n.x+n.w/2, cy=n.y+n.h/2;
    const dx=otherX-cx, dy=otherY-cy;
    if(dx===0&&dy===0) return {x:cx,y:cy};
    const w2=n.w/2, h2=n.h/2;
    const sx=w2/Math.abs(dx||0.001), sy=h2/Math.abs(dy||0.001);
    const s=Math.min(sx,sy);
    return {x:cx+dx*s, y:cy+dy*s};
}

for (let i=0;i<edges.length;i++){
    const e = edges[i];
    const fromN=nodeMap[e.from], toN=nodeMap[e.to];
    if(!fromN||!toN) { edgeGroups.push(null); continue; }

    const fc=fromN.x+fromN.w/2, fcy=fromN.y+fromN.h/2;
    const tc=toN.x+toN.w/2, tcy=toN.y+toN.h/2;
    const p1=clipEdge(fromN,tc,tcy), p2=clipEdge(toN,fc,fcy);

    const g = document.createElementNS('http://www.w3.org/2000/svg','g');
    g.style.display='none';
    g.dataset.idx=i;

    const col = edgeTypeColor[e.type]||'#6B7280';
    const line = document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',p1.x);line.setAttribute('y1',p1.y);
    line.setAttribute('x2',p2.x);line.setAttribute('y2',p2.y);
    line.setAttribute('stroke',col);
    line.classList.add('edge-line');
    line.setAttribute('marker-end','url(#arrow-'+(e.type in markerColors?e.type:'internal')+')');
    if(edgeTypeDash[e.type]) line.setAttribute('stroke-dasharray',edgeTypeDash[e.type]);
    g.appendChild(line);

    // Edge number label with background
    const mx=(p1.x+p2.x)/2, my=(p1.y+p2.y)/2;
    const labelText = '#'+(i+1);
    const bgR = document.createElementNS('http://www.w3.org/2000/svg','rect');
    bgR.setAttribute('x',mx-14);bgR.setAttribute('y',my-10);
    bgR.setAttribute('width',28);bgR.setAttribute('height',18);
    bgR.classList.add('edge-label-bg');
    g.appendChild(bgR);

    const lbl = document.createElementNS('http://www.w3.org/2000/svg','text');
    lbl.setAttribute('x',mx);lbl.setAttribute('y',my+4);
    lbl.classList.add('edge-label');
    lbl.setAttribute('font-size','11');lbl.setAttribute('fill','#6B7280');
    lbl.textContent=labelText;
    g.appendChild(lbl);

    svg.appendChild(g);
    edgeGroups.push(g);
}

// ===== LEGEND TABLE =====
const legendBody = document.getElementById('legend-body');
for (let i=0;i<edges.length;i++){
    const e=edges[i];
    const tr=document.createElement('tr');
    tr.id='leg-'+i;tr.className='future';
    tr.innerHTML='<td style="font-family:JetBrains Mono,monospace;font-weight:600">'+(i+1)+'</td>'
        +'<td style="font-family:JetBrains Mono,monospace;font-size:11px">'+e.label
        +'<span class="badge badge-'+e.type+'">'+e.type.replace('_',' ')+'</span></td>'
        +'<td>'+e.desc+'</td>'
        +'<td style="font-family:JetBrains Mono,monospace;font-size:10px">'+e.ex+'</td>';
    legendBody.appendChild(tr);
}

// ===== STEP STATE =====
let currentStep = 0;
const total = edges.length;
document.getElementById('step-display').textContent = 'Step 0 / '+total;

function getPhaseLabel(s){
    if(s===0) return '28 boxes, no arrows yet';
    if(s<=2) return 'Phase A Layer 1: Foundation utilities';
    if(s<=8) return 'Phase A Layer 2: Core math modules';
    if(s<=12) return 'Phase A Layer 3: Pipeline integration';
    if(s<=17) return 'Phase A Layer 4: MMICRL + Real data';
    if(s<=27) return 'Phase A Layer 5-6: Environments';
    if(s<=43) return 'Phase A Layer 6: Agents + Networks';
    if(s<=62) return 'Phase B Layer 7: Training scripts';
    if(s<=66) return 'Phase B Layer 8: Evaluation';
    if(s<=77) return 'Phase B Layer 9-10: Launchers + Notebooks';
    if(s<=85) return 'Dangling outputs (external)';
    return 'Tests (223 pytest tests)';
}

function getNodeLabel(id){
    if(id.startsWith('EXT_')) return extLabels[id]||id.replace('EXT_','');
    const b=boxes.find(b=>b.id===id);
    return b?b.title:id;
}

function updateDisplay(){
    document.getElementById('step-display').textContent='Step '+currentStep+' / '+total;
    document.getElementById('phase-label').textContent=getPhaseLabel(currentStep);
    document.getElementById('btn-prev').disabled=(currentStep===0);
    document.getElementById('btn-next').disabled=(currentStep===total);

    // Show/hide edges, set active class
    for(let i=0;i<total;i++){
        const g=edgeGroups[i];
        if(!g) continue;
        if(i<currentStep){
            g.style.display='';
            if(i===currentStep-1){g.classList.add('edge-active');}
            else{g.classList.remove('edge-active');}
        } else {
            g.style.display='none';
            g.classList.remove('edge-active');
        }
    }

    // Update legend
    for(let i=0;i<total;i++){
        const tr=document.getElementById('leg-'+i);
        if(i<currentStep-1) tr.className='past';
        else if(i===currentStep-1){tr.className='active';tr.scrollIntoView({block:'nearest',behavior:'smooth'});}
        else tr.className='future';
    }

    // Update info box
    const box=document.getElementById('info-box');
    if(currentStep===0){
        box.classList.remove('pulse');
        box.innerHTML='<div class="arrow-label">Ready</div><div class="desc">Press <kbd>&rarr;</kbd> or click <b>Next</b> to reveal the first data-flow arrow.</div>';
    } else {
        box.classList.add('pulse');
        const e=edges[currentStep-1];
        box.innerHTML='<span class="num">#'+currentStep+' of '+total+'</span>'
            +'<div class="arrow-label">'+e.label+'</div>'
            +'<div class="path-line"><span>'+getNodeLabel(e.from)+'</span><span class="arr">&#10132;</span><span>'+getNodeLabel(e.to)+'</span></div>'
            +'<div class="desc">'+e.desc+'</div>'
            +'<div class="ex">'+e.ex+'</div>';
    }
}

function next(){if(currentStep<total){currentStep++;updateDisplay();}}
function prev(){if(currentStep>0){currentStep--;updateDisplay();}}
function goTo(n){currentStep=Math.max(0,Math.min(total,n));updateDisplay();}

document.addEventListener('keydown',function(ev){
    if(ev.key==='ArrowRight'||ev.key===' '){ev.preventDefault();next();}
    else if(ev.key==='ArrowLeft'){ev.preventDefault();prev();}
    else if(ev.key==='Home'){ev.preventDefault();goTo(0);}
    else if(ev.key==='End'){ev.preventDefault();goTo(total);}
});

updateDisplay();
</script>
</body></html>"""

    html = html.replace('BOXES_JSON', json.dumps(BOXES))
    html = html.replace('EDGES_JSON', json.dumps(EDGES))
    html = html.replace('EXT_LABELS_JSON', json.dumps(EXT_LABELS))

    out_path = os.path.join(OUT, "master_slideshow.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"OK: {out_path} ({len(EDGES)} steps)")


if __name__ == "__main__":
    build()
