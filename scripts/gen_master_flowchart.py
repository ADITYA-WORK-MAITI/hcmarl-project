"""
Master inter-file data flow diagram for HC-MARL project.

28 boxes (25 individual files + 3 grouped) connected by arrows whose labels
are the EXACT data/variable names from the dangling arrows in the 68 per-file
code flowcharts.

Excluded (7 files with zero unique data flow):
  4 __init__.py (pure re-exports), tests/__init__.py (empty),
  gen_directory_structure.py (static PNG util), setup.py (package metadata)
"""

import os
import graphviz

os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + ";" + os.environ.get("PATH", "")

OUT = "diagrams/code_flowcharts"
os.makedirs(OUT, exist_ok=True)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Colour palette ──
CORE   = "#1E3A5F"   # core hcmarl/ modules  (steel blue)
SUBPKG = "#2D4A22"   # sub-package modules   (forest green)
SCRIPT = "#6B3A1E"   # scripts/              (brown)
GROUP  = "#4A2A5C"   # grouped boxes         (purple)
DANG_IN  = "#DC2626" # dangling-in arrow     (red)
DANG_OUT = "#3B82F6" # dangling-out arrow    (blue)
INTERN   = "#475569" # internal edge         (slate)
TEST_C   = "#10B981" # test edge colour      (emerald)

# Counter for unique phantom node IDs
_phantom_counter = 0


def make_box(g, nid, title, subtitle, color, extra_rows=None):
    """Create an HTML-table node."""
    sub = f'<FONT POINT-SIZE="9" COLOR="#CBD5E1">  {esc(subtitle)}  </FONT>'
    hdr = (
        f'<TR><TD COLSPAN="1" BGCOLOR="{color}" ALIGN="LEFT">'
        f'<FONT COLOR="white" FACE="Consolas" POINT-SIZE="11"><B>  {esc(title)}  </B></FONT>'
        f'</TD></TR>'
        f'<TR><TD ALIGN="LEFT">{sub}</TD></TR>'
    )
    extras = ""
    if extra_rows:
        for row in extra_rows:
            extras += (
                f'<TR><TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="8" '
                f'COLOR="#6B7280">  {esc(row)}  </FONT></TD></TR>'
            )
    html = (
        f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
        f'CELLPADDING="4" COLOR="#CBD5E1">{hdr}{extras}</TABLE>'
    )
    g.node(nid, "<" + html + ">")


def dangling_in(g, target, label, source_desc):
    """Dashed red arrow INTO a root box from external source.
    Uses a tiny point node to minimise layout distortion."""
    global _phantom_counter
    _phantom_counter += 1
    pid = f"_din_{_phantom_counter}"
    g.node(pid, "", shape="point", width="0.08", color=DANG_IN)
    g.edge(pid, target,
           label=f"  {label}\n[from {source_desc}]  ",
           style="dashed", color=DANG_IN, fontname="Consolas",
           fontsize="6", fontcolor=DANG_IN, constraint="false")


def dangling_out(g, source, label, target_desc):
    """Dotted blue arrow OUT OF a leaf box to external target.
    Uses a tiny point node to minimise layout distortion."""
    global _phantom_counter
    _phantom_counter += 1
    pid = f"_dout_{_phantom_counter}"
    g.node(pid, "", shape="point", width="0.08", color=DANG_OUT)
    g.edge(source, pid,
           label=f"  {label}\n[to {target_desc}]  ",
           style="dotted", color=DANG_OUT, fontname="Consolas",
           fontsize="6", fontcolor=DANG_OUT, constraint="false")


def edge(g, src, dst, label, **kw):
    """Solid internal edge with data label."""
    style = kw.pop("style", "solid")
    color = kw.pop("color", INTERN)
    constraint = kw.pop("constraint", "true")
    g.edge(src, dst, label=f"  {label}  " if label else "",
           style=style, color=color, fontname="Consolas", fontsize="7",
           fontcolor="#334155", constraint=constraint, **kw)


def build():
    g = graphviz.Digraph("master_data_flow", format="png", engine="dot")
    g.attr(
        dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
        nodesep="0.25", ranksep="0.7", margin="0.2",
        newrank="true",
        concentrate="true",
        size="28,50",  # constrain width, allow height to grow
        ratio="compress",
    )
    g.attr("node", fontname="Consolas", fontsize="10", shape="none")
    g.attr("edge", fontname="Consolas", fontsize="7", color=INTERN)

    # ==================================================================
    # 28 BOXES
    # ==================================================================

    # ── Core modules (9) ──
    make_box(g, "utils", "utils.py", "Logging, YAML, seeding, safe_log, clip", CORE)
    make_box(g, "three_cc_r", "three_cc_r.py", "3CC-r fatigue ODE (Eqs 1-8)", CORE)
    make_box(g, "ecbf", "ecbf_filter.py", "ECBF safety filter CBF-QP (Eqs 12-23)", CORE)
    make_box(g, "nswf", "nswf_allocator.py", "Nash Social Welfare allocator (Eqs 31-33)", CORE)
    make_box(g, "pipeline", "pipeline.py", "End-to-end HC-MARL pipeline", CORE)
    make_box(g, "mmicrl", "mmicrl.py", "MM-ICRL with CFDE flows (Eqs 9-11)", CORE)
    make_box(g, "real_data", "real_data_calibration.py", "Path G: WSD4FEDSRM calibration", CORE)
    make_box(g, "warehouse", "warehouse_env.py", "Single/Multi-agent warehouse env", CORE)
    make_box(g, "logger", "logger.py", "HCMARLLogger + W&B", CORE)

    # ── Sub-package modules (11) ──
    make_box(g, "pettingzoo", "pettingzoo_wrapper.py", "PettingZoo AEC environment", SUBPKG)
    make_box(g, "reward_fn", "reward_functions.py", "nswf_reward() + safety_cost()", SUBPKG)
    make_box(g, "task_prof", "task_profiles.py", "TaskProfileManager", SUBPKG)
    make_box(g, "networks", "networks.py", "Actor, Critic, CostCritic nets", SUBPKG)
    make_box(g, "mappo", "mappo.py", "MAPPO + RolloutBuffer", SUBPKG)
    make_box(g, "ippo", "ippo.py", "IPPO (independent PPO)", SUBPKG)
    make_box(g, "mappo_lag", "mappo_lag.py", "MAPPO-Lagrangian", SUBPKG)
    make_box(g, "hcmarl_agent", "hcmarl_agent.py", "HCMARLAgent (pipeline-aware)", SUBPKG)
    make_box(g, "legacy", "_legacy.py", "10 baseline classes", SUBPKG)
    make_box(g, "omnisafe_w", "omnisafe_wrapper.py", "OmniSafe wrapper", SUBPKG)
    make_box(g, "safepo_w", "safepo_wrapper.py", "SafePO wrapper", SUBPKG)

    # ── Scripts (5) ──
    make_box(g, "train", "scripts/train.py", "Main training: all 7 methods", SCRIPT)
    make_box(g, "evaluate", "scripts/evaluate.py", "Evaluation: 9 metrics", SCRIPT)
    make_box(g, "run_abl", "scripts/run_ablations.py", "Ablation launcher", SCRIPT)
    make_box(g, "run_base", "scripts/run_baselines.py", "Baseline launcher", SCRIPT)
    make_box(g, "run_scale", "scripts/run_scaling.py", "Scaling launcher", SCRIPT)

    # ── Grouped boxes (3) ──
    make_box(g, "configs", "config/*.yaml (13 files)",
             "Training, method, ablation configs + experiment_matrix.yaml", GROUP,
             extra_rows=[
                 "hcmarl_full, default, dry_run_50k",
                 "mappo/ippo/mappo_lag_config",
                 "ablation_no_{ecbf,nswf,mmicrl,reperfusion,divergent}",
                 "experiment_matrix, task_profiles",
             ])
    make_box(g, "notebooks", "notebooks/*.ipynb (4 files)",
             "Colab training notebooks", GROUP,
             extra_rows=[
                 "train_hcmarl, train_baselines",
                 "train_ablations, train_scaling",
             ])
    make_box(g, "tests", "tests/*.py (13 files)",
             "238 pytest tests across all modules", GROUP,
             extra_rows=[
                 "test_three_cc_r, test_ecbf, test_nswf, test_pipeline",
                 "test_phase2, test_phase3, test_real_data_calibration",
                 "test_all_methods, test_pettingzoo, test_warehouse_env",
                 "test_hcmarl_agent, test_env_integration",
             ])

    # ==================================================================
    # RANK CONSTRAINTS — minimal, let dot optimise freely
    # ==================================================================
    with g.subgraph() as s:
        s.attr(rank="min")
        for n in ["utils", "three_cc_r", "configs"]:
            s.node(n)

    with g.subgraph() as s:
        s.attr(rank="same")
        for n in ["train", "evaluate"]:
            s.node(n)

    # tests at bottom (no explicit rank=max, let edges pull it down)

    # ==================================================================
    # DANGLING-IN ARROWS (external libs/data -> root boxes)
    # All constraint=false to avoid layout distortion
    # ==================================================================
    dangling_in(g, "three_cc_r", "np.array, solve_ivp, dataclass", "numpy, scipy, dataclasses")
    dangling_in(g, "ecbf", "cp.Variable, cp.Problem", "cvxpy")
    dangling_in(g, "mmicrl", "nn.Module, nn.Linear", "torch.nn")
    dangling_in(g, "warehouse", "gym.Env, spaces", "gymnasium")
    dangling_in(g, "utils", "logging, yaml.safe_load", "logging, pyyaml")
    dangling_in(g, "logger", "wandb", "wandb")
    dangling_in(g, "real_data", "WSD4FEDSRM/ CSV files", "data/WSD4FEDSRM/")
    dangling_in(g, "omnisafe_w", "omnisafe library", "pip install omnisafe")
    dangling_in(g, "safepo_w", "safepo library", "pip install safepo")

    # ==================================================================
    # DANGLING-OUT ARROWS (leaf boxes -> external outputs)
    # ==================================================================
    dangling_out(g, "train", "checkpoints .pt", "checkpoints/")
    dangling_out(g, "train", "summary.json, config.yaml", "logs/<method>/")
    dangling_out(g, "evaluate", "results JSON", "results/<method>_eval.json")
    dangling_out(g, "mappo", "checkpoint .pt file", "checkpoints/")
    dangling_out(g, "logger", "training_log.csv", "logs/")
    dangling_out(g, "logger", "W&B dashboard", "wandb.ai")
    dangling_out(g, "notebooks", "checkpoints + logs", "Google Drive")
    dangling_out(g, "notebooks", "eval JSONs", "results/")

    # ==================================================================
    # INTERNAL EDGES — derived from 68 per-file dangling arrows
    # Each label = EXACT data/variable name flowing between files.
    # ==================================================================

    # ── utils.py -> dependents ──
    edge(g, "utils", "three_cc_r", "clip_and_normalise")
    edge(g, "utils", "nswf", "safe_log")
    edge(g, "utils", "pipeline", "logger, config dict")
    edge(g, "utils", "train", "seed_everything")

    # ── three_cc_r.py -> dependents ──
    edge(g, "three_cc_r", "ecbf", "MuscleParams, ThreeCCrState")
    edge(g, "three_cc_r", "pipeline", "ThreeCCr, get_muscle, SHOULDER")
    edge(g, "three_cc_r", "real_data", "MuscleParams, ThreeCCr, SHOULDER")
    edge(g, "three_cc_r", "pettingzoo", "get_muscle()")

    # ── ecbf <-> pipeline (bidirectional) ──
    edge(g, "ecbf", "pipeline", "C_filtered, ECBFParams")
    edge(g, "pipeline", "ecbf", "C_nominal", constraint="false")
    edge(g, "ecbf", "warehouse", "C_safe")

    # ── nswf <-> pipeline (bidirectional) ──
    edge(g, "nswf", "pipeline", "AllocationResult")
    edge(g, "pipeline", "nswf", "utility_matrix, fatigue_levels", constraint="false")
    edge(g, "nswf", "warehouse", "D_i values")

    # ── real_data -> dependents ──
    edge(g, "real_data", "mmicrl", "demo trajectories")
    edge(g, "real_data", "train", "calibration results")

    # ── mmicrl -> dependents ──
    edge(g, "mmicrl", "ecbf", "theta_max", constraint="false")
    edge(g, "mmicrl", "train", "MI, type_proportions, theta_per_type")

    # ── warehouse <-> reward_fn (bidirectional) ──
    edge(g, "warehouse", "reward_fn", "productivity, fatigue dict", constraint="false")
    edge(g, "reward_fn", "warehouse", "reward float, cost float")
    edge(g, "reward_fn", "pettingzoo", "nswf_reward, safety_cost")
    edge(g, "reward_fn", "mappo_lag", "cost float")

    # ── warehouse -> agents/scripts ──
    edge(g, "warehouse", "mmicrl", "env, policy", constraint="false")
    edge(g, "warehouse", "networks", "obs tensor (obs_dim,)")
    edge(g, "warehouse", "train", "env instances")

    # ── task_profiles -> pettingzoo ──
    edge(g, "task_prof", "pettingzoo", "demand_vector, task_intensity")

    # ── config -> dependents ──
    edge(g, "configs", "task_prof", "config YAML")
    edge(g, "configs", "train", "env params, PPO hyperparams, ECBF gains")
    edge(g, "configs", "pipeline", "ECBF params, NSWF params")
    edge(g, "configs", "run_abl", "ablation configs")
    edge(g, "configs", "run_base", "per-method configs")
    edge(g, "configs", "run_scale", "scaling configs")
    edge(g, "configs", "notebooks", "YAML configs")

    # ── pettingzoo <-> agents ──
    edge(g, "pettingzoo", "networks", "global_state tensor")
    edge(g, "pettingzoo", "mappo", "observations dict, global_state")
    edge(g, "mappo", "pettingzoo", "actions dict", constraint="false")
    edge(g, "ippo", "pettingzoo", "actions dict", constraint="false")
    edge(g, "mappo_lag", "pettingzoo", "actions, value, cost_value", constraint="false")
    edge(g, "hcmarl_agent", "pettingzoo", "actions dict", constraint="false")
    edge(g, "legacy", "pettingzoo", "actions dict", constraint="false")
    edge(g, "pettingzoo", "legacy", "observations")
    edge(g, "pettingzoo", "train", "WarehousePettingZoo")

    # ── networks -> agents ──
    edge(g, "networks", "mappo", "ActorNetwork, CriticNetwork")
    edge(g, "networks", "ippo", "ActorNetwork, CriticNetwork")
    edge(g, "networks", "mappo_lag", "ActorNetwork, CriticNetwork, CostCriticNetwork")

    # ── agent internals ──
    edge(g, "mappo", "ippo", "RolloutBuffer")
    edge(g, "mappo", "hcmarl_agent", "MAPPO")
    edge(g, "mappo_lag", "safepo_w", "MAPPOLagrangian")

    # ── agents/baselines -> train.py ──
    edge(g, "hcmarl_agent", "train", "HCMARLAgent")
    edge(g, "mappo", "train", "MAPPO, loss metrics")
    edge(g, "ippo", "train", "IPPO")
    edge(g, "mappo_lag", "train", "MAPPOLagrangian, loss + lambda")
    edge(g, "omnisafe_w", "train", "OmniSafeWrapper, actions")
    edge(g, "safepo_w", "train", "SafePOWrapper, actions")
    edge(g, "mmicrl", "train", "DemonstrationCollector, MMICRL")

    # ── logger <-> train ──
    edge(g, "logger", "train", "HCMARLLogger")
    edge(g, "train", "logger", "episode_data", constraint="false")

    # ── pipeline <-> train/evaluate ──
    edge(g, "pipeline", "train", "step_result dict")
    edge(g, "train", "pipeline", "utility_matrix", constraint="false")
    edge(g, "pipeline", "evaluate", "history list")

    # ── evaluate imports ──
    edge(g, "train", "evaluate", "create_agent()")
    edge(g, "pettingzoo", "evaluate", "WarehousePettingZoo")
    edge(g, "logger", "evaluate", "METRIC_NAMES")

    # ── launchers -> train ──
    edge(g, "run_abl", "train", "ablation_name, config_path", constraint="false")
    edge(g, "run_base", "train", "method, config_path", constraint="false")
    edge(g, "run_scale", "train", "n_workers, config_path", constraint="false")

    # ── notebooks imports ──
    edge(g, "train", "notebooks", "scripts/train.py")
    edge(g, "run_base", "notebooks", "scripts/run_baselines.py")
    edge(g, "run_abl", "notebooks", "scripts/run_ablations.py")
    edge(g, "run_scale", "notebooks", "scripts/run_scaling.py")

    # ── source modules -> tests (green, constraint=false to keep tests at bottom) ──
    edge(g, "three_cc_r", "tests",
         "SHOULDER, ELBOW, GRIP,\nThreeCCr, ThreeCCrState, MuscleParams",
         color=TEST_C, constraint="false")
    edge(g, "ecbf", "tests",
         "ECBFParams, ECBFFilter",
         color=TEST_C, constraint="false")
    edge(g, "nswf", "tests",
         "NSWFAllocator, NSWFParams, AllocationResult",
         color=TEST_C, constraint="false")
    edge(g, "pipeline", "tests",
         "HCMARLPipeline, TaskProfile, WorkerState",
         color=TEST_C, constraint="false")
    edge(g, "mmicrl", "tests",
         "CFDE, _MADE, MMICRL,\nDemonstrationCollector, OnlineAdapter",
         color=TEST_C, constraint="false")
    edge(g, "real_data", "tests",
         "predict_endurance_time,\nPOPULATION_FR, POPULATION_CV_F",
         color=TEST_C, constraint="false")
    edge(g, "warehouse", "tests",
         "SingleWorkerWarehouseEnv,\nWarehouseMultiAgentEnv",
         color=TEST_C, constraint="false")
    edge(g, "pettingzoo", "tests",
         "WarehousePettingZoo",
         color=TEST_C, constraint="false")
    edge(g, "hcmarl_agent", "tests",
         "HCMARLAgent",
         color=TEST_C, constraint="false")
    edge(g, "logger", "tests",
         "HCMARLLogger",
         color=TEST_C, constraint="false")

    # One constrained edge to pull tests below notebooks
    edge(g, "notebooks", "tests", "", style="invis")

    # ==================================================================
    # RENDER main graph (no legend node — legend rendered separately)
    # ==================================================================
    out_path = os.path.join(OUT, "master_data_flow")
    g.render(out_path + "_graph", cleanup=True)
    print(f"OK: {out_path}_graph.png")

    # ==================================================================
    # LEGEND as separate graph
    # ==================================================================
    legend_entries = [
        # -- Core data flows --
        ("clip_and_normalise", "Clipping + normalisation utility fn", "clip_and_normalise(x, lo, hi)"),
        ("safe_log", "Numerically stable log utility", "safe_log(0.0) = -23.03"),
        ("logger, config dict", "Logger instance + parsed YAML dict", "{'env': {'n_workers': 3}}"),
        ("seed_everything", "Global RNG seeding function", "seed_everything(42)"),
        ("MuscleParams, ThreeCCrState", "Muscle dataclass + ODE state vector", "MuscleParams(F=180, R=45, r=13.5)"),
        ("ThreeCCr, get_muscle, SHOULDER", "ODE solver + factory + preset", "ThreeCCr(SHOULDER).simulate(60)"),
        ("get_muscle()", "Factory returning MuscleParams", "get_muscle('shoulder')"),
        ("C_filtered, ECBFParams", "Safety-filtered cmd + filter params", "C_filtered = array([0.8, 0.6])"),
        ("C_nominal", "Unfiltered RL/baseline command", "array([1.0, 0.9, 0.7])"),
        ("C_safe", "Per-step safe command (scalar)", "0.75"),
        ("theta_max", "Per-muscle fatigue threshold from MMICRL", "array([0.85, 0.72])"),
        ("AllocationResult", "Task assignment + surplus values", "AllocationResult(assignments=[0,1,2])"),
        ("utility_matrix, fatigue_levels", "Agent-task utilities + fatigue state", "U:(3,5), f:(3,) arrays"),
        ("D_i values", "Per-agent Nash disagreement values", "array([0.1, 0.2, 0.15])"),
        ("step_result dict", "Pipeline step output: states, rewards, costs", "{'fatigue':[...], 'cost':0.02}"),
        ("history list", "Full episode trajectory for evaluation", "[step_result_0, ...]"),
        ("demo trajectories", "Calibrated demos from WSD data", "[(F_t, R_t, r_t), ...]"),
        ("calibration results", "Path G fitted params + validation", "{'F_opt':185.2, 'r2':0.91}"),
        ("MI, type_proportions, theta_per_type", "Mutual info + worker-type mix + thresholds", "MI=2.3, [0.4,0.6]"),
        ("DemonstrationCollector, MMICRL", "Demo collector + MMICRL model", "MMICRL(obs_dim=18, n_flows=4)"),
        ("productivity, fatigue dict", "Task load sum + per-muscle MF", "prod=5.2, {'shoulder':0.6}"),
        ("reward float, cost float", "NSWF reward + safety cost scalars", "r=3.14, c=0.02"),
        ("nswf_reward, safety_cost", "Reward/cost function callables", "nswf_reward(prod, fatigue)"),
        ("cost float", "Safety constraint cost", "0.05"),
        ("env instances", "Constructed env objects for training", "SingleWorkerWarehouseEnv(n_tasks=5)"),
        ("env, policy", "Env + policy for demo collection", "(gym.Env, nn.Module)"),
        ("obs tensor (obs_dim,)", "Observation vector from env", "array shape (18,)"),
        ("demand_vector, task_intensity", "Per-task muscle demands + intensity", "array([0.3,0.6,0.1]), 0.8"),
        ("config YAML", "Parsed YAML configuration dict", "task_profiles.yaml"),
        ("global_state tensor", "Concatenated global obs for critic", "tensor shape (N*obs_dim,)"),
        ("observations dict, global_state", "Per-agent obs + shared state", "{'agent_0': tensor, ...}"),
        ("actions dict", "Per-agent discrete/continuous actions", "{'agent_0':2, 'agent_1':0}"),
        ("actions, value, cost_value", "Actions + V(s) + Vc(s) from Lagrangian", "acts, 3.14, 0.02"),
        ("observations", "Raw observations for baseline agents", "{'agent_0': array([...])}"),
        ("WarehousePettingZoo", "PettingZoo AEC environment class", "WarehousePettingZoo(n_workers=3)"),
        ("ActorNetwork, CriticNetwork", "Actor + Critic neural nets", "ActorNetwork(18, 5)"),
        ("ActorNetwork, CriticNetwork, CostCriticNetwork", "All 3 nets for Lagrangian", "CostCriticNetwork(54)"),
        ("RolloutBuffer", "Shared rollout buffer from MAPPO", "RolloutBuffer(capacity=2048)"),
        ("MAPPO", "MAPPO agent class", "MAPPO(obs_dim=18, n_actions=5)"),
        ("MAPPOLagrangian", "Lagrangian agent class", "MAPPOLagrangian(...)"),
        ("HCMARLAgent", "Pipeline-aware agent class", "HCMARLAgent(pipeline, mappo)"),
        ("MAPPO, loss metrics", "MAPPO class + training loss values", "policy_loss=0.03, value_loss=0.1"),
        ("IPPO", "Independent PPO agent class", "IPPO(obs_dim=18, n_actions=5)"),
        ("MAPPOLagrangian, loss + lambda", "Lagrangian agent + dual variable", "lambda=0.5, cost_loss=0.01"),
        ("OmniSafeWrapper, actions", "OmniSafe wrapper + actions", "OmniSafeWrapper('PPOLag')"),
        ("SafePOWrapper, actions", "SafePO wrapper + actions", "SafePOWrapper(mappo_lag)"),
        ("HCMARLLogger", "Logger class for metrics tracking", "HCMARLLogger(log_dir)"),
        ("METRIC_NAMES", "List of 9 eval metric names", "['mean_reward', 'mean_cost', ...]"),
        ("episode_data", "Per-episode metrics from training loop", "{'reward':4.2, 'cost':0.01}"),
        ("env params, PPO hyperparams, ECBF gains", "Full training config sections", "lr=3e-4, gamma_ecbf=0.1"),
        ("ECBF params, NSWF params", "Default safety + allocation params", "gamma_1=1.0, alpha_nswf=0.5"),
        ("ablation configs", "5 ablation YAML files", "ablation_no_ecbf.yaml"),
        ("per-method configs", "3 per-method YAML files (mappo/ippo/mappo_lag)", "mappo_config.yaml"),
        ("experiment_matrix", "single source of truth for grid launchers", "experiment_matrix.yaml"),
        ("YAML configs", "Training configs consumed by notebooks", "hcmarl_full_config.yaml"),
        ("create_agent()", "Agent factory function from train.py", "create_agent('hcmarl', config)"),
        ("ablation_name, config_path", "Ablation ID + YAML path", "'no_ecbf', 'config/ablation_...'"),
        ("method, config_path", "Method name + YAML path", "'mappo', 'config/mappo_config.yaml'"),
        ("scripts/train.py", "Training script invoked by notebook", "!python scripts/train.py --config ..."),
        ("scripts/run_baselines.py", "Baseline launcher invoked", "!python scripts/run_baselines.py"),
        ("scripts/run_ablations.py", "Ablation launcher invoked", "!python scripts/run_ablations.py"),
        # -- External I/O (dangling arrows) --
        ("np.array, solve_ivp, dataclass", "NumPy, SciPy ODE solver, dataclass", "np.array([1,3,5])"),
        ("cp.Variable, cp.Problem", "CVXPY optimisation primitives for QP", "cp.Variable(3)"),
        ("nn.Module, nn.Linear", "PyTorch neural network base classes", "nn.Linear(18, 64)"),
        ("gym.Env, spaces", "Gymnasium base env + space defs", "spaces.Discrete(5)"),
        ("logging, yaml.safe_load", "Python logging + YAML parser", "yaml.safe_load(open('x.yaml'))"),
        ("wandb", "Weights & Biases experiment tracking", "wandb.init(project='hcmarl')"),
        ("WSD4FEDSRM/ CSV files", "Real shoulder-fatigue dataset (67 subj)", "WSD_P01.csv ... WSD_P67.csv"),
        ("omnisafe library", "OmniSafe safe RL library (optional)", "omnisafe.Agent('PPOLag')"),
        ("safepo library", "SafePO safe MARL library (optional)", "safepo.MAPPO_Lagrangian()"),
        ("checkpoints .pt", "Saved model weights", "checkpoint_final.pt (27 MB)"),
        ("summary.json, config.yaml", "Training summary + used config", "{'best_reward': 8.3, ...}"),
        ("results JSON", "Evaluation metrics per method", "hcmarl_eval.json"),
        ("checkpoint .pt file", "MAPPO saved weights (per-agent)", "mappo_seed0.pt"),
        ("training_log.csv", "Per-episode metric log file", "episode,reward,cost,..."),
        ("W&B dashboard", "Live W&B metrics dashboard", "wandb.ai/run/hcmarl-001"),
        ("checkpoints + logs", "Colab outputs saved to Google Drive", "Drive/hcmarl/checkpoints/"),
        ("eval JSONs", "Evaluation result files from notebooks", "results/hcmarl_eval.json"),
        # -- Test arrows (green) --
        ("SHOULDER, ELBOW, GRIP, ...", "Core 3CC-r types for tests", "SHOULDER.F = 180.0"),
        ("ECBFParams, ECBFFilter", "Safety filter classes for tests", "ECBFFilter(params, state)"),
        ("NSWFAllocator, NSWFParams, AllocationResult", "Allocator classes for tests", "NSWFAllocator(NSWFParams())"),
        ("HCMARLPipeline, TaskProfile, WorkerState", "Pipeline + data classes for tests", "HCMARLPipeline.from_config()"),
        ("CFDE, _MADE, MMICRL, ...", "Flow model + collector for tests", "CFDE(dim=6, n_flows=4)"),
        ("predict_endurance_time, ...", "Calibration fn + constants for tests", "POPULATION_FR = (180.0, 45.0)"),
        ("SingleWorkerWarehouseEnv, ...", "Env classes for tests", "SingleWorkerWarehouseEnv()"),
    ]

    lg = graphviz.Digraph("legend", format="png")
    lg.attr(dpi="200", bgcolor="white", margin="0.2")
    lg.attr("node", fontname="Consolas", shape="none")

    rows_html = (
        '<TR>'
        '<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="8"><B>  Arrow Label  </B></FONT></TD>'
        '<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="8"><B>  What it is  </B></FONT></TD>'
        '<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="8"><B>  Example  </B></FONT></TD>'
        '</TR>'
    )
    for name, desc, ex in legend_entries:
        rows_html += (
            '<TR>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#475569">  {esc(name)}  </FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#475569">  {esc(desc)}  </FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#6B7280">  {esc(ex)}  </FONT></TD>'
            '</TR>'
        )
    legend_html = (
        '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
        'CELLPADDING="3" BGCOLOR="#F8FAFC" COLOR="#CBD5E1">'
        '<TR><TD COLSPAN="3" BGCOLOR="#334155">'
        '<FONT COLOR="white"><B>  INTER-FILE DATA FLOW LEGEND  </B></FONT></TD></TR>'
        f'{rows_html}</TABLE>'
    )
    lg.node("legend", "<" + legend_html + ">")
    lg.render(out_path + "_legend", cleanup=True)
    print(f"OK: {out_path}_legend.png")

    # ==================================================================
    # STITCH: graph on top, legend below
    # ==================================================================
    from PIL import Image

    img_graph = Image.open(out_path + "_graph.png")
    img_legend = Image.open(out_path + "_legend.png")

    # Scale legend to match graph width
    scale = img_graph.width / img_legend.width
    if scale < 1:
        new_w = img_graph.width
        new_h = int(img_legend.height * scale)
        img_legend = img_legend.resize((new_w, new_h), Image.LANCZOS)

    total_h = img_graph.height + img_legend.height + 40  # 40px gap
    combined = Image.new("RGB", (max(img_graph.width, img_legend.width), total_h), "white")
    combined.paste(img_graph, (0, 0))
    combined.paste(img_legend, (0, img_graph.height + 40))
    combined.save(out_path + ".png", dpi=(200, 200))
    print(f"OK: {out_path}.png ({combined.width}x{combined.height} px)")

    # Clean up intermediates
    os.remove(out_path + "_graph.png")
    os.remove(out_path + "_legend.png")


if __name__ == "__main__":
    build()
