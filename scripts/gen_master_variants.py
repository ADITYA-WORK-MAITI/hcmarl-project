"""
Generate 6 variants of the master inter-file data flow diagram.

Variant A: Numbered edges + legend key
Variant B: 3-panel split (Core / Agents+Env / Scripts+Infra)
Variant CF: Clustered subgraphs + edge bundling
Variant D: Two-level (6-box overview)
Variant E: Interactive HTML (vis.js)
Variant H: Adjacency matrix table
"""

import os, json
import graphviz
from PIL import Image, ImageDraw, ImageFont

os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + ";" + os.environ.get("PATH", "")

OUT = "diagrams/code_flowcharts"
os.makedirs(OUT, exist_ok=True)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Colours ──
CORE   = "#1E3A5F"
SUBPKG = "#2D4A22"
SCRIPT = "#6B3A1E"
GROUP  = "#4A2A5C"
DANG_IN  = "#DC2626"
DANG_OUT = "#3B82F6"
INTERN   = "#475569"
TEST_C   = "#10B981"

# ==================================================================
# SHARED DATA: All 28 boxes and all edges
# ==================================================================

BOXES = [
    # (id, title, subtitle, color, group, extra_rows)
    ("utils", "utils.py", "Logging, YAML, seeding, safe_log, clip", CORE, "core", None),
    ("three_cc_r", "three_cc_r.py", "3CC-r fatigue ODE (Eqs 1-8)", CORE, "core", None),
    ("ecbf", "ecbf_filter.py", "ECBF safety filter CBF-QP (Eqs 12-23)", CORE, "core", None),
    ("nswf", "nswf_allocator.py", "NSWF allocator (Eqs 31-33)", CORE, "core", None),
    ("pipeline", "pipeline.py", "End-to-end HC-MARL pipeline", CORE, "core", None),
    ("mmicrl", "mmicrl.py", "MM-ICRL CFDE flows (Eqs 9-11)", CORE, "core", None),
    ("real_data", "real_data_calibration.py", "Path G: WSD4FEDSRM calibration", CORE, "core", None),
    ("warehouse", "warehouse_env.py", "Single/Multi-agent warehouse env", CORE, "env", None),
    ("logger", "logger.py", "HCMARLLogger + W&B", CORE, "core", None),
    ("pettingzoo", "pettingzoo_wrapper.py", "PettingZoo AEC environment", SUBPKG, "env", None),
    ("reward_fn", "reward_functions.py", "nswf_reward() + safety_cost()", SUBPKG, "env", None),
    ("task_prof", "task_profiles.py", "TaskProfileManager", SUBPKG, "env", None),
    ("networks", "networks.py", "Actor, Critic, CostCritic nets", SUBPKG, "agents", None),
    ("mappo", "mappo.py", "MAPPO + RolloutBuffer", SUBPKG, "agents", None),
    ("ippo", "ippo.py", "IPPO (independent PPO)", SUBPKG, "agents", None),
    ("mappo_lag", "mappo_lag.py", "MAPPO-Lagrangian", SUBPKG, "agents", None),
    ("hcmarl_agent", "hcmarl_agent.py", "HCMARLAgent (pipeline-aware)", SUBPKG, "agents", None),
    ("legacy", "_legacy.py", "10 baseline classes", SUBPKG, "agents", None),
    ("omnisafe_w", "omnisafe_wrapper.py", "OmniSafe wrapper", SUBPKG, "agents", None),
    ("safepo_w", "safepo_wrapper.py", "SafePO wrapper", SUBPKG, "agents", None),
    ("train", "scripts/train.py", "Main training: all 7 methods", SCRIPT, "scripts", None),
    ("evaluate", "scripts/evaluate.py", "Evaluation: 9 metrics", SCRIPT, "scripts", None),
    ("run_abl", "scripts/run_ablations.py", "Ablation launcher", SCRIPT, "scripts", None),
    ("run_base", "scripts/run_baselines.py", "Baseline launcher", SCRIPT, "scripts", None),
    ("run_scale", "scripts/run_scaling.py", "Scaling launcher", SCRIPT, "scripts", None),
    ("configs", "config/*.yaml (20 files)", "Training, method, ablation, scaling", GROUP, "infra", None),
    ("notebooks", "notebooks/*.ipynb (4)", "Colab training notebooks", GROUP, "infra", None),
    ("tests", "tests/*.py (13 files)", "238 pytest tests", GROUP, "infra", None),
]

# (src, dst, label, description, example, edge_type)
# edge_type: "internal", "dangling_in", "dangling_out", "test"
EDGES = [
    # dangling-in (external → box)
    ("EXT_numpy", "three_cc_r", "np.array, solve_ivp, dataclass", "NumPy, SciPy ODE, dataclass", "np.array([1,3,5])", "dangling_in"),
    ("EXT_cvxpy", "ecbf", "cp.Variable, cp.Problem", "CVXPY optimisation primitives", "cp.Variable(3)", "dangling_in"),
    ("EXT_torch", "mmicrl", "nn.Module, nn.Linear", "PyTorch NN base classes", "nn.Linear(18, 64)", "dangling_in"),
    ("EXT_gym", "warehouse", "gym.Env, spaces", "Gymnasium env + spaces", "spaces.Discrete(5)", "dangling_in"),
    ("EXT_logging", "utils", "logging, yaml.safe_load", "Python logging + YAML", "yaml.safe_load(f)", "dangling_in"),
    ("EXT_wandb", "logger", "wandb", "Weights & Biases tracking", "wandb.init(project='hcmarl')", "dangling_in"),
    ("EXT_wsd", "real_data", "WSD4FEDSRM/ CSV files", "Real fatigue dataset (67 subj)", "WSD_P01.csv...P67.csv", "dangling_in"),
    ("EXT_omnisafe", "omnisafe_w", "omnisafe library", "OmniSafe safe RL lib", "omnisafe.Agent('PPOLag')", "dangling_in"),
    ("EXT_safepo", "safepo_w", "safepo library", "SafePO safe MARL lib", "safepo.MAPPO_Lagrangian()", "dangling_in"),
    # dangling-out (box → external)
    ("train", "EXT_ckpt1", "checkpoints .pt", "Saved model weights", "checkpoint_final.pt", "dangling_out"),
    ("train", "EXT_logs1", "summary.json, config.yaml", "Training summary + config", "{'best_reward':8.3}", "dangling_out"),
    ("evaluate", "EXT_results", "results JSON", "Eval metrics per method", "hcmarl_eval.json", "dangling_out"),
    ("mappo", "EXT_ckpt2", "checkpoint .pt file", "MAPPO saved weights", "mappo_seed0.pt", "dangling_out"),
    ("logger", "EXT_logcsv", "training_log.csv", "Per-episode metric log", "episode,reward,cost,...", "dangling_out"),
    ("logger", "EXT_wandb2", "W&B dashboard", "Live W&B dashboard", "wandb.ai/run/hcmarl-001", "dangling_out"),
    ("notebooks", "EXT_drive", "checkpoints + logs", "Colab -> Google Drive", "Drive/hcmarl/checkpoints/", "dangling_out"),
    ("notebooks", "EXT_evalj", "eval JSONs", "Eval result files", "hcmarl_eval.json", "dangling_out"),
    # internal edges
    ("utils", "three_cc_r", "clip_and_normalise", "Clipping + normalisation utility fn", "clip_and_normalise(x, lo, hi)", "internal"),
    ("utils", "nswf", "safe_log", "Numerically stable log utility", "safe_log(0.0) = -23.03", "internal"),
    ("utils", "pipeline", "logger, config dict", "Logger instance + parsed YAML dict", "{'env': {'n_workers': 3}}", "internal"),
    ("utils", "train", "seed_everything", "Global RNG seeding function", "seed_everything(42)", "internal"),
    ("three_cc_r", "ecbf", "MuscleParams, ThreeCCrState", "Muscle dataclass + ODE state", "MuscleParams(F=180, R=45)", "internal"),
    ("three_cc_r", "pipeline", "ThreeCCr, get_muscle, SHOULDER", "ODE solver + factory + preset", "ThreeCCr(SHOULDER).simulate(60)", "internal"),
    ("three_cc_r", "real_data", "MuscleParams, ThreeCCr, SHOULDER", "Muscle params + ODE + preset", "SHOULDER.F = 180.0", "internal"),
    ("three_cc_r", "pettingzoo", "get_muscle()", "Factory returning MuscleParams", "get_muscle('shoulder')", "internal"),
    ("ecbf", "pipeline", "C_filtered, ECBFParams", "Safety-filtered cmd + params", "C_filtered = array([0.8, 0.6])", "internal"),
    ("pipeline", "ecbf", "C_nominal", "Unfiltered RL/baseline command", "array([1.0, 0.9, 0.7])", "internal"),
    ("ecbf", "warehouse", "C_safe", "Per-step safe command (scalar)", "0.75", "internal"),
    ("nswf", "pipeline", "AllocationResult", "Task assignment + surpluses", "AllocationResult(assignments=[0,1,2])", "internal"),
    ("pipeline", "nswf", "utility_matrix, fatigue_levels", "Agent-task utils + fatigue", "U:(3,5), f:(3,) arrays", "internal"),
    ("nswf", "warehouse", "D_i values", "Per-agent Nash disagreement", "array([0.1, 0.2, 0.15])", "internal"),
    ("real_data", "mmicrl", "demo trajectories", "Calibrated demos from WSD data", "[(F_t, R_t, r_t), ...]", "internal"),
    ("real_data", "train", "calibration results", "Path G fitted params", "{'F_opt':185.2, 'r2':0.91}", "internal"),
    ("mmicrl", "ecbf", "theta_max", "Per-muscle fatigue threshold", "array([0.85, 0.72])", "internal"),
    ("mmicrl", "train", "MI, type_proportions, theta_per_type", "Mutual info + mix + thresholds", "MI=2.3, [0.4,0.6]", "internal"),
    ("warehouse", "reward_fn", "productivity, fatigue dict", "Task load + per-muscle MF", "prod=5.2, {'shoulder':0.6}", "internal"),
    ("reward_fn", "warehouse", "reward float, cost float", "NSWF reward + safety cost", "r=3.14, c=0.02", "internal"),
    ("reward_fn", "pettingzoo", "nswf_reward, safety_cost", "Reward/cost callables", "nswf_reward(prod, fatigue)", "internal"),
    ("reward_fn", "mappo_lag", "cost float", "Safety constraint cost", "0.05", "internal"),
    ("warehouse", "mmicrl", "env, policy", "Env + policy for demos", "(gym.Env, nn.Module)", "internal"),
    ("warehouse", "networks", "obs tensor (obs_dim,)", "Observation vector from env", "array shape (18,)", "internal"),
    ("warehouse", "train", "env instances", "Constructed env objects", "SingleWorkerWarehouseEnv()", "internal"),
    ("task_prof", "pettingzoo", "demand_vector, task_intensity", "Per-task demands + intensity", "array([0.3,0.6,0.1]), 0.8", "internal"),
    ("configs", "task_prof", "config YAML", "Parsed YAML config dict", "task_profiles.yaml", "internal"),
    ("configs", "train", "env params, PPO hyperparams, ECBF gains", "Full training config", "lr=3e-4, gamma_ecbf=0.1", "internal"),
    ("configs", "pipeline", "ECBF params, NSWF params", "Default safety/alloc params", "gamma_1=1.0, alpha=0.5", "internal"),
    ("configs", "run_abl", "ablation configs", "5 ablation YAML files", "ablation_no_ecbf.yaml", "internal"),
    ("configs", "run_base", "per-method configs", "6 per-method YAML files", "mappo_config.yaml", "internal"),
    ("configs", "run_scale", "scaling configs", "5 scaling YAML files", "scaling_n3.yaml", "internal"),
    ("configs", "notebooks", "YAML configs", "Configs consumed by notebooks", "hcmarl_full_config.yaml", "internal"),
    ("pettingzoo", "networks", "global_state tensor", "Concatenated obs for critic", "tensor shape (N*obs_dim,)", "internal"),
    ("pettingzoo", "mappo", "observations dict, global_state", "Per-agent obs + shared state", "{'agent_0': tensor, ...}", "internal"),
    ("mappo", "pettingzoo", "actions dict", "Per-agent actions", "{'agent_0':2, 'agent_1':0}", "internal"),
    ("ippo", "pettingzoo", "actions dict", "Per-agent actions", "{'agent_0':2, 'agent_1':0}", "internal"),
    ("mappo_lag", "pettingzoo", "actions, value, cost_value", "Actions + V(s) + Vc(s)", "acts, 3.14, 0.02", "internal"),
    ("hcmarl_agent", "pettingzoo", "actions dict", "Per-agent actions", "{'agent_0':2, 'agent_1':0}", "internal"),
    ("legacy", "pettingzoo", "actions dict", "Per-agent actions", "{'agent_0':2}", "internal"),
    ("pettingzoo", "legacy", "observations", "Raw obs for baselines", "{'agent_0': array([...])}", "internal"),
    ("pettingzoo", "train", "WarehousePettingZoo", "PettingZoo AEC env class", "WarehousePettingZoo(n=3)", "internal"),
    ("networks", "mappo", "ActorNetwork, CriticNetwork", "Actor + Critic nets", "ActorNetwork(18, 5)", "internal"),
    ("networks", "ippo", "ActorNetwork, CriticNetwork", "Actor + Critic nets", "ActorNetwork(18, 5)", "internal"),
    ("networks", "mappo_lag", "ActorNetwork, CriticNetwork, CostCriticNetwork", "All 3 nets", "CostCriticNetwork(54)", "internal"),
    ("mappo", "ippo", "RolloutBuffer", "Shared rollout buffer", "RolloutBuffer(capacity=2048)", "internal"),
    ("mappo", "hcmarl_agent", "MAPPO", "MAPPO agent class", "MAPPO(obs_dim=18, n_actions=5)", "internal"),
    ("mappo_lag", "safepo_w", "MAPPOLagrangian", "Lagrangian agent class", "MAPPOLagrangian(...)", "internal"),
    ("hcmarl_agent", "train", "HCMARLAgent", "Pipeline-aware agent", "HCMARLAgent(pipeline, mappo)", "internal"),
    ("mappo", "train", "MAPPO, loss metrics", "MAPPO + training losses", "policy_loss=0.03", "internal"),
    ("ippo", "train", "IPPO", "Independent PPO agent", "IPPO(obs_dim=18)", "internal"),
    ("mappo_lag", "train", "MAPPOLagrangian, loss + lambda", "Lagrangian + dual var", "lambda=0.5", "internal"),
    ("omnisafe_w", "train", "OmniSafeWrapper, actions", "OmniSafe wrapper + acts", "OmniSafeWrapper('PPOLag')", "internal"),
    ("safepo_w", "train", "SafePOWrapper, actions", "SafePO wrapper + acts", "SafePOWrapper(mappo_lag)", "internal"),
    ("mmicrl", "train", "DemonstrationCollector, MMICRL", "Demo collector + model", "MMICRL(obs_dim=18)", "internal"),
    ("logger", "train", "HCMARLLogger", "Logger class", "HCMARLLogger(log_dir)", "internal"),
    ("train", "logger", "episode_data", "Per-episode metrics", "{'reward':4.2, 'cost':0.01}", "internal"),
    ("pipeline", "train", "step_result dict", "Pipeline step output", "{'fatigue':[...], 'cost':0.02}", "internal"),
    ("train", "pipeline", "utility_matrix", "Agent-task utility matrix", "(N,M) float array", "internal"),
    ("pipeline", "evaluate", "history list", "Episode trajectory for eval", "[step_result_0, ...]", "internal"),
    ("train", "evaluate", "create_agent()", "Agent factory from train", "create_agent('hcmarl', cfg)", "internal"),
    ("pettingzoo", "evaluate", "WarehousePettingZoo", "PettingZoo env class", "WarehousePettingZoo(n=3)", "internal"),
    ("logger", "evaluate", "METRIC_NAMES", "9 eval metric names", "['mean_reward',...]", "internal"),
    ("run_abl", "train", "ablation_name, config_path", "Ablation ID + path", "'no_ecbf', 'config/...'", "internal"),
    ("run_base", "train", "method, config_path", "Method + path", "'mappo', 'config/...'", "internal"),
    ("run_scale", "train", "n_workers, config_path", "N + path", "6, 'config/scaling_n6.yaml'", "internal"),
    ("train", "notebooks", "scripts/train.py", "Training script path", "!python scripts/train.py", "internal"),
    ("run_base", "notebooks", "scripts/run_baselines.py", "Baseline launcher path", "!python scripts/run_baselines.py", "internal"),
    ("run_abl", "notebooks", "scripts/run_ablations.py", "Ablation launcher path", "!python scripts/run_ablations.py", "internal"),
    ("run_scale", "notebooks", "scripts/run_scaling.py", "Scaling launcher path", "!python scripts/run_scaling.py", "internal"),
    # test edges
    ("three_cc_r", "tests", "SHOULDER, ThreeCCr, MuscleParams, ...", "Core 3CC-r types", "SHOULDER.F = 180.0", "test"),
    ("ecbf", "tests", "ECBFParams, ECBFFilter", "Filter classes", "ECBFFilter(params, state)", "test"),
    ("nswf", "tests", "NSWFAllocator, NSWFParams, AllocationResult", "Allocator classes", "NSWFAllocator(NSWFParams())", "test"),
    ("pipeline", "tests", "HCMARLPipeline, TaskProfile, WorkerState", "Pipeline + data classes", "HCMARLPipeline.from_config()", "test"),
    ("mmicrl", "tests", "CFDE, _MADE, MMICRL, ...", "Flow model + collector", "CFDE(dim=6, n_flows=4)", "test"),
    ("real_data", "tests", "predict_endurance_time, POPULATION_FR, ...", "Calibration fn + constants", "POPULATION_FR=(180, 45)", "test"),
    ("warehouse", "tests", "SingleWorkerWarehouseEnv, ...", "Env classes", "SingleWorkerWarehouseEnv()", "test"),
    ("pettingzoo", "tests", "WarehousePettingZoo", "PettingZoo env", "WarehousePettingZoo(n=3)", "test"),
    ("hcmarl_agent", "tests", "HCMARLAgent", "Pipeline-aware agent", "HCMARLAgent(pipeline, mappo)", "test"),
    ("logger", "tests", "HCMARLLogger", "Logger class", "HCMARLLogger(log_dir)", "test"),
]

# Map box IDs to groups for panel/cluster assignment
BOX_GROUPS = {b[0]: b[4] for b in BOXES}

# Back-edges (should not constrain layout)
BACK_EDGES = {
    ("pipeline", "ecbf"), ("pipeline", "nswf"), ("mmicrl", "ecbf"),
    ("warehouse", "reward_fn"), ("warehouse", "mmicrl"),
    ("mappo", "pettingzoo"), ("ippo", "pettingzoo"), ("mappo_lag", "pettingzoo"),
    ("hcmarl_agent", "pettingzoo"), ("legacy", "pettingzoo"),
    ("train", "logger"), ("train", "pipeline"),
    ("run_abl", "train"), ("run_base", "train"), ("run_scale", "train"),
}

# Box IDs that are real project files (not external phantoms)
REAL_IDS = {b[0] for b in BOXES}


def make_box(g, nid, title, subtitle, color, extra_rows=None):
    sub = f'<FONT POINT-SIZE="8" COLOR="#CBD5E1">  {esc(subtitle)}  </FONT>'
    hdr = (
        f'<TR><TD COLSPAN="1" BGCOLOR="{color}" ALIGN="LEFT">'
        f'<FONT COLOR="white" FACE="Consolas" POINT-SIZE="10"><B>  {esc(title)}  </B></FONT>'
        f'</TD></TR>'
        f'<TR><TD ALIGN="LEFT">{sub}</TD></TR>'
    )
    extras = ""
    if extra_rows:
        for row in extra_rows:
            extras += (
                f'<TR><TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="7" '
                f'COLOR="#6B7280">  {esc(row)}  </FONT></TD></TR>'
            )
    html = (
        f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
        f'CELLPADDING="3" COLOR="#CBD5E1">{hdr}{extras}</TABLE>'
    )
    g.node(nid, "<" + html + ">")


def add_all_boxes(g, filter_ids=None):
    for nid, title, sub, color, grp, extra in BOXES:
        if filter_ids and nid not in filter_ids:
            continue
        make_box(g, nid, title, sub, color, extra)


def build_legend_html(entries, fontsize="7"):
    """Build legend HTML table from list of (num, label, desc, example)."""
    rows = (
        '<TR>'
        f'<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="{fontsize}"><B>  #  </B></FONT></TD>'
        f'<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="{fontsize}"><B>  Arrow Label  </B></FONT></TD>'
        f'<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="{fontsize}"><B>  What it is  </B></FONT></TD>'
        f'<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="{fontsize}"><B>  Example  </B></FONT></TD>'
        '</TR>'
    )
    for num, label, desc, ex in entries:
        rows += (
            '<TR>'
            f'<TD ALIGN="CENTER"><FONT POINT-SIZE="{fontsize}" COLOR="#DC2626"><B>  {num}  </B></FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="{fontsize}" COLOR="#475569">  {esc(label)}  </FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="{fontsize}" COLOR="#475569">  {esc(desc)}  </FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="{fontsize}" COLOR="#6B7280">  {esc(ex)}  </FONT></TD>'
            '</TR>'
        )
    return (
        '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
        'CELLPADDING="3" BGCOLOR="#F8FAFC" COLOR="#CBD5E1">'
        '<TR><TD COLSPAN="4" BGCOLOR="#334155">'
        '<FONT COLOR="white"><B>  INTER-FILE DATA FLOW LEGEND  </B></FONT></TD></TR>'
        f'{rows}</TABLE>'
    )


def stitch_graph_legend(graph_path, legend_path, out_path):
    """Stitch graph PNG on top, legend PNG below."""
    img_g = Image.open(graph_path)
    img_l = Image.open(legend_path)
    # Scale legend to graph width
    if img_l.width > img_g.width:
        ratio = img_g.width / img_l.width
        img_l = img_l.resize((img_g.width, int(img_l.height * ratio)), Image.LANCZOS)
    w = max(img_g.width, img_l.width)
    h = img_g.height + img_l.height + 40
    combined = Image.new("RGB", (w, h), "white")
    combined.paste(img_g, (0, 0))
    combined.paste(img_l, (0, img_g.height + 40))
    combined.save(out_path, dpi=(200, 200))
    os.remove(graph_path)
    os.remove(legend_path)
    return combined.size


# ==================================================================
# VARIANT A: Numbered edges
# ==================================================================
def variant_a():
    print("--- Variant A: Numbered edges ---")

    # Assign unique numbers to edges, dedup identical labels
    label_to_num = {}
    num_counter = [0]
    legend_entries = []

    def get_num(label, desc, ex):
        if label in label_to_num:
            return label_to_num[label]
        num_counter[0] += 1
        n = num_counter[0]
        label_to_num[label] = n
        legend_entries.append((n, label, desc, ex))
        return n

    g = graphviz.Digraph("variant_a", format="png", engine="dot")
    g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
           nodesep="0.3", ranksep="0.7", margin="0.2", newrank="true",
           concentrate="true", size="30,50", ratio="compress")
    g.attr("node", fontname="Consolas", fontsize="10", shape="none")
    g.attr("edge", fontname="Consolas", fontsize="9")

    add_all_boxes(g)

    # Rank constraints
    with g.subgraph() as s:
        s.attr(rank="min")
        for n in ["utils", "three_cc_r", "configs"]:
            s.node(n)
    with g.subgraph() as s:
        s.attr(rank="same")
        for n in ["train", "evaluate"]:
            s.node(n)

    pc = [0]
    for src, dst, label, desc, ex, etype in EDGES:
        n = get_num(label, desc, ex)
        nlabel = f" {n} "

        if etype == "dangling_in":
            pc[0] += 1
            pid = f"_din_{pc[0]}"
            g.node(pid, "", shape="point", width="0.06", color=DANG_IN)
            g.edge(pid, dst, label=nlabel, style="dashed", color=DANG_IN,
                   fontname="Consolas", fontsize="8", fontcolor=DANG_IN,
                   constraint="false")
        elif etype == "dangling_out":
            pc[0] += 1
            pid = f"_dout_{pc[0]}"
            g.node(pid, "", shape="point", width="0.06", color=DANG_OUT)
            g.edge(src, pid, label=nlabel, style="dotted", color=DANG_OUT,
                   fontname="Consolas", fontsize="8", fontcolor=DANG_OUT,
                   constraint="false")
        elif etype == "test":
            g.edge(src, dst, label=nlabel, color=TEST_C, fontcolor=TEST_C,
                   fontname="Consolas", fontsize="8", constraint="false")
        else:
            cons = "false" if (src, dst) in BACK_EDGES else "true"
            g.edge(src, dst, label=nlabel, color=INTERN,
                   fontname="Consolas", fontsize="8", fontcolor="#1E3A5F",
                   constraint=cons)

    # Invisible edge to pull tests down
    g.edge("notebooks", "tests", style="invis")

    # Render graph
    gpath = os.path.join(OUT, "master_A_graph")
    g.render(gpath, cleanup=True)

    # Render legend
    lg = graphviz.Digraph("legend_a", format="png")
    lg.attr(dpi="200", bgcolor="white", margin="0.2")
    lg.attr("node", fontname="Consolas", shape="none")
    lg.node("legend", "<" + build_legend_html(legend_entries) + ">")
    lpath = os.path.join(OUT, "master_A_legend")
    lg.render(lpath, cleanup=True)

    sz = stitch_graph_legend(gpath + ".png", lpath + ".png",
                             os.path.join(OUT, "master_A_numbered.png"))
    print(f"OK: master_A_numbered.png ({sz[0]}x{sz[1]})")


# ==================================================================
# VARIANT B: 3-panel
# ==================================================================
def variant_b():
    print("--- Variant B: 3-panel ---")

    panels = {
        "B1_core": {
            "ids": {"utils", "three_cc_r", "ecbf", "nswf", "pipeline", "mmicrl",
                    "real_data", "logger"},
            "title": "Panel B1: Core Algorithm Pipeline",
        },
        "B2_env_agents": {
            "ids": {"warehouse", "pettingzoo", "reward_fn", "task_prof", "networks",
                    "mappo", "ippo", "mappo_lag", "hcmarl_agent", "legacy",
                    "omnisafe_w", "safepo_w"},
            "title": "Panel B2: Environments and Agents",
        },
        "B3_scripts": {
            "ids": {"train", "evaluate", "run_abl", "run_base", "run_scale",
                    "configs", "notebooks", "tests"},
            "title": "Panel B3: Scripts, Configs and Experiments",
        },
    }

    for panel_key, panel_info in panels.items():
        ids = panel_info["ids"]
        g = graphviz.Digraph(panel_key, format="png", engine="dot")
        g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
               nodesep="0.3", ranksep="0.7", margin="0.2",
               label=f"<<B>{panel_info['title']}</B>>",
               labelloc="t", fontsize="14")
        g.attr("node", fontname="Consolas", fontsize="10", shape="none")
        g.attr("edge", fontname="Consolas", fontsize="7")

        add_all_boxes(g, ids)

        pc = [0]
        for src, dst, label, desc, ex, etype in EDGES:
            src_in = src in ids or (etype == "dangling_in" and dst in ids)
            dst_in = dst in ids or (etype == "dangling_out" and src in ids)

            if etype == "dangling_in" and dst in ids:
                pc[0] += 1
                pid = f"_din_{pc[0]}"
                g.node(pid, "", shape="point", width="0.06", color=DANG_IN)
                g.edge(pid, dst, label=f"  {label}  ", style="dashed",
                       color=DANG_IN, fontname="Consolas", fontsize="6",
                       fontcolor=DANG_IN, constraint="false")
            elif etype == "dangling_out" and src in ids:
                pc[0] += 1
                pid = f"_dout_{pc[0]}"
                g.node(pid, "", shape="point", width="0.06", color=DANG_OUT)
                g.edge(src, pid, label=f"  {label}  ", style="dotted",
                       color=DANG_OUT, fontname="Consolas", fontsize="6",
                       fontcolor=DANG_OUT, constraint="false")
            elif src in ids and dst in ids:
                color = TEST_C if etype == "test" else INTERN
                cons = "false" if (src, dst) in BACK_EDGES or etype == "test" else "true"
                g.edge(src, dst, label=f"  {label}  ", color=color,
                       fontname="Consolas", fontsize="7", fontcolor="#334155",
                       constraint=cons)
            elif src in ids and dst not in ids:
                # Cross-panel outgoing: show as dangling
                pc[0] += 1
                pid = f"_cross_{pc[0]}"
                g.node(pid, esc(dst), shape="plaintext", fontname="Consolas",
                       fontsize="7", fontcolor="#94A3B8")
                g.edge(src, pid, label=f"  {label}  ", style="dashed",
                       color="#94A3B8", fontname="Consolas", fontsize="6",
                       fontcolor="#94A3B8", constraint="false")
            elif dst in ids and src in ids == False:
                pass  # handled above

        out = os.path.join(OUT, f"master_{panel_key}")
        g.render(out, cleanup=True)
        print(f"OK: {out}.png")


# ==================================================================
# VARIANT CF: Clustered + Edge bundling
# ==================================================================
def variant_cf():
    print("--- Variant CF: Clustered + edge bundling ---")

    g = graphviz.Digraph("variant_cf", format="png", engine="dot")
    g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
           nodesep="0.25", ranksep="0.7", margin="0.2", newrank="true",
           compound="true", size="30,50", ratio="compress")
    g.attr("node", fontname="Consolas", fontsize="10", shape="none")
    g.attr("edge", fontname="Consolas", fontsize="7")

    clusters = {
        "core": ("Core Algorithm (Phase A)", "#1E3A5F",
                 ["utils", "three_cc_r", "ecbf", "nswf", "pipeline", "mmicrl", "real_data", "logger"]),
        "env": ("Environments", "#2D4A22",
                ["warehouse", "pettingzoo", "reward_fn", "task_prof"]),
        "agents": ("Agents and Baselines", "#4A2A5C",
                   ["networks", "mappo", "ippo", "mappo_lag", "hcmarl_agent", "legacy", "omnisafe_w", "safepo_w"]),
        "scripts": ("Training Scripts", "#6B3A1E",
                    ["train", "evaluate", "run_abl", "run_base", "run_scale"]),
        "infra": ("Configs, Notebooks and Tests", "#4A4A4A",
                  ["configs", "notebooks", "tests"]),
    }

    for cname, (clabel, ccolor, cids) in clusters.items():
        with g.subgraph(name=f"cluster_{cname}") as c:
            c.attr(label=f"<<B>{clabel}</B>>", style="rounded,dashed",
                   color=ccolor, fontname="Consolas", fontsize="11",
                   fontcolor=ccolor, penwidth="2", margin="12")
            for nid in cids:
                box = [b for b in BOXES if b[0] == nid][0]
                make_box(c, nid, box[1], box[2], box[3], box[5])

    # Deduplicate edges: merge same (src, dst) with different labels
    merged = {}
    for src, dst, label, desc, ex, etype in EDGES:
        key = (src, dst, etype)
        if key not in merged:
            merged[key] = (src, dst, label, desc, ex, etype)
        # skip duplicates (same src/dst pair)

    # Bundle: merge agent → pettingzoo "actions dict" into one edge from cluster
    # We do this by skipping individual agent→pettingzoo and adding one bundled
    bundled_skip = set()
    agents_to_pz = [e for e in EDGES if e[1] == "pettingzoo" and e[0] in
                    {"mappo", "ippo", "mappo_lag", "hcmarl_agent", "legacy"} and e[2].startswith("actions")]
    if agents_to_pz:
        for e in agents_to_pz:
            bundled_skip.add((e[0], e[1], e[2]))
        # Add one bundled edge from mappo (as representative) with combined label
        g.edge("mappo", "pettingzoo", label="  actions dict\n(all 5 agents)  ",
               color=INTERN, fontname="Consolas", fontsize="7", fontcolor="#334155",
               constraint="false", style="bold")

    pc = [0]
    for src, dst, label, desc, ex, etype in EDGES:
        if (src, dst, label) in bundled_skip:
            continue

        if etype == "dangling_in":
            pc[0] += 1
            pid = f"_din_{pc[0]}"
            g.node(pid, "", shape="point", width="0.06", color=DANG_IN)
            g.edge(pid, dst, label=f"  {label}  ", style="dashed", color=DANG_IN,
                   fontname="Consolas", fontsize="6", fontcolor=DANG_IN, constraint="false")
        elif etype == "dangling_out":
            pc[0] += 1
            pid = f"_dout_{pc[0]}"
            g.node(pid, "", shape="point", width="0.06", color=DANG_OUT)
            g.edge(src, pid, label=f"  {label}  ", style="dotted", color=DANG_OUT,
                   fontname="Consolas", fontsize="6", fontcolor=DANG_OUT, constraint="false")
        elif etype == "test":
            g.edge(src, dst, label=f"  {label}  ", color=TEST_C, fontcolor=TEST_C,
                   fontname="Consolas", fontsize="6", constraint="false")
        else:
            cons = "false" if (src, dst) in BACK_EDGES else "true"
            g.edge(src, dst, label=f"  {label}  ", color=INTERN,
                   fontname="Consolas", fontsize="7", fontcolor="#334155", constraint=cons)

    g.edge("notebooks", "tests", style="invis")

    out = os.path.join(OUT, "master_CF_clustered")
    g.render(out, cleanup=True)
    print(f"OK: {out}.png")


# ==================================================================
# VARIANT D: Two-level (6-box overview)
# ==================================================================
def variant_d():
    print("--- Variant D: 6-box overview ---")

    g = graphviz.Digraph("variant_d", format="png", engine="dot")
    g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
           nodesep="0.6", ranksep="1.0", margin="0.3")
    g.attr("node", fontname="Consolas", fontsize="12", shape="none")
    g.attr("edge", fontname="Consolas", fontsize="9")

    make_box(g, "core", "Core Algorithm (9 files)",
             "three_cc_r, ecbf_filter, nswf_allocator,\npipeline, mmicrl, real_data_calibration,\nutils, logger",
             CORE)
    make_box(g, "env", "Environments (4 files)",
             "warehouse_env, pettingzoo_wrapper,\nreward_functions, task_profiles",
             SUBPKG)
    make_box(g, "agents", "Agents and Baselines (8 files)",
             "networks, mappo, ippo, mappo_lag,\nhcmarl_agent, _legacy,\nomnisafe_wrapper, safepo_wrapper",
             "#4A2A5C")
    make_box(g, "scripts", "Training Scripts (5 files)",
             "train, evaluate, run_ablations,\nrun_baselines, run_scaling",
             SCRIPT)
    make_box(g, "cfg", "Configs and Notebooks (24 files)",
             "20 YAML configs + 4 Colab notebooks",
             GROUP)
    make_box(g, "tst", "Tests (13 files)",
             "238 pytest tests across all modules",
             "#4A4A4A")

    # Cluster-level data flow
    g.edge("core", "env", label="  MuscleParams, C_safe, D_i,\nget_muscle(), reward/cost  ")
    g.edge("env", "agents", label="  obs tensor, global_state,\ndemand_vector, task_intensity  ")
    g.edge("agents", "env", label="  actions dict,\nactions, value, cost_value  ", constraint="false")
    g.edge("agents", "scripts", label="  HCMARLAgent, MAPPO, IPPO,\nMAPPOLagrangian, loss metrics  ")
    g.edge("core", "scripts", label="  step_result dict, calibration results,\nDemonstrationCollector, MMICRL,\nHCMARLLogger, seed_everything  ")
    g.edge("env", "scripts", label="  WarehousePettingZoo, env instances  ")
    g.edge("cfg", "scripts", label="  env params, PPO hyperparams,\nablation/method/scaling configs  ")
    g.edge("cfg", "core", label="  ECBF params, NSWF params  ", constraint="false")

    # Dangling in
    pc = [0]
    for src_label, target, arrow_label in [
        ("numpy, scipy, dataclasses", "core", "np.array, solve_ivp, dataclass"),
        ("cvxpy", "core", "cp.Variable, cp.Problem"),
        ("torch.nn", "core", "nn.Module, nn.Linear"),
        ("gymnasium", "env", "gym.Env, spaces"),
        ("wandb", "core", "wandb"),
        ("data/WSD4FEDSRM/", "core", "WSD4FEDSRM/ CSV files"),
    ]:
        pc[0] += 1
        pid = f"_din_{pc[0]}"
        g.node(pid, "", shape="point", width="0.06", color=DANG_IN)
        g.edge(pid, target, label=f"  {arrow_label}  ", style="dashed",
               color=DANG_IN, fontname="Consolas", fontsize="7", fontcolor=DANG_IN,
               constraint="false")

    # Dangling out
    for src, tgt_label, arrow_label in [
        ("scripts", "checkpoints/", "checkpoints .pt, summary.json"),
        ("scripts", "results/", "results JSON, eval JSONs"),
        ("core", "logs/", "training_log.csv"),
        ("core", "wandb.ai", "W&B dashboard"),
    ]:
        pc[0] += 1
        pid = f"_dout_{pc[0]}"
        g.node(pid, "", shape="point", width="0.06", color=DANG_OUT)
        g.edge(src, pid, label=f"  {arrow_label}  ", style="dotted",
               color=DANG_OUT, fontname="Consolas", fontsize="7", fontcolor=DANG_OUT,
               constraint="false")

    # Test edge
    g.edge("core", "tst", label="  all public classes + constants  ", color=TEST_C,
           fontcolor=TEST_C, fontname="Consolas", fontsize="8")
    g.edge("env", "tst", label="  env classes  ", color=TEST_C,
           fontcolor=TEST_C, fontname="Consolas", fontsize="8")
    g.edge("agents", "tst", label="  agent classes  ", color=TEST_C,
           fontcolor=TEST_C, fontname="Consolas", fontsize="8", constraint="false")

    out = os.path.join(OUT, "master_D_overview")
    g.render(out, cleanup=True)
    print(f"OK: {out}.png")


# ==================================================================
# VARIANT E: Interactive HTML (vis.js)
# ==================================================================
def variant_e():
    print("--- Variant E: Interactive HTML ---")

    nodes_js = []
    color_map = {CORE: "#1E3A5F", SUBPKG: "#2D4A22", SCRIPT: "#6B3A1E", GROUP: "#4A2A5C"}

    for nid, title, sub, color, grp, _ in BOXES:
        nodes_js.append({
            "id": nid,
            "label": title,
            "title": f"<b>{title}</b><br>{sub}",
            "color": {"background": color, "border": color,
                      "highlight": {"background": "#F59E0B", "border": "#D97706"}},
            "font": {"color": "white", "face": "Consolas", "size": 12},
            "shape": "box",
            "group": grp,
        })

    edges_js = []
    eid = 0
    for src, dst, label, desc, ex, etype in EDGES:
        if not (src in REAL_IDS or src.startswith("EXT_")) or not (dst in REAL_IDS or dst.startswith("EXT_")):
            continue
        if src.startswith("EXT_") or dst.startswith("EXT_"):
            continue  # skip dangling in HTML for clarity; show only internal

        eid += 1
        color = TEST_C if etype == "test" else INTERN
        dashes = etype == "test"
        edges_js.append({
            "from": src, "to": dst,
            "label": label, "title": f"{label}: {desc}<br>e.g. {ex}",
            "color": {"color": color, "highlight": "#F59E0B"},
            "font": {"size": 8, "face": "Consolas", "color": "#334155", "strokeWidth": 0},
            "arrows": "to",
            "dashes": dashes,
        })

    html = f"""<!DOCTYPE html>
<html><head>
<title>HC-MARL Inter-File Data Flow</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
body {{ margin: 0; padding: 0; font-family: Consolas, monospace; }}
#graph {{ width: 100vw; height: 100vh; }}
#info {{ position: fixed; top: 10px; left: 10px; background: rgba(255,255,255,0.95);
         padding: 12px; border: 1px solid #CBD5E1; border-radius: 6px; font-size: 12px;
         max-width: 300px; z-index: 10; }}
</style>
</head><body>
<div id="info">
<b>HC-MARL Inter-File Data Flow</b><br>
28 boxes = files, arrows = data/variables.<br>
<span style="color:#1E3A5F">&#9632;</span> Core
<span style="color:#2D4A22">&#9632;</span> Sub-pkg
<span style="color:#6B3A1E">&#9632;</span> Scripts
<span style="color:#4A2A5C">&#9632;</span> Grouped<br>
<b>Hover</b> edges for details. <b>Scroll</b> to zoom. <b>Drag</b> to pan.
</div>
<div id="graph"></div>
<script>
var nodes = new vis.DataSet({json.dumps(nodes_js)});
var edges = new vis.DataSet({json.dumps(edges_js)});
var container = document.getElementById('graph');
var data = {{ nodes: nodes, edges: edges }};
var options = {{
    layout: {{ hierarchical: {{ direction: 'UD', sortMethod: 'hubsize', nodeSpacing: 150, levelSeparation: 200 }} }},
    physics: {{ enabled: false }},
    edges: {{ smooth: {{ type: 'cubicBezier' }}, font: {{ align: 'middle' }} }},
    interaction: {{ hover: true, tooltipDelay: 100 }},
}};
new vis.Network(container, data, options);
</script>
</body></html>"""

    out = os.path.join(OUT, "master_E_interactive.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"OK: {out}")


# ==================================================================
# VARIANT H: Adjacency matrix
# ==================================================================
def variant_h():
    print("--- Variant H: Adjacency matrix ---")

    box_ids = [b[0] for b in BOXES]
    n = len(box_ids)
    id_to_idx = {bid: i for i, bid in enumerate(box_ids)}

    # Build matrix
    matrix = [[None] * n for _ in range(n)]
    for src, dst, label, desc, ex, etype in EDGES:
        if src in id_to_idx and dst in id_to_idx:
            i, j = id_to_idx[src], id_to_idx[dst]
            if matrix[i][j]:
                matrix[i][j] += f"\n{label}"
            else:
                matrix[i][j] = label

    # Short labels for headers
    short = [b[1].replace(".py", "").replace("scripts/", "").replace("config/", "cfg/")
             .replace("notebooks/", "nb/").replace("tests/", "t/").replace("*.yaml (20 files)", "*.yaml")
             .replace("*.ipynb (4)", "*.ipynb").replace("*.py (13 files)", "*.py")
             for b in BOXES]

    # Render with PIL
    cell_w, cell_h = 110, 20
    hdr_w = 180
    hdr_h = 120
    img_w = hdr_w + n * cell_w + 2
    img_h = hdr_h + n * cell_h + 2

    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("consola.ttf", 9)
        font_hdr = ImageFont.truetype("consola.ttf", 8)
    except Exception:
        font = ImageFont.load_default()
        font_hdr = font

    # Column headers (rotated text is hard in PIL, use horizontal)
    for j in range(n):
        x = hdr_w + j * cell_w
        draw.text((x + 2, hdr_h - 12), short[j][:14], fill="#1E3A5F", font=font_hdr)
        draw.line([(x, hdr_h), (x, img_h)], fill="#E2E8F0")

    # Row headers + cells
    for i in range(n):
        y = hdr_h + i * cell_h
        draw.text((2, y + 3), short[i][:22], fill="#1E3A5F", font=font_hdr)
        draw.line([(hdr_w, y), (img_w, y)], fill="#E2E8F0")

        for j in range(n):
            x = hdr_w + j * cell_w
            val = matrix[i][j]
            if val:
                # Color: green for test, blue for internal
                is_test = any(e[5] == "test" and e[0] == box_ids[i] and e[1] == box_ids[j] for e in EDGES)
                color = "#10B981" if is_test else "#475569"
                # Truncate label to fit cell
                txt = val.split("\n")[0][:16]
                draw.rectangle([(x+1, y+1), (x+cell_w-1, y+cell_h-1)], fill="#F0F9FF")
                draw.text((x + 2, y + 3), txt, fill=color, font=font)

    # Border
    draw.rectangle([(hdr_w, hdr_h), (img_w-1, img_h-1)], outline="#94A3B8")

    # Title
    draw.text((2, 2), "HC-MARL Inter-File Data Flow Matrix (rows=source, cols=target)", fill="#0F172A", font=font)

    out = os.path.join(OUT, "master_H_matrix.png")
    img.save(out, dpi=(200, 200))
    print(f"OK: {out} ({img_w}x{img_h})")


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    variant_a()
    variant_b()
    variant_cf()
    variant_d()
    variant_e()
    variant_h()
    print("\n=== ALL 6 VARIANTS DONE ===")
