"""
Generate code flowcharts for scripts/*.py and setup.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from flowchart_framework import FlowchartBuilder, HDR, CLS, FUNC, CONST, PROP, UTIL

OUT = "diagrams/code_flowcharts"
os.makedirs(OUT, exist_ok=True)


# =====================================================================
# scripts/train.py
# =====================================================================
def gen_train():
    fb = FlowchartBuilder("scripts/train.py",
        "Main training script: 7 methods, MMICRL pre-train, full loop",
        lines="545",
        equations="Eqs 2-4, 12-23, 31-35",
        imports_desc="argparse, yaml, torch, numpy, all hcmarl modules")

    fb.make_node("methods", "METHODS registry (dict)", CONST, [
        ("hcmarl", "HC-MARL (MAPPO+ECBF+NSWF)"),
        ("mappo", "MAPPO (no safety filter)"),
        ("ippo", "IPPO (independent)"),
        ("mappo_lag", "MAPPO-Lagrangian"),
    ])

    fb.make_node("create_agent", "create_agent(method, obs_dim, ...)", FUNC, [
        ("Input", "method str, obs/action dims, cfg, device"),
        ("Logic", "switch on method -> HCMARLAgent/MAPPO/IPPO/..."),
        ("Return", "agent instance"),
    ])

    fb.make_node("mmicrl_pretrain", "run_mmicrl_pretrain(cfg)", FUNC, [
        ("1", "DemonstrationCollector + RandomPolicy"),
        ("2", "collect_from_env (50 episodes)"),
        ("3", "MMICRL.fit(collector)"),
        ("Output", "theta_per_type, type_proportions"),
        ("Saves", "logs/mmicrl/mmicrl_results.json"),
    ])

    fb.make_node("train_fn", "train(cfg, method, seed, device, ...)", FUNC, [
        ("Setup", "seed, env, agent, logger, dirs"),
        ("Loop", "while global_step < total_steps"),
        ("Per step", "get_actions -> env.step -> store transitions"),
        ("Metrics", "reward, cost, safety_rate, Jain, SAI, peak_MF"),
        ("Update", "agent.update() at episode end"),
        ("Lambda", "EMA cost -> update_lambda (Lagrangian only)"),
        ("Checkpoint", "every N steps + best model"),
    ])

    fb.make_node("cli", "main() CLI entrypoint", FUNC, [
        ("--config", "YAML config path"),
        ("--method", "hcmarl|mappo|ippo|mappo_lag"),
        ("--seed", "int (default 0)"),
        ("--device", "cpu|cuda|auto"),
        ("--mmicrl", "flag: run MMICRL pre-training"),
        ("--ecbf-mode", "on|off (ablation)"),
    ])

    fb.edge("file_hdr", "methods", "", style="bold")
    fb.edge("file_hdr", "create_agent", "", style="bold")
    fb.edge("file_hdr", "mmicrl_pretrain", "", style="bold")
    fb.edge("file_hdr", "train_fn", "", style="bold")
    fb.edge("file_hdr", "cli", "", style="bold")

    fb.edge("methods", "create_agent", "method name")
    fb.edge("create_agent", "train_fn", "agent instance")
    fb.edge("mmicrl_pretrain", "train_fn", "theta_per_type, type_proportions", color="#DC2626")
    fb.edge("cli", "mmicrl_pretrain", "args.mmicrl", style="dashed")
    fb.edge("cli", "train_fn", "cfg, method, seed, device")

    fb.dangling_in("create_agent", "HCMARLAgent", "agents/hcmarl_agent.py")
    fb.dangling_in("create_agent", "MAPPO, IPPO, MAPPOLagrangian", "agents/*.py")
    fb.dangling_in("create_agent", "OmniSafeWrapper, SafePOWrapper", "baselines/*.py")
    fb.dangling_in("train_fn", "WarehousePettingZoo", "envs/pettingzoo_wrapper.py")
    fb.dangling_in("train_fn", "HCMARLLogger", "logger.py")
    fb.dangling_in("mmicrl_pretrain", "DemonstrationCollector, MMICRL", "mmicrl.py")
    fb.dangling_out("train_fn", "checkpoints .pt", "checkpoints/")
    fb.dangling_out("train_fn", "summary.json, config.yaml", "logs/<method>/")

    fb.add_legend_entry("method name", "string key into METHODS registry", "hcmarl")
    fb.add_legend_entry("agent instance", "constructed agent object passed to train()", "HCMARLAgent")
    fb.add_legend_entry("theta_per_type, type_proportions", "MMICRL-learned per-type thresholds and mix", "{'0':{'shoulder':0.70}}")
    fb.add_legend_entry("args.mmicrl", "bool flag: run MMICRL pre-training before train()", "True")
    fb.add_legend_entry("cfg, method, seed, device", "parsed CLI args forwarded to train()", "cfg=TrainingConfig, seed=0")

    fb.set_rank("top", ["methods", "cli"])
    fb.set_rank("mid", ["create_agent", "mmicrl_pretrain"])
    fb.set_rank("bot", ["train_fn"])
    fb.render(OUT)


# =====================================================================
# scripts/evaluate.py
# =====================================================================
def gen_evaluate():
    fb = FlowchartBuilder("scripts/evaluate.py",
        "Evaluation: load checkpoint, run N episodes, compute 9 metrics",
        lines="161",
        imports_desc="argparse, yaml, torch, numpy, train.create_agent")

    fb.make_node("eval_fn", "evaluate(cfg, method, checkpoint, n_episodes, ...)", FUNC, [
        ("Setup", "env + agent from create_agent(), load checkpoint"),
        ("Loop", "n_episodes rollouts, no gradient"),
        ("Per step", "track violations, recovery time, tasks, peak_MF"),
        ("Metrics", "9 HC-MARL metrics (mean +/- std)"),
    ])

    fb.make_node("metrics", "9 HC-MARL Metrics", CONST, [
        ("violation_rate", "violations / (steps * workers * muscles)"),
        ("cumulative_cost", "total violations per episode"),
        ("safety_rate", "fraction of steps with 0 violations"),
        ("tasks_completed", "total non-rest tasks"),
        ("cumulative_reward", "sum of rewards"),
        ("jain_fairness", "Jain index over per-worker tasks"),
        ("peak_fatigue", "max MF across all muscles"),
        ("forced_rest_rate", "forced rests / (steps * workers)"),
        ("constraint_recovery_time", "mean steps to recover from violation"),
    ])

    fb.make_node("eval_cli", "main() CLI", FUNC, [
        ("--checkpoint", "path to .pt file"),
        ("--config", "YAML config"),
        ("--method", "agent type"),
        ("--n-episodes", "100 (default)"),
        ("--output", "results JSON path"),
    ])

    fb.edge("file_hdr", "eval_fn", "", style="bold")
    fb.edge("file_hdr", "eval_cli", "", style="bold")
    fb.edge("eval_cli", "eval_fn", "cfg, method, checkpoint")
    fb.edge("eval_fn", "metrics", "metrics_dict")

    fb.dangling_in("eval_fn", "create_agent()", "scripts/train.py")
    fb.dangling_in("eval_fn", "WarehousePettingZoo", "envs/pettingzoo_wrapper.py")
    fb.dangling_in("eval_fn", "checkpoint .pt", "checkpoints/")
    fb.dangling_in("eval_fn", "HCMARLLogger.METRIC_NAMES", "logger.py")
    fb.dangling_out("eval_fn", "results JSON", "results/<method>_eval.json")

    fb.add_legend_entry("cfg, method, checkpoint", "parsed CLI args forwarded to evaluate()", "cfg=dict, method='hcmarl', checkpoint='ckpt.pt'")
    fb.add_legend_entry("metrics_dict", "dict of 9 metric mean/std values", "{'violation_rate_mean': 0.012, ...}")
    fb.add_legend_entry("create_agent()", "agent factory function imported from train.py", "create_agent('hcmarl', 19, 73, 6, 6, cfg, 'cpu')")
    fb.add_legend_entry("WarehousePettingZoo", "PettingZoo env for evaluation rollouts", "WarehousePettingZoo(n_workers=6)")
    fb.add_legend_entry("checkpoint .pt", "saved model weights loaded into agent", "checkpoints/hcmarl/seed_0/checkpoint_final.pt")
    fb.add_legend_entry("HCMARLLogger.METRIC_NAMES", "list of 9 metric name strings", "['violation_rate', 'safety_rate', ...]")
    fb.add_legend_entry("results JSON", "evaluation results saved to disk", "results/hcmarl_seed0_eval.json")
    fb.render(OUT)


# =====================================================================
# scripts/run_ablations.py
# =====================================================================
def gen_run_ablations():
    fb = FlowchartBuilder("scripts/run_ablations.py",
        "Batch ablation launcher: 5 ablations x 5 seeds = 25 runs",
        lines="53",
        imports_desc="argparse, subprocess")

    fb.make_node("ablations", "ABLATIONS list", CONST, [
        ("no_ecbf", "ECBF safety filter removed"),
        ("no_nswf", "NSWF replaced with round-robin"),
        ("no_mmicrl", "fixed theta_max (no learned types)"),
        ("no_reperfusion", "r=1 always (no reperfusion switch)"),
        ("no_divergent", "constant D_i=kappa"),
    ])

    fb.make_node("launcher", "main() — nested loop", FUNC, [
        ("Outer", "for ablation in ABLATIONS"),
        ("Inner", "for seed in [0,1,2,3,4]"),
        ("Config", "config/ablation_{name}.yaml"),
        ("Cmd", "python scripts/train.py --config ... --method hcmarl --seed N"),
        ("--dry-run", "print commands only"),
    ])

    fb.edge("file_hdr", "ablations", "", style="bold")
    fb.edge("file_hdr", "launcher", "", style="bold")
    fb.edge("ablations", "launcher", "ablation_name")

    fb.dangling_in("launcher", "ablation configs", "config/ablation_*.yaml")
    fb.dangling_out("launcher", "subprocess calls", "scripts/train.py")

    fb.add_legend_entry("ablation_name", "string name of component to remove", "'no_ecbf'")
    fb.add_legend_entry("ablation configs", "per-ablation YAML config files", "config/ablation_no_ecbf.yaml")
    fb.add_legend_entry("subprocess calls", "spawned train.py processes", "python scripts/train.py --config ... --seed 0")
    fb.render(OUT)


# =====================================================================
# scripts/run_baselines.py
# =====================================================================
def gen_run_baselines():
    fb = FlowchartBuilder("scripts/run_baselines.py",
        "Batch baseline launcher: 6 methods x 5 seeds = 30 runs",
        lines="68",
        imports_desc="argparse, subprocess, yaml")

    fb.make_node("baseline_list", "experiment_matrix.yaml -> headline.methods", CONST, [
        ("hcmarl", "config/hcmarl_full_config.yaml"),
        ("mappo", "config/mappo_config.yaml"),
        ("ippo", "config/ippo_config.yaml"),
        ("mappo_lag", "config/mappo_lag_config.yaml"),
    ])

    fb.make_node("bl_launcher", "main() — matrix-driven loop", FUNC, [
        ("Outer", "for method, spec in matrix['headline']['methods']"),
        ("Inner", "for seed in matrix['headline']['seeds']  # 10 seeds"),
        ("Cmd", "python scripts/train.py --config X --method Y --seed N --run-name Y"),
        ("--dry-run", "print commands only"),
    ])

    fb.edge("file_hdr", "baseline_list", "", style="bold")
    fb.edge("file_hdr", "bl_launcher", "", style="bold")
    fb.edge("baseline_list", "bl_launcher", "method, config_path")

    fb.dangling_in("bl_launcher", "per-method configs", "config/*_config.yaml")
    fb.dangling_out("bl_launcher", "subprocess calls", "scripts/train.py")

    fb.add_legend_entry("method, config_path", "baseline method name + its YAML config path", "'mappo', 'config/mappo_config.yaml'")
    fb.add_legend_entry("per-method configs", "per-method YAML config files", "config/mappo_config.yaml")
    fb.add_legend_entry("subprocess calls", "spawned train.py processes", "python scripts/train.py --method mappo --seed 0")
    fb.render(OUT)


# =====================================================================
# scripts/run_scaling.py — REMOVED (2026-04-16 venue-audit verdict:
# scaling study is not in scope). Nothing to diagram here.
# =====================================================================


# =====================================================================
# scripts/gen_directory_structure.py
# =====================================================================
def gen_gen_directory_structure():
    fb = FlowchartBuilder("scripts/gen_directory_structure.py",
        "Generates directory tree diagram (Graphviz)", lines="243",
        imports_desc="graphviz")

    fb.make_node("dir_node_fn", "dir_node(name, label, color, items)", FUNC, [
        ("Input", "node id, folder name, color, file list"),
        ("Output", "HTML-table Graphviz node"),
    ])

    fb.make_node("tree_nodes", "Directory tree nodes (12 dirs)", CONST, [
        ("root", "hcmarl_project/"),
        ("hcmarl", "core framework (10 files)"),
        ("envs/agents/baselines", "sub-packages"),
        ("scripts/config/tests", "infrastructure"),
        ("notebooks/data/checkpoints", "runtime"),
        ("diagrams/REFERENCES/logs/docs", "assets"),
    ])

    fb.make_node("render_dir", "g.render('diagrams/directory_structure')", FUNC, [
        ("Output", "diagrams/directory_structure.png"),
    ])

    fb.edge("file_hdr", "dir_node_fn", "", style="bold")
    fb.edge("file_hdr", "tree_nodes", "", style="bold")
    fb.edge("dir_node_fn", "tree_nodes", "html_table")
    fb.edge("tree_nodes", "render_dir", "graphviz.Digraph")

    fb.dangling_out("render_dir", "PNG", "diagrams/directory_structure.png")

    fb.add_legend_entry("html_table", "HTML-formatted Graphviz node", "<TABLE>...</TABLE>")
    fb.add_legend_entry("graphviz.Digraph", "assembled graph object", "Digraph('dir_structure')")
    fb.add_legend_entry("PNG", "rendered output file", "diagrams/directory_structure.png")
    fb.render(OUT, stem_override="gen_directory_structure")


# =====================================================================
# setup.py
# =====================================================================
def gen_setup():
    fb = FlowchartBuilder("setup.py",
        "Package metadata and dependencies", lines="35",
        imports_desc="setuptools")

    fb.make_node("setup_call", "setup() call", FUNC, [
        ("name", "'hcmarl'"),
        ("version", "'0.1.0'"),
        ("author", "'Aditya Maiti'"),
        ("python_requires", "'>=3.9'"),
    ])

    fb.make_node("deps", "install_requires (core)", CONST, [
        ("numpy", ">=1.24.0"),
        ("scipy", ">=1.10.0"),
        ("cvxpy", ">=1.4.0"),
        ("osqp", ">=0.6.3"),
        ("pyyaml", ">=6.0"),
    ])

    fb.make_node("extras", "extras_require", CONST, [
        ("dev", "pytest>=7.4.0, pytest-cov>=4.1.0"),
        ("rl", "torch>=2.0.0, gymnasium, pettingzoo, wandb"),
    ])

    fb.edge("file_hdr", "setup_call", "", style="bold")
    fb.edge("file_hdr", "deps", "", style="bold")
    fb.edge("file_hdr", "extras", "", style="bold")
    fb.edge("deps", "setup_call", "install_requires")
    fb.edge("extras", "setup_call", "extras_require")

    fb.dangling_in("setup_call", "find_packages()", "hcmarl/")
    fb.dangling_out("setup_call", "pip install -e .", "venv/")

    fb.add_legend_entry("install_requires", "core dependency list", "['numpy>=1.24.0', ...]")
    fb.add_legend_entry("extras_require", "optional dependency groups", "{'dev': [...], 'rl': [...]}")
    fb.add_legend_entry("find_packages()", "auto-discovered packages", "['hcmarl', 'hcmarl.envs', ...]")
    fb.add_legend_entry("pip install -e .", "editable install output", "venv/lib/hcmarl.egg-link")
    fb.render(OUT)


if __name__ == "__main__":
    generators = [
        gen_train, gen_evaluate, gen_run_ablations, gen_run_baselines,
        gen_gen_directory_structure, gen_setup,
    ]
    for fn in generators:
        try:
            fn()
        except Exception as e:
            print(f"FAIL: {fn.__name__}: {e}")
