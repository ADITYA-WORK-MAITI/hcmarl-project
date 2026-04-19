"""
Generate code flowcharts for notebooks/*.ipynb (4) and config/*.yaml (20).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from flowchart_framework import FlowchartBuilder, HDR, CLS, FUNC, CONST, PROP, UTIL

OUT = "diagrams/code_flowcharts"
os.makedirs(OUT, exist_ok=True)


# =====================================================================
# notebooks/train_hcmarl.ipynb
# =====================================================================
def gen_nb_train_hcmarl():
    fb = FlowchartBuilder("notebooks/train_hcmarl.ipynb",
        "Colab: HC-MARL full training (5 seeds x 5M steps)",
        lines="12 cells",
        imports_desc="torch, wandb, hcmarl")

    fb.make_node("setup", "Cells 1-4: Setup", CONST, [
        ("Install", "pip install torch, gymnasium, pettingzoo, cvxpy, wandb"),
        ("Clone", "git clone hcmarl-project"),
        ("Drive", "mount Google Drive for checkpoints"),
        ("W&B", "wandb.login() (optional)"),
    ])
    fb.make_node("gpu", "Cell 5: GPU Check", FUNC, [
        ("Assert", "torch.cuda.is_available()"),
        ("Print", "GPU name + VRAM"),
    ])
    fb.make_node("train_loop", "Cell 6: Training Loop", FUNC, [
        ("Seeds", "[0, 1, 2, 3, 4]"),
        ("Command", "python scripts/train.py --config hcmarl_full_config.yaml --method hcmarl --seed N --device cuda"),
    ])
    fb.make_node("eval_cell", "Cell 7: Evaluation", FUNC, [
        ("Glob", "checkpoints/*/seed_*/checkpoint_final.pt"),
        ("Command", "python scripts/evaluate.py for each"),
    ])
    fb.make_node("save_drive", "Cell 8: Save to Drive", FUNC, [
        ("Copies", "checkpoints/, logs/, results/ -> Drive"),
    ])

    fb.edge("file_hdr", "setup", "", style="bold")
    fb.edge("setup", "gpu", "torch, pip packages")
    fb.edge("gpu", "train_loop", "device='cuda'")
    fb.edge("train_loop", "eval_cell", "checkpoint_final.pt")
    fb.edge("eval_cell", "save_drive", "eval_results.json")

    fb.dangling_in("train_loop", "hcmarl_full_config.yaml", "config/")
    fb.dangling_in("train_loop", "scripts/train.py", "scripts/")
    fb.dangling_out("save_drive", "checkpoints + logs", "Google Drive")
    fb.dangling_out("eval_cell", "eval JSONs", "results/")

    fb.add_legend_entry("torch, pip packages", "installed packages passed to GPU check cell", "torch 2.x, gymnasium, cvxpy")
    fb.add_legend_entry("device='cuda'", "verified CUDA device string passed to train loop", "device='cuda' (T4 GPU)")
    fb.add_legend_entry("checkpoint_final.pt", "saved model checkpoint file produced by training", "checkpoints/hcmarl/seed_0/checkpoint_final.pt")
    fb.add_legend_entry("eval_results.json", "per-seed evaluation metrics file", "results/hcmarl_seed0_eval.json")
    fb.render(OUT, stem_override="nb_train_hcmarl")


# =====================================================================
# notebooks/train_baselines.ipynb
# =====================================================================
def gen_nb_train_baselines():
    fb = FlowchartBuilder("notebooks/train_baselines.ipynb",
        "Colab: 4 headline methods x 10 seeds = 40 runs (matrix-driven)",
        lines="12 cells",
        imports_desc="torch, wandb, hcmarl")

    fb.make_node("bl_setup", "Cells 1-5: Setup (same template)", CONST, [
        ("Install + Clone + Drive + W&B + GPU", "identical to train_hcmarl"),
    ])
    fb.make_node("bl_train", "Cell 6: Batch Training", FUNC, [
        ("Command", "python scripts/run_baselines.py --device cuda"),
        ("Methods", "hcmarl, mappo, ippo, mappo_lag (read from matrix)"),
        ("Source", "config/experiment_matrix.yaml headline section"),
    ])
    fb.make_node("bl_eval", "Cell 7-8: Eval + Save", FUNC, [
        ("Evaluate", "all checkpoints found via glob"),
        ("Save", "copy to Google Drive"),
    ])

    fb.edge("file_hdr", "bl_setup", "", style="bold")
    fb.edge("bl_setup", "bl_train", "pip packages, repo")
    fb.edge("bl_train", "bl_eval", "checkpoint_final.pt (x40)")

    fb.dangling_in("bl_train", "experiment_matrix.yaml", "config/")
    fb.dangling_in("bl_train", "4 per-method configs", "config/*_config.yaml")
    fb.dangling_in("bl_train", "scripts/run_baselines.py", "scripts/")
    fb.dangling_out("bl_eval", "40 eval JSONs", "results/ + Drive")

    fb.add_legend_entry("pip packages, repo", "installed packages and cloned repo passed to train cell", "torch, gymnasium, hcmarl/")
    fb.add_legend_entry("checkpoint_final.pt (x40)", "40 final checkpoint files (4 methods x 10 seeds)", "checkpoints/mappo/seed_0/checkpoint_final.pt")
    fb.render(OUT, stem_override="nb_train_baselines")


# =====================================================================
# notebooks/train_ablations.ipynb
# =====================================================================
def gen_nb_train_ablations():
    fb = FlowchartBuilder("notebooks/train_ablations.ipynb",
        "Colab: 5 ablations x 5 seeds = 25 runs",
        lines="12 cells",
        imports_desc="torch, wandb, hcmarl")

    fb.make_node("ab_setup", "Cells 1-5: Setup (same template)", CONST, [
        ("Install + Clone + Drive + W&B + GPU", "identical to train_hcmarl"),
    ])
    fb.make_node("ab_train", "Cell 6: Batch Ablation Training", FUNC, [
        ("Command", "python scripts/run_ablations.py --device cuda"),
        ("Ablations", "no_ecbf, no_nswf, no_mmicrl, no_reperfusion, no_divergent"),
    ])
    fb.make_node("ab_eval", "Cell 7-8: Eval + Save", FUNC, [
        ("Evaluate", "all ablation checkpoints"),
        ("Save", "copy to Google Drive"),
    ])

    fb.edge("file_hdr", "ab_setup", "", style="bold")
    fb.edge("ab_setup", "ab_train", "pip packages, repo")
    fb.edge("ab_train", "ab_eval", "checkpoint_final.pt (x25)")

    fb.dangling_in("ab_train", "5 ablation configs", "config/ablation_*.yaml")
    fb.dangling_in("ab_train", "scripts/run_ablations.py", "scripts/")
    fb.dangling_out("ab_eval", "25 eval JSONs", "results/ + Drive")

    fb.add_legend_entry("pip packages, repo", "installed packages and cloned repo passed to train cell", "torch, gymnasium, hcmarl/")
    fb.add_legend_entry("checkpoint_final.pt (x25)", "25 final checkpoint files (5 ablations x 5 seeds)", "checkpoints/no_ecbf/seed_0/checkpoint_final.pt")
    fb.render(OUT, stem_override="nb_train_ablations")


# =====================================================================
# notebooks/train_scaling.ipynb — REMOVED (2026-04-16 venue-audit verdict:
# scaling study is not in scope). Nothing to diagram here.
# =====================================================================


# =====================================================================
# config/hcmarl_full_config.yaml
# =====================================================================
def gen_cfg_hcmarl_full():
    fb = FlowchartBuilder("config/hcmarl_full_config.yaml",
        "Full HC-MARL config: 5M steps, N=6, all components",
        lines="70")

    fb.make_node("env_cfg", "environment:", CONST, [
        ("n_workers", "6"),
        ("max_steps", "480 (8-hour shift, 1-min resolution)"),
        ("dt", "1.0 min"),
        ("kappa", "1.0 (D_i scaling)"),
        ("muscle_groups", "6: shoulder, ankle, knee, elbow, trunk, grip"),
        ("theta_max", "per-muscle: 0.70, 0.80, 0.60, 0.45, 0.65, 0.35"),
        ("tasks", "6: heavy_lift, light_sort, carry, overhead_reach, push_cart, rest"),
    ])
    fb.make_node("train_cfg", "training:", CONST, [
        ("total_steps", "5,000,000"),
        ("eval_interval", "50,000"),
        ("checkpoint_interval", "500,000"),
        ("seeds", "[0, 1, 2, 3, 4]"),
    ])
    fb.make_node("algo_cfg", "algorithm:", CONST, [
        ("lr_actor", "3e-4"), ("lr_critic", "1e-3"),
        ("gamma", "0.99"), ("gae_lambda", "0.95"),
        ("clip_eps", "0.2"), ("entropy_coeff", "0.01"),
        ("n_epochs", "10"), ("batch_size", "256"),
    ])
    fb.make_node("ecbf_cfg", "ecbf:", CONST, [
        ("alpha1/2/3", "0.5, 0.5, 0.5"), ("kp", "10.0"),
    ])
    fb.make_node("log_cfg", "logging:", CONST, [
        ("use_wandb", "true"), ("project_name", "hcmarl-neurips2026"),
    ])

    fb.edge("file_hdr", "env_cfg", "", style="bold")
    fb.edge("file_hdr", "train_cfg", "", style="bold")
    fb.edge("file_hdr", "algo_cfg", "", style="bold")
    fb.edge("file_hdr", "ecbf_cfg", "", style="bold")
    fb.edge("file_hdr", "log_cfg", "", style="bold")

    fb.dangling_out("env_cfg", "env params", "scripts/train.py -> WarehousePettingZoo")
    fb.dangling_out("algo_cfg", "PPO hyperparams", "scripts/train.py -> create_agent()")
    fb.dangling_out("ecbf_cfg", "ECBF gains", "scripts/train.py -> HCMARLAgent")

    fb.render(OUT, stem_override="cfg_hcmarl_full")


# =====================================================================
# config/default_config.yaml
# =====================================================================
def gen_cfg_default():
    fb = FlowchartBuilder("config/default_config.yaml",
        "Default config: 4 workers, 3 muscles, per-muscle ECBF",
        lines="83")

    fb.make_node("def_env", "environment:", CONST, [
        ("num_workers", "4"),
        ("muscle_names", "[shoulder, elbow, grip]"),
        ("dt", "1.0"), ("kp", "10.0"),
        ("tasks", "4: box_lift_overhead, carry_heavy, sorting_light, packing"),
    ])
    fb.make_node("def_ecbf", "ecbf: (per-muscle config)", CONST, [
        ("shoulder", "theta_max=0.70, alpha1=0.05, alpha2=0.05, alpha3=0.10"),
        ("elbow", "theta_max=0.45"),
        ("grip", "theta_max=0.35"),
    ])
    fb.make_node("def_nswf", "nswf:", CONST, [
        ("kappa", "1.0"), ("epsilon", "0.001"),
    ])

    fb.edge("file_hdr", "def_env", "", style="bold")
    fb.edge("file_hdr", "def_ecbf", "", style="bold")
    fb.edge("file_hdr", "def_nswf", "", style="bold")

    fb.dangling_out("def_env", "env params", "pipeline.py")
    fb.dangling_out("def_ecbf", "per-muscle ECBF params", "ecbf_filter.py")
    fb.dangling_out("def_nswf", "NSWF params", "nswf_allocator.py")

    fb.render(OUT, stem_override="cfg_default")


# =====================================================================
# config/dry_run_50k.yaml
# =====================================================================
def gen_cfg_dry_run():
    fb = FlowchartBuilder("config/dry_run_50k.yaml",
        "50K-step verification config (not for experiments)",
        lines="49")

    fb.make_node("dry_env", "environment:", CONST, [
        ("n_workers", "4"), ("max_steps", "60"),
        ("theta_max", "same as full config"),
        ("tasks", "6 (same as full config)"),
    ])
    fb.make_node("dry_train", "training:", CONST, [
        ("total_steps", "50,000 (fast verification)"),
        ("eval_interval", "10,000"),
        ("n_eval_episodes", "3"),
    ])
    fb.make_node("dry_algo", "algorithm:", CONST, [
        ("n_epochs", "5 (reduced from 10)"),
        ("batch_size", "128 (reduced from 256)"),
    ])

    fb.edge("file_hdr", "dry_env", "", style="bold")
    fb.edge("file_hdr", "dry_train", "", style="bold")
    fb.edge("file_hdr", "dry_algo", "", style="bold")

    fb.dangling_out("dry_train", "quick verification", "scripts/train.py")

    fb.render(OUT, stem_override="cfg_dry_run_50k")


# =====================================================================
# config/task_profiles.yaml
# =====================================================================
def gen_cfg_task_profiles():
    fb = FlowchartBuilder("config/task_profiles.yaml",
        "Task demand profiles: %MVC per muscle per task (Eq 34)",
        lines="61")

    fb.make_node("profiles", "task_profiles:", CONST, [
        ("heavy_lift", "shoulder:0.45 knee:0.40 trunk:0.50 grip:0.55"),
        ("light_sort", "shoulder:0.10 elbow:0.15 grip:0.20"),
        ("carry", "shoulder:0.25 trunk:0.30 grip:0.45"),
        ("overhead_reach", "shoulder:0.55 elbow:0.35 grip:0.30"),
        ("push_cart", "shoulder:0.20 trunk:0.25 grip:0.40"),
        ("rest", "all 0.00"),
    ])
    fb.make_node("sources", "Literature Sources", CONST, [
        ("[1] Granata & Marras 1995", "trunk EMG during lifting"),
        ("[2] de Looze et al. 2000", "shoulder loads during pushing"),
        ("[3] Hoozemans et al. 2004", "shoulder/grip cart pushing"),
        ("[4] Snook & Ciriello 1991", "Liberty Mutual MMH tables"),
        ("[5] Nordander et al. 2000", "repetitive work"),
        ("[6] Anton et al. 2001", "overhead work shoulder loads"),
        ("[7] McGill et al. 2013", "trunk stabilisation demands"),
    ])

    fb.edge("file_hdr", "profiles", "", style="bold")
    fb.edge("file_hdr", "sources", "[1]-[7]", style="dashed")

    fb.dangling_out("profiles", "T_L,g demands", "task_profiles.py, pettingzoo_wrapper.py")

    fb.add_legend_entry("[1]-[7]", "literature citation keys linking file header to sources node", "[1] Granata & Marras 1995 ... [7] McGill et al. 2013")
    fb.render(OUT, stem_override="cfg_task_profiles")


# =====================================================================
# Per-method configs (3): mappo, ippo, mappo_lag
# (ppo_lag / cpo / macpo configs were dropped in the Round 4 fake-baselines
# audit — those methods had no in-repo implementation.)
# =====================================================================
def gen_cfg_per_method():
    methods = [
        ("mappo", "MAPPO baseline config", "mappo_config"),
        ("ippo", "IPPO baseline config", "ippo_config"),
        ("mappo_lag", "MAPPO-Lagrangian config (+ lambda_lr, cost_limit)", "mappo_lag_config"),
    ]
    for method_name, desc, stem in methods:
        fb = FlowchartBuilder(f"config/{stem}.yaml", desc, lines="~23")

        fb.make_node("m_train", "training:", CONST, [
            ("total_steps", "5,000,000"),
            ("seeds", "[0, 1, 2, 3, 4]"),
        ])
        fb.make_node("m_algo", "algorithm:", CONST, [
            ("lr_actor", "3e-4"), ("lr_critic", "1e-3"),
            ("gamma", "0.99"), ("clip_eps", "0.2"),
            ("n_epochs", "10"), ("batch_size", "256"),
        ])

        fb.edge("file_hdr", "m_train", "", style="bold")
        fb.edge("file_hdr", "m_algo", "", style="bold")

        fb.dangling_out("m_algo", f"hyperparams for {method_name}", "scripts/train.py")

        fb.render(OUT, stem_override=f"cfg_{stem}")


# =====================================================================
# Ablation configs (5): no_ecbf, no_nswf, no_mmicrl, no_reperfusion, no_divergent
# =====================================================================
def gen_cfg_ablations():
    ablations = [
        ("no_ecbf", "ECBF disabled (ecbf.enabled: false)"),
        ("no_nswf", "NSWF replaced with round-robin"),
        ("no_mmicrl", "Fixed theta_max (no learned types)"),
        ("no_reperfusion", "r=1 always (no reperfusion switch)"),
        ("no_divergent", "Constant D_i=kappa (no fatigue dependence)"),
    ]
    for abl_name, desc in ablations:
        fb = FlowchartBuilder(f"config/ablation_{abl_name}.yaml",
            f"Ablation: {desc}", lines="~73")

        fb.make_node("abl_base", "Base config (same as hcmarl_full)", CONST, [
            ("environment", "N=6, 480 steps, 6 muscles, 6 tasks"),
            ("training", "5M steps, 5 seeds"),
            ("algorithm", "same PPO hyperparams"),
        ])
        fb.make_node("abl_change", f"Ablation: {abl_name}", FUNC, [
            ("What changes", desc),
            ("ablation key", f"'{abl_name}'"),
        ])

        fb.edge("file_hdr", "abl_base", "", style="bold")
        fb.edge("file_hdr", "abl_change", "", style="bold")
        fb.edge("abl_base", "abl_change", "ablated_param", color="#DC2626")

        fb.dangling_out("abl_change", "ablated config", "scripts/run_ablations.py -> train.py")

        fb.add_legend_entry("ablated_param", f"override key disabling {abl_name} component", f"{desc}")
        fb.render(OUT, stem_override=f"cfg_ablation_{abl_name}")


# =====================================================================
# Scaling configs — REMOVED (2026-04-16 venue-audit verdict: no scaling
# study). The only allocator-N sweep that remains lives in
# tests/test_post_scaling_drop.py (pure-math Hungarian sanity).
# =====================================================================


if __name__ == "__main__":
    generators = [
        gen_nb_train_hcmarl, gen_nb_train_baselines,
        gen_nb_train_ablations,
        gen_cfg_hcmarl_full, gen_cfg_default, gen_cfg_dry_run,
        gen_cfg_task_profiles, gen_cfg_per_method, gen_cfg_ablations,
    ]
    for fn in generators:
        try:
            fn()
        except Exception as e:
            print(f"FAIL: {fn.__name__}: {e}")
