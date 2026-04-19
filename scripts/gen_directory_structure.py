"""Generate directory structure diagram using Graphviz."""
import graphviz
import os
os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + ";" + os.environ.get("PATH", "")

g = graphviz.Digraph("directory_structure", format="png")
g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
       nodesep="0.4", ranksep="0.6", margin="0.3", size="24,40")
g.attr("node", fontname="Consolas", fontsize="10", shape="none")
g.attr("edge", arrowsize="0.5", color="#64748B")


def dir_node(name, label, color, items):
    rows = '<TR><TD COLSPAN="2" BGCOLOR="{}" ALIGN="LEFT"><FONT COLOR="white"><B>  {}/  </B></FONT></TD></TR>'.format(color, label)
    for fname, desc in items:
        desc_text = desc if desc else " "
        rows += '<TR><TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="9">  {}  </FONT></TD>'.format(fname)
        rows += '<TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="8" COLOR="#6B7280">{}</FONT></TD></TR>'.format(desc_text)
    html = '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2" COLOR="#CBD5E1">{}</TABLE>'.format(rows)
    g.node(name, "<" + html + ">")


# ═══════════════════════════════════════════════════════════════
# ROOT
# ═══════════════════════════════════════════════════════════════
dir_node("root", "hcmarl_project", "#0F172A", [
    ("MATHEMATICAL MODELLING.pdf", "360KB  15pp  Sections 1-8  28 refs"),
    ("TIMELINE.txt", "26-day plan (Phases A/B/C/D)"),
    ("README.md", ""),
    ("requirements.txt", ""),
    ("setup.py", ""),
    (".gitignore", ""),
])

# ═══════════════════════════════════════════════════════════════
# hcmarl/ — CORE FRAMEWORK
# ═══════════════════════════════════════════════════════════════
dir_node("hcmarl", "hcmarl", "#1E3A5F", [
    ("__init__.py", "13 lines"),
    ("three_cc_r.py", "386 ln | 3CC-r ODE  Eqs 1-8  Table 1"),
    ("ecbf_filter.py", "428 ln | ECBF CBF-QP  Eqs 12-30"),
    ("nswf_allocator.py", "360 ln | NSWF allocator  Eqs 31-33"),
    ("mmicrl.py", "1251 ln | CFDE + MMICRL  Eqs 9-11"),
    ("pipeline.py", "411 ln | 7-step pipeline  S7.3"),
    ("real_data_calibration.py", "1101 ln | Path G  WSD4FEDSRM"),
    ("warehouse_env.py", "451 ln | Single+Multi Gymnasium env"),
    ("logger.py", "91 ln | CSV + W&amp;B logging"),
    ("utils.py", "165 ln"),
])

dir_node("envs", "hcmarl/envs", "#1E3A5F", [
    ("__init__.py", "18 ln  lazy imports"),
    ("pettingzoo_wrapper.py", "163 ln | N-agent parallel env"),
    ("reward_functions.py", "93 ln | Eq 32-33 canonical reward"),
    ("task_profiles.py", "66 ln | Eq 34 task demands"),
])

dir_node("agents", "hcmarl/agents", "#1E3A5F", [
    ("__init__.py", "6 ln"),
    ("mappo.py", "277 ln | MAPPO  per-agent GAE"),
    ("mappo_lag.py", "284 ln | MAPPO-Lag  cost critic"),
    ("ippo.py", "120 ln | IPPO  independent PPO"),
    ("hcmarl_agent.py", "34 ln | HC-MARL wrapper"),
    ("networks.py", "71 ln | Actor/Critic/CostCritic"),
])

dir_node("baselines", "hcmarl/baselines", "#1E3A5F", [
    ("__init__.py", "11 ln"),
    ("_legacy.py", "323 ln | 10 heuristic baselines"),
    ("omnisafe_wrapper.py", "241 ln | PPO-Lag  CPO"),
    ("safepo_wrapper.py", "115 ln | MACPO fallback"),
])

# ═══════════════════════════════════════════════════════════════
# scripts/
# ═══════════════════════════════════════════════════════════════
dir_node("scripts", "scripts", "#6B2142", [
    ("train.py", "990 ln | Main training loop  4 methods + budget kill-switch"),
    ("evaluate.py", "161 ln | Checkpoint eval  9 metrics"),
    ("run_baselines.py", "68 ln | 4 methods x 10 seeds (matrix-driven)"),
    ("run_ablations.py", "53 ln | 5-rung build-up x 10 seeds"),
])

# ═══════════════════════════════════════════════════════════════
# config/
# ═══════════════════════════════════════════════════════════════
dir_node("config", "config", "#6B5B1E", [
    ("hcmarl_full_config.yaml", "70 ln"),
    ("default_config.yaml", "83 ln"),
    ("dry_run_50k.yaml", "49 ln"),
    ("task_profiles.yaml", "60 ln"),
    ("mappo_config.yaml", "23 ln"),
    ("ippo_config.yaml", "23 ln"),
    ("mappo_lag_config.yaml", "25 ln"),
    ("ablation_no_ecbf.yaml", "73 ln"),
    ("ablation_no_nswf.yaml", "73 ln"),
    ("ablation_no_mmicrl.yaml", "73 ln"),
    ("ablation_no_reperfusion.yaml", "73 ln"),
    ("ablation_no_divergent.yaml", "73 ln"),
    ("experiment_matrix.yaml", "single source of truth"),
])

# ═══════════════════════════════════════════════════════════════
# tests/
# ═══════════════════════════════════════════════════════════════
dir_node("tests", "tests", "#1A4A2A", [
    ("__init__.py", "1 ln"),
    ("test_real_data_calibration.py", "541 ln  53 tests"),
    ("test_three_cc_r.py", "462 ln  51 tests"),
    ("test_ecbf.py", "294 ln  33 tests"),
    ("test_nswf.py", "275 ln  24 tests"),
    ("test_pipeline.py", "269 ln  17 tests"),
    ("test_phase3.py", "271 ln  15 tests"),
    ("test_phase2.py", "186 ln  12 tests"),
    ("test_all_methods.py", "52 ln  6 tests"),
    ("test_pettingzoo.py", "36 ln  4 tests"),
    ("test_warehouse_env.py", "33 ln  4 tests"),
    ("test_hcmarl_agent.py", "35 ln  3 tests"),
    ("test_env_integration.py", "27 ln  1 test"),
])

# ═══════════════════════════════════════════════════════════════
# notebooks/
# ═══════════════════════════════════════════════════════════════
dir_node("notebooks", "notebooks", "#4A2A5C", [
    ("train_hcmarl.ipynb", "HC-MARL 5 seeds x 5M steps"),
    ("train_baselines.ipynb", "6 baselines x 5 seeds"),
    ("train_ablations.ipynb", "5 ablations x 5 seeds"),
    ("train_scaling.ipynb", "N=3,4,6,8,12 x 5 seeds"),
])

# ═══════════════════════════════════════════════════════════════
# data/
# ═══════════════════════════════════════════════════════════════
dir_node("data", "data", "#6B3A1E", [
    ("wsd4fedsrm/WSD4FEDSRM/", "~1.6 GB extracted  34 subjects  Zenodo 8415066"),
])

dir_node("wsd_sub", "data/wsd4fedsrm/WSD4FEDSRM", "#6B3A1E", [
    ("Borg data/", "borg_data.csv  endurance + RPE every 10s"),
    ("MVIC force data/", "MVIC_force_data.csv"),
    ("Demographic and antropometric data/", "6 CSVs  demographics+anthropometrics"),
    ("KSS data/", "KSS_data.csv  Karolinska Sleepiness Scale"),
    ("EMG, IMU, and PPG data/", "8,364 CSVs  10 task dirs  34 subjects each"),
])

# ═══════════════════════════════════════════════════════════════
# checkpoints/
# ═══════════════════════════════════════════════════════════════
dir_node("checkpoints", "checkpoints", "#3A3A3A", [
    ("hcmarl/seed_0/", "4 files x 130KB  (50K dry run)"),
    ("mappo/seed_0/", "4 files x 130KB"),
    ("mappo_lag/seed_0/", "4 files x 234KB"),
    ("ippo/seed_0/", "4 files x 193KB"),
])

# ═══════════════════════════════════════════════════════════════
# diagrams/
# ═══════════════════════════════════════════════════════════════
dir_node("diagrams", "diagrams", "#3A3A3A", [
    ("01-37_*.png", "37 onboarding diagrams  300 DPI"),
    ("understanding/", "6 PNGs + 2 PDFs"),
])

# ═══════════════════════════════════════════════════════════════
# REFERENCES/
# ═══════════════════════════════════════════════════════════════
dir_node("refs", "REFERENCES", "#3A3A3A", [
    ("1.pdf through 28.pdf", "26 cited papers  52 MB total"),
    ("(25.pdf  26.pdf missing)", "Khalil 2002  Rohmert 1960"),
])

# ═══════════════════════════════════════════════════════════════
# logs/
# ═══════════════════════════════════════════════════════════════
dir_node("logs_dir", "logs", "#3A3A3A", [
    ("project_log.md", "~780 lines  session-by-session"),
    ("timeline_tracker.md", "~135 lines  day-by-day"),
])

# ═══════════════════════════════════════════════════════════════
# docs/
# ═══════════════════════════════════════════════════════════════
dir_node("docs", "docs", "#3A3A3A", [
    ("ARCHITECTURE_FOR_ADVISOR.md", "14-section advisor document"),
])

# ═══════════════════════════════════════════════════════════════
# RANK GROUPINGS — force vertical layout
# ═══════════════════════════════════════════════════════════════

# Row 1: root
# Row 2: hcmarl, scripts, config, tests
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("hcmarl")
    s.node("scripts")
    s.node("config")
    s.node("tests")

# Row 3: subpackages + notebooks + data
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("envs")
    s.node("agents")
    s.node("baselines")
    s.node("notebooks")
    s.node("data")

# Row 4: remaining
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("wsd_sub")
    s.node("checkpoints")
    s.node("diagrams")
    s.node("refs")
    s.node("logs_dir")
    s.node("docs")

# ═══════════════════════════════════════════════════════════════
# EDGES
# ═══════════════════════════════════════════════════════════════
for child in ["hcmarl", "scripts", "config", "tests",
              "notebooks", "data",
              "checkpoints", "diagrams", "refs", "logs_dir", "docs"]:
    g.edge("root", child, style="bold")

g.edge("hcmarl", "envs", style="bold")
g.edge("hcmarl", "agents", style="bold")
g.edge("hcmarl", "baselines", style="bold")
g.edge("data", "wsd_sub", style="bold")

# Render
g.render("diagrams/directory_structure", cleanup=True)
print("SUCCESS: diagrams/directory_structure.png")
