"""Generate flowchart diagram for hcmarl/pipeline.py"""
import graphviz
import os
os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + ";" + os.environ.get("PATH", "")

g = graphviz.Digraph("pipeline", format="png")
g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
       nodesep="0.5", ranksep="0.7", margin="0.3", size="34,50")
g.attr("node", fontname="Consolas", fontsize="10", shape="none")
g.attr("edge", fontname="Consolas", fontsize="8", color="#475569")

HDR  = "#0F172A"
CLS  = "#1E3A5F"
FUNC = "#2D4A22"
STEP = "#7C2D12"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mk(name, title, color, rows):
    hdr = '<TR><TD COLSPAN="2" BGCOLOR="{}" ALIGN="LEFT"><FONT COLOR="white"><B>  {}  </B></FONT></TD></TR>'.format(color, esc(title))
    body = ""
    for l, r in rows:
        rv = r if r else " "
        body += '<TR><TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="9">  {}  </FONT></TD>'.format(esc(l))
        body += '<TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="8" COLOR="#6B7280">  {}  </FONT></TD></TR>'.format(esc(rv))
    html = '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" COLOR="#CBD5E1">{}{}</TABLE>'.format(hdr, body)
    g.node(name, "<" + html + ">")


# ── FILE HEADER ──
mk("file_hdr", "hcmarl/pipeline.py", HDR, [
    ("Purpose", "End-to-end HC-MARL control loop (7 steps)"),
    ("Section", "Math PDF Section 7.3"),
    ("Lines", "411"),
    ("Imports", "ecbf_filter, nswf_allocator, three_cc_r, utils"),
])

# ── TaskProfile ──
mk("TaskProfile", "@dataclass TaskProfile (Def 7.1)", CLS, [
    ("task_id: int", "1-indexed for productive tasks"),
    ("name: str", "e.g. 'heavy_lift'"),
    ("demands: dict[str,float]", "muscle_name -> TL in [0,1]"),
    ("get_load(muscle_name)", "returns TL or 0.0"),
])

# ── WorkerState ──
mk("WorkerState", "@dataclass WorkerState", CLS, [
    ("worker_id: int", ""),
    ("muscle_states: dict", "str -> ThreeCCrState per muscle"),
    ("current_task: int|None", "0 or None = resting"),
    ("fresh(worker_id, muscles)", "@classmethod: fully rested"),
    ("max_fatigue()", "max MF across muscles"),
    ("fatigue_for_allocation()", "= max_fatigue()"),
])

# ── HCMARLPipeline.__init__ ──
mk("Pipeline_init", "class HCMARLPipeline.__init__", CLS, [
    ("num_workers: int", "N"),
    ("muscle_names: list[str]", "e.g. ['shoulder','elbow','grip']"),
    ("task_profiles: list[TaskProfile]", "productive tasks"),
    ("ecbf_params_per_muscle", "dict[str, ECBFParams]"),
    ("nswf_params", "NSWFParams (optional)"),
    ("kp: float", "10.0 (baseline neural drive gain)"),
    ("dt: float", "1.0 min (integration step)"),
    ("Builds", "ThreeCCr models, ECBFFilters, NSWFAllocator"),
    ("Inits", "WorkerState.fresh() for each worker"),
])

# ── 7-STEP LOOP ──
mk("step1", "Step 1: Observe States", STEP, [
    ("_observe_states()", "returns list[WorkerState]"),
])

mk("step2", "Step 2: NSWF Task Allocation", STEP, [
    ("_allocate_tasks(utility_matrix)", ""),
    ("Computes", "fatigue_for_allocation() per worker"),
    ("Calls", "NSWFAllocator.allocate(U, MF)"),
    ("Returns", "AllocationResult"),
])

mk("steps3_6", "Steps 3-6: Per-Worker Update", STEP, [
    ("_update_worker(worker, task_id)", ""),
    ("Step 3", "Load translation: TaskProfile.get_load()"),
    ("Step 4", "Neural drive: baseline_neural_drive(TL, MA)"),
    ("Step 5", "Safety filter: ECBFFilter.filter(state, C_nom, TL)"),
    ("Step 6", "State update: ThreeCCr.step_euler(state, C*, TL, dt)"),
    ("Returns", "diagnostics dict per muscle"),
])

mk("step7", "Step 7: Advance Time", STEP, [
    ("time += dt", ""),
    ("step_count += 1", ""),
    ("history.append(result)", "full step diagnostics"),
])

# ── step() main method ──
mk("step_main", "step(utility_matrix) -> dict", FUNC, [
    ("Input", "utility_matrix: ndarray(N,M) or None"),
    ("Default", "ones(N,M) if None"),
    ("Runs", "Steps 1 through 7 sequentially"),
    ("Returns", "dict: step, time, allocation, workers"),
])

# ── from_config ──
mk("from_config", "from_config(config_path) @classmethod", FUNC, [
    ("Input", "YAML config path"),
    ("Loads", "num_workers, muscle_names, dt, kp"),
    ("Builds", "TaskProfile list from config"),
    ("Builds", "ECBFParams per muscle (default theta 1.1x threshold)"),
    ("Builds", "NSWFParams"),
    ("Returns", "HCMARLPipeline instance"),
])

# ── summary ──
mk("summary", "summary() -> str", FUNC, [
    ("Returns", "readable status: step, time, per-worker MF"),
])

# ── External nodes ──
for ext in ["three_cc_r", "ecbf_filter", "nswf_allocator", "utils",
            "test_pipeline", "train_py"]:
    g.node(ext, ext.replace("_", " "), shape="plaintext",
           fontname="Consolas", fontsize="8", fontcolor="#94A3B8")

# ── EDGES ──
g.edge("file_hdr", "TaskProfile", label="  defines  ", style="bold")
g.edge("file_hdr", "WorkerState", label="  defines  ", style="bold")
g.edge("file_hdr", "Pipeline_init", label="  defines  ", style="bold")

g.edge("Pipeline_init", "step_main", label="  self.*  ", style="dashed")
g.edge("Pipeline_init", "from_config", label="  @classmethod  ")
g.edge("Pipeline_init", "summary", style="dashed")

# 7-step loop chain
g.edge("step_main", "step1", label="  1  ", color="#DC2626", style="bold")
g.edge("step1", "step2", label="  2  worker states  ", color="#DC2626", style="bold")
g.edge("step2", "steps3_6", label="  3-6  assignments  ", color="#DC2626", style="bold")
g.edge("steps3_6", "step7", label="  7  diagnostics  ", color="#DC2626", style="bold")

g.edge("TaskProfile", "steps3_6", label="  get_load()  ")
g.edge("WorkerState", "step1", label="  self.workers  ")
g.edge("WorkerState", "step2", label="  fatigue_for_allocation()  ")

# Cross-file: inbound
g.edge("three_cc_r", "Pipeline_init",
       label="  ThreeCCr, ThreeCCrState, MuscleParams, get_muscle  ",
       style="dotted", color="#3B82F6")
g.edge("ecbf_filter", "Pipeline_init",
       label="  ECBFFilter, ECBFParams  ",
       style="dotted", color="#3B82F6")
g.edge("nswf_allocator", "Pipeline_init",
       label="  NSWFAllocator, NSWFParams, AllocationResult  ",
       style="dotted", color="#3B82F6")
g.edge("utils", "Pipeline_init",
       label="  load_yaml, get_logger  ",
       style="dotted", color="#3B82F6")

# Cross-file: outbound
g.edge("Pipeline_init", "test_pipeline",
       label="  HCMARLPipeline, TaskProfile  ",
       style="dotted", color="#10B981", dir="forward")
g.edge("Pipeline_init", "train_py",
       label="  HCMARLPipeline (via config)  ",
       style="dotted", color="#3B82F6", dir="forward")

# ── LAYOUT ──
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("TaskProfile")
    s.node("WorkerState")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("step1")
    s.node("step2")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("steps3_6")
    s.node("step7")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("from_config")
    s.node("summary")

# LEGEND
legend_html = """<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#F8FAFC" COLOR="#CBD5E1">
<TR><TD COLSPAN="2" BGCOLOR="#334155"><FONT COLOR="white"><B>  LEGEND  </B></FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#DC2626">red bold</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#DC2626">7-step pipeline loop</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">solid arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">direct data flow</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">dashed arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">self.attribute access</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#3B82F6">blue dotted</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#3B82F6">cross-file import</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#10B981">green dotted</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#10B981">test import</FONT></TD></TR>
</TABLE>"""
g.node("legend", "<" + legend_html + ">")

g.render("diagrams/flowchart_pipeline", cleanup=True)
print("SUCCESS: diagrams/flowchart_pipeline.png")
