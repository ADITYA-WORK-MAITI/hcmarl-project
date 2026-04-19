"""Generate flowchart diagram for hcmarl/nswf_allocator.py"""
import graphviz
import os
os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + ";" + os.environ.get("PATH", "")

g = graphviz.Digraph("nswf_allocator", format="png")
g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
       nodesep="0.5", ranksep="0.7", margin="0.3", size="30,45")
g.attr("node", fontname="Consolas", fontsize="10", shape="none")
g.attr("edge", fontname="Consolas", fontsize="8", color="#475569")

HDR  = "#0F172A"
CLS  = "#1E3A5F"
FUNC = "#2D4A22"
RES  = "#7C2D12"


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
mk("file_hdr", "hcmarl/nswf_allocator.py", HDR, [
    ("Purpose", "Nash Social Welfare task allocator"),
    ("Equations", "Eqs 31-33 (Math PDF Sec 6)"),
    ("Lines", "360"),
    ("Imports", "numpy, hcmarl.utils.safe_log"),
])

# ── NSWFParams ──
mk("NSWFParams", "@dataclass NSWFParams", CLS, [
    ("kappa: float", "1.0 (disagreement scaling, Eq 32)"),
    ("epsilon: float", "1e-3 (rest-task surplus, Eq 31)"),
    ("__post_init__", "validates kappa>0, epsilon>0"),
])

# ── AllocationResult ──
mk("AllocationResult", "@dataclass AllocationResult", RES, [
    ("assignments", "dict[int,int]: worker i -> task j"),
    ("objective_value", "float: NSWF objective (Eq 33)"),
    ("surpluses", "dict[int,float]: U(i,j*)-D_i"),
    ("disagreement_utilities", "dict[int,float]: D_i(MF_i)"),
])

# ── NSWFAllocator.__init__ ──
mk("NSWF_init", "class NSWFAllocator.__init__", CLS, [
    ("params", "NSWFParams (or default)"),
])

# ── Disagreement utility ──
mk("disagreement", "disagreement_utility(MF)", FUNC, [
    ("Input", "MF: float in [0,1)"),
    ("Formula", "kappa * MF^2 / (1-MF) (Eq 32)"),
    ("Properties", "D(0)=0, D'>0, D->inf as MF->1"),
    ("Return", "float"),
])

mk("disagreement_deriv", "disagreement_derivative(MF)", FUNC, [
    ("Input", "MF: float in [0,1)"),
    ("Formula", "kappa*MF*(2-MF)/(1-MF)^2"),
    ("Return", "float (always positive for MF>0)"),
])

# ── Rest utility ──
mk("rest_utility", "rest_utility(MF)", FUNC, [
    ("Input", "MF: float"),
    ("Formula", "D_i(MF) + epsilon (Eq 31)"),
    ("Return", "float (surplus = epsilon exactly)"),
])

# ── Surplus ──
mk("surplus", "surplus(utility, MF)", FUNC, [
    ("Input", "U(i,j): float, MF: float"),
    ("Formula", "U(i,j) - D_i(MF)"),
    ("Return", "float (+ve = individually rational)"),
])

# ── Allocate ──
mk("allocate", "allocate(utility_matrix, fatigue_levels)", FUNC, [
    ("Input", "U: ndarray(N,M), MF: ndarray(N,)"),
    ("Computes", "D_i for all workers"),
    ("Computes", "surplus_matrix = U - D[:, None]"),
    ("Branch", "N,M<=8 -> exact | else -> greedy"),
    ("Return", "AllocationResult"),
])

# ── Exact solver ──
mk("solve_exact", "_solve_exact(N, M, surplus, D, eps)", FUNC, [
    ("Method", "recursive enumeration"),
    ("Constraint", "1 task per worker, 1 worker per task"),
    ("Rest", "task 0 always available"),
    ("Objective", "sum_i ln(surplus_i) (Eq 33)"),
    ("Complexity", "O(M! / (M-N)!) for small N,M"),
])

# ── Greedy solver ──
mk("solve_greedy", "_solve_greedy(N, M, surplus, D, eps)", FUNC, [
    ("Method", "sort by ln(surplus)-ln(eps) desc"),
    ("Greedy", "assign best unassigned pair"),
    ("Unassigned", "default to rest (task 0)"),
    ("Complexity", "O(NM log(NM))"),
])

# ── External nodes ──
for ext in ["utils", "pipeline", "test_nswf", "test_pipeline"]:
    g.node(ext, ext.replace("_", " "), shape="plaintext",
           fontname="Consolas", fontsize="8", fontcolor="#94A3B8")

# ── EDGES ──
g.edge("file_hdr", "NSWFParams", label="  defines  ", style="bold")
g.edge("file_hdr", "AllocationResult", label="  defines  ", style="bold")
g.edge("file_hdr", "NSWF_init", label="  defines  ", style="bold")

g.edge("NSWF_init", "disagreement", label="  self.params  ", style="dashed")
g.edge("NSWF_init", "allocate", label="  self.params  ", style="dashed")
g.edge("disagreement", "disagreement_deriv", label="  same formula base  ", style="dotted")
g.edge("disagreement", "rest_utility", label="  D_i(MF)  ")
g.edge("disagreement", "surplus", label="  D_i(MF)  ")

g.edge("allocate", "disagreement", label="  D_i for each worker  ")
g.edge("allocate", "solve_exact", label="  N,M <= 8  ")
g.edge("allocate", "solve_greedy", label="  N,M > 8  ")
g.edge("solve_exact", "AllocationResult", label="  returns  ", color="#DC2626")
g.edge("solve_greedy", "AllocationResult", label="  returns  ", color="#DC2626")

# Cross-file
g.edge("utils", "NSWF_init", label="  safe_log  ", style="dotted", color="#3B82F6")
g.edge("NSWF_init", "pipeline", label="  NSWFAllocator, NSWFParams, AllocationResult  ",
       style="dotted", color="#3B82F6", dir="forward")
g.edge("NSWF_init", "test_nswf", label="  full API  ",
       style="dotted", color="#10B981", dir="forward")
g.edge("NSWFParams", "test_pipeline", label="  NSWFParams  ",
       style="dotted", color="#10B981", dir="forward")

# ── LAYOUT ──
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("NSWFParams")
    s.node("AllocationResult")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("disagreement")
    s.node("rest_utility")
    s.node("surplus")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("solve_exact")
    s.node("solve_greedy")

# LEGEND
legend_html = """<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#F8FAFC" COLOR="#CBD5E1">
<TR><TD COLSPAN="2" BGCOLOR="#334155"><FONT COLOR="white"><B>  LEGEND  </B></FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">solid arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">direct data flow</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">dashed arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">self.attribute access</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#DC2626">red arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#DC2626">allocation result output</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#3B82F6">blue dotted</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#3B82F6">cross-file import (prod)</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#10B981">green dotted</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#10B981">cross-file import (test)</FONT></TD></TR>
</TABLE>"""
g.node("legend", "<" + legend_html + ">")

g.render("diagrams/flowchart_nswf_allocator", cleanup=True)
print("SUCCESS: diagrams/flowchart_nswf_allocator.png")
