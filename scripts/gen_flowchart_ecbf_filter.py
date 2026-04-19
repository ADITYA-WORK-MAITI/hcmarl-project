"""Generate flowchart diagram for hcmarl/ecbf_filter.py"""
import graphviz
import os
os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + ";" + os.environ.get("PATH", "")

g = graphviz.Digraph("ecbf_filter", format="png")
g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
       nodesep="0.5", ranksep="0.7", margin="0.3", size="32,55")
g.attr("node", fontname="Consolas", fontsize="10", shape="none")
g.attr("edge", fontname="Consolas", fontsize="8", color="#475569")

HDR   = "#0F172A"
CLS   = "#1E3A5F"
FUNC  = "#2D4A22"
CONST = "#6B3A1E"
PROP  = "#4A2A5C"
DIAG  = "#7C2D12"


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
mk("file_hdr", "hcmarl/ecbf_filter.py", HDR, [
    ("Purpose", "ECBF dual-barrier safety filter (CBF-QP)"),
    ("Equations", "Eqs 12-30 (Math PDF Sec 5)"),
    ("Lines", "428"),
    ("Imports", "cvxpy, numpy, hcmarl.three_cc_r"),
])

# ── ECBFParams ──
mk("ECBFParams", "@dataclass ECBFParams", CLS, [
    ("theta_max: float", "max fatigue fraction (Eq 12)"),
    ("alpha1: float", "0.05 (ECBF gain psi_1, Eq 16)"),
    ("alpha2: float", "0.05 (ECBF gain Eq 17)"),
    ("alpha3: float", "0.1 (resting-floor CBF gain, Eq 23)"),
])

mk("validate", "ECBFParams.validate(muscle)", FUNC, [
    ("Input", "MuscleParams"),
    ("Check", "theta_max in (0,1)"),
    ("Check", "theta_max >= muscle.theta_min_max (Eq 26)"),
    ("Check", "all gains > 0"),
    ("Raises", "ValueError"),
])

# ── ECBFDiagnostics ──
mk("ECBFDiag", "@dataclass ECBFDiagnostics", DIAG, [
    ("C_nominal", "float: input neural drive"),
    ("C_filtered", "float: output safe drive"),
    ("h, h2", "barrier values (Eqs 12, 21)"),
    ("psi_0, psi_1", "composite barriers (Eqs 15-16)"),
    ("h_dot", "barrier derivative (Eq 13)"),
    ("C_upper_ecbf", "ECBF upper bound (Eq 19)"),
    ("C_upper_cbf", "CBF upper bound (Eq 23)"),
    ("qp_status", "str: CVXPY solver status"),
    ("was_clipped", "bool: C_nominal modified?"),
])

# ── ECBFFilter.__init__ ──
mk("ECBFFilter_init", "class ECBFFilter.__init__", CLS, [
    ("muscle: MuscleParams", "from three_cc_r"),
    ("ecbf_params: ECBFParams", "gains + threshold"),
    ("Pre-extracts", "_F, _R, _Rr, _theta_max, _alpha1-3"),
])

# ── Barrier functions ──
mk("barriers", "Barrier Functions", FUNC, [
    ("h(MF)", "theta_max - MF (Eq 12)"),
    ("h2(MR)", "MR (Eq 21)"),
    ("h_dot(MA, MF, Reff)", "-F*MA + Reff*MF (Eq 13)"),
    ("h_ddot(MA, MF, C, Reff)", "-F*C + F^2*MA + Reff*F*MA - Reff^2*MF (Eq 14)"),
    ("psi_0(MF)", "= h(MF) (Eq 15)"),
    ("psi_1(MA, MF, Reff)", "h_dot + alpha1*h (Eq 16)"),
    ("h2_dot(MF, C, Reff)", "Reff*MF - C (Eq 22)"),
])

# ── Analytical bounds ──
mk("bounds", "Analytical Upper Bounds", FUNC, [
    ("ecbf_upper_bound(MA, MF, Reff)", "C <= ... (Eq 19)"),
    ("cbf_upper_bound(MA, MF, Reff)", "C <= Reff*MF + alpha3*MR (Eq 23)"),
])

# ── QP Solver ──
mk("filter_qp", "ECBFFilter.filter() — QP Solver", FUNC, [
    ("Input", "ThreeCCrState, C_nominal, target_load"),
    ("Solver", "CVXPY OSQP (warm_start=True)"),
    ("Objective", "min ||C - C_nom||^2 (Eq 20)"),
    ("Constraint 1", "ECBF: -F*C + [terms] >= 0 (Eq 18)"),
    ("Constraint 2", "CBF: C <= Reff*MF + alpha3*MR (Eq 23)"),
    ("Constraint 3", "C >= 0"),
    ("Fallback", "SolverError -> analytical bounds"),
    ("Infeasible", "force C=0 (Remark 5.13)"),
    ("Return", "(C_filtered, ECBFDiagnostics)"),
])

# ── Analytical filter ──
mk("filter_analytical", "ECBFFilter.filter_analytical()", FUNC, [
    ("Input", "ThreeCCrState, C_nominal, target_load"),
    ("Method", "min(C_nom, ub_ecbf, ub_cbf), max(0)"),
    ("Return", "float C* (no QP overhead)"),
])

# ── Rest-phase analysis ──
mk("rest_analysis", "Rest-Phase Analysis (Sec 5.4)", FUNC, [
    ("rest_phase_safe(state)", "h>=0 and h2>=0 (Thm 5.7)"),
    ("psi1_jump_at_rest(MF)", "R*(r-1)*MF (Eq 28)"),
    ("min_rest_duration_bound(MA)", "(1/F)*ln(...) (Eq 30)"),
])

# ── External nodes ──
for ext in ["three_cc_r", "pipeline", "test_ecbf", "test_pipeline"]:
    g.node(ext, ext.replace("_", " "), shape="plaintext",
           fontname="Consolas", fontsize="8", fontcolor="#94A3B8")

# ── EDGES ──
g.edge("file_hdr", "ECBFParams", label="  defines  ", style="bold")
g.edge("file_hdr", "ECBFDiag", label="  defines  ", style="bold")
g.edge("file_hdr", "ECBFFilter_init", label="  defines  ", style="bold")

g.edge("ECBFParams", "validate", label="  method  ")
g.edge("ECBFFilter_init", "barriers", label="  self._F, _R, _alpha*  ", style="dashed")
g.edge("ECBFFilter_init", "bounds", label="  self._F, _alpha*  ", style="dashed")
g.edge("ECBFFilter_init", "filter_qp", label="  self.*  ", style="dashed")
g.edge("ECBFFilter_init", "filter_analytical", style="dashed")
g.edge("ECBFFilter_init", "rest_analysis", style="dashed")

g.edge("barriers", "filter_qp", label="  h, h2, psi_1, h_dot  ")
g.edge("bounds", "filter_qp", label="  C_ub_ecbf, C_ub_cbf  ", style="dashed")
g.edge("barriers", "bounds", label="  barrier values  ")
g.edge("filter_qp", "ECBFDiag", label="  returns diagnostics  ", color="#DC2626")
g.edge("bounds", "filter_analytical", label="  ub_ecbf, ub_cbf  ")

# Cross-file: inbound
g.edge("three_cc_r", "ECBFFilter_init", label="  MuscleParams, ThreeCCrState  ",
       style="dotted", color="#3B82F6")

# Cross-file: outbound
g.edge("ECBFFilter_init", "pipeline", label="  ECBFFilter, ECBFParams  ",
       style="dotted", color="#3B82F6", dir="forward")
g.edge("ECBFFilter_init", "test_ecbf", label="  full API  ",
       style="dotted", color="#10B981", dir="forward")
g.edge("ECBFParams", "test_pipeline", label="  ECBFParams  ",
       style="dotted", color="#10B981", dir="forward")

# ── LAYOUT ──
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("ECBFParams")
    s.node("ECBFDiag")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("validate")
    s.node("ECBFFilter_init")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("barriers")
    s.node("bounds")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("filter_qp")
    s.node("filter_analytical")
    s.node("rest_analysis")

# LEGEND
legend_html = """<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#F8FAFC" COLOR="#CBD5E1">
<TR><TD COLSPAN="2" BGCOLOR="#334155"><FONT COLOR="white"><B>  LEGEND  </B></FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">solid arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">direct data flow</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">dashed arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">self.attribute access</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#DC2626">red arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#DC2626">safety-critical output</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#3B82F6">blue dotted</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#3B82F6">cross-file import (prod)</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#10B981">green dotted</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#10B981">cross-file import (test)</FONT></TD></TR>
</TABLE>"""
g.node("legend", "<" + legend_html + ">")

g.render("diagrams/flowchart_ecbf_filter", cleanup=True)
print("SUCCESS: diagrams/flowchart_ecbf_filter.png")
