"""Generate flowchart diagram for hcmarl/three_cc_r.py"""
import graphviz
import os
os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + ";" + os.environ.get("PATH", "")

g = graphviz.Digraph("three_cc_r", format="png")
g.attr(dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
       nodesep="0.5", ranksep="0.7", margin="0.3", size="30,50")
g.attr("node", fontname="Consolas", fontsize="10", shape="none")
g.attr("edge", fontname="Consolas", fontsize="8", color="#475569")

# ── Colour palette ──
HDR   = "#0F172A"   # file header
CLS   = "#1E3A5F"   # class header
FUNC  = "#2D4A22"   # standalone function
CONST = "#6B3A1E"   # constants / module-level data
PROP  = "#4A2A5C"   # property / derived


def esc(text):
    """Escape text for Graphviz HTML labels."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_node(name, title, color, rows):
    """Build an HTML-table Graphviz node.
    rows: list of (left_col, right_col) tuples.
    """
    hdr = '<TR><TD COLSPAN="2" BGCOLOR="{}" ALIGN="LEFT"><FONT COLOR="white"><B>  {}  </B></FONT></TD></TR>'.format(color, esc(title))
    body = ""
    for left, right in rows:
        r = right if right else " "
        body += '<TR><TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="9">  {}  </FONT></TD>'.format(esc(left))
        body += '<TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="8" COLOR="#6B7280">  {}  </FONT></TD></TR>'.format(esc(r))
    html = '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" COLOR="#CBD5E1">{}{}</TABLE>'.format(hdr, body)
    g.node(name, "<" + html + ">")


# ═══════════════════════════════════════════════════════════════
# FILE HEADER
# ═══════════════════════════════════════════════════════════════
make_node("file_hdr", "hcmarl/three_cc_r.py", HDR, [
    ("Purpose", "3CC-r fatigue ODE model"),
    ("Equations", "Eqs 1-8, Eq 35 (Math PDF Sec 3)"),
    ("Lines", "386"),
    ("Imports", "numpy, scipy.integrate.solve_ivp, dataclasses"),
])

# ═══════════════════════════════════════════════════════════════
# MuscleParams dataclass
# ═══════════════════════════════════════════════════════════════
make_node("MuscleParams", "@dataclass MuscleParams (frozen)", CLS, [
    ("name: str", "'shoulder'"),
    ("F: float", "0.0146 (fatigue rate, min^-1)"),
    ("R: float", "0.00058 (recovery rate, min^-1)"),
    ("r: float", "15 (reperfusion multiplier)"),
])

make_node("MuscleParams_props", "MuscleParams @properties", PROP, [
    ("C_max", "F*R/(F+R) = max sustainable drive"),
    ("delta_max", "R/(F+R) = max sustainable duty"),
    ("Rr", "R*r = reperfusion recovery rate"),
    ("theta_min_max", "F/(F+R*r) = rest-phase threshold"),
    ("Rr_over_F", "Rr/F = rest overshoot ratio"),
])

# ═══════════════════════════════════════════════════════════════
# Module-level constants
# ═══════════════════════════════════════════════════════════════
make_node("constants", "Module Constants (Table 1)", CONST, [
    ("SHOULDER", "F=0.0146  R=0.00058  r=15"),
    ("ANKLE", "F=0.00589 R=0.0182   r=15"),
    ("KNEE", "F=0.0150  R=0.00175  r=15"),
    ("ELBOW", "F=0.00912 R=0.00094  r=15"),
    ("TRUNK", "F=0.00657 R=0.00354  r=15"),
    ("GRIP", "F=0.00794 R=0.00109  r=30"),
    ("ALL_MUSCLES", "list[MuscleParams] len=6"),
    ("MUSCLE_REGISTRY", "dict[str, MuscleParams] len=6"),
])

# ═══════════════════════════════════════════════════════════════
# get_muscle()
# ═══════════════════════════════════════════════════════════════
make_node("get_muscle", "get_muscle(name: str)", FUNC, [
    ("Input", "name: str  e.g. 'shoulder'"),
    ("Lookup", "MUSCLE_REGISTRY[name.lower()]"),
    ("Return", "MuscleParams"),
    ("Raises", "KeyError if unknown muscle"),
])

# ═══════════════════════════════════════════════════════════════
# ThreeCCrState dataclass
# ═══════════════════════════════════════════════════════════════
make_node("ThreeCCrState", "@dataclass ThreeCCrState", CLS, [
    ("MR: float", "resting fraction [0,1]"),
    ("MA: float", "active fraction [0,1]"),
    ("MF: float", "fatigued fraction [0,1]"),
    ("__post_init__", "validates MR+MA+MF=1 (Eq 1)"),
])

make_node("State_methods", "ThreeCCrState methods", FUNC, [
    ("as_array()", "-> np.ndarray [MR, MA, MF]"),
    ("from_array(arr)", "@classmethod -> ThreeCCrState"),
    ("fresh()", "@classmethod -> MR=1, MA=0, MF=0"),
])

# ═══════════════════════════════════════════════════════════════
# ThreeCCr class
# ═══════════════════════════════════════════════════════════════
make_node("ThreeCCr_init", "class ThreeCCr.__init__", CLS, [
    ("params: MuscleParams", "calibrated muscle group"),
    ("kp: float", "10.0 (proportional gain, Eq 35)"),
])

make_node("R_eff", "ThreeCCr.R_eff(target_load)", FUNC, [
    ("Input", "target_load: float [0,1]"),
    ("Logic", "TL>0 -> R  |  TL=0 -> R*r"),
    ("Return", "float (effective recovery rate)"),
    ("Equation", "Eq 5: reperfusion switch"),
])

make_node("baseline_neural_drive", "ThreeCCr.baseline_neural_drive(TL, MA)", FUNC, [
    ("Input", "target_load: float, MA: float"),
    ("Logic", "TL<=0 -> 0  |  else kp*max(TL-MA,0)"),
    ("Return", "float C(t) in [0, kp]"),
    ("Equation", "Eq 35: proportional controller"),
])

make_node("ode_rhs", "ThreeCCr.ode_rhs(state, C, TL)", FUNC, [
    ("Input", "state: [MR,MA,MF], C: float, TL: float"),
    ("dMA/dt", "C - F*MA  (Eq 2)"),
    ("dMF/dt", "F*MA - Reff*MF  (Eq 3)"),
    ("dMR/dt", "Reff*MF - C  (Eq 4)"),
    ("Return", "np.ndarray [dMR, dMA, dMF]"),
])

make_node("step_euler", "ThreeCCr.step_euler(state, C, TL, dt)", FUNC, [
    ("Input", "ThreeCCrState, C, TL, dt=1.0 min"),
    ("Method", "x_new = x + dt * ode_rhs(x, C, TL)"),
    ("Clamp", "clip [0,1] then renormalize sum=1"),
    ("Return", "ThreeCCrState (new)"),
    ("Used by", "RL env step loop"),
])

make_node("simulate", "ThreeCCr.simulate(state0, TL, dur, dt_eval, C_override)", FUNC, [
    ("Input", "state0, TL, duration, dt_eval=0.1"),
    ("Method", "scipy solve_ivp RK45 (rtol=1e-8)"),
    ("C_override", "optional fixed C (for testing)"),
    ("Return", "dict: t, MR, MA, MF, C arrays"),
    ("Used by", "calibration, plotting, validation"),
])

make_node("verify_conservation", "ThreeCCr.verify_conservation(state)", FUNC, [
    ("Input", "ThreeCCrState"),
    ("Check", "|MR+MA+MF - 1| &lt; tol"),
    ("Return", "bool"),
])

make_node("steady_state_work", "ThreeCCr.steady_state_work()", FUNC, [
    ("Logic", "MR=0, MA=delta_max, MF=1-delta_max"),
    ("Equation", "Eqs 7-8 (Theorem 3.4)"),
    ("Return", "ThreeCCrState"),
])

# ═══════════════════════════════════════════════════════════════
# CROSS-FILE I/O — dangling inward arrows (consumers importing FROM this file)
# ═══════════════════════════════════════════════════════════════
# Invisible source nodes for external imports
for ext in ["ecbf_filter", "pipeline", "real_data_calib", "pettingzoo_wrapper",
            "test_three_cc_r", "test_pipeline", "test_ecbf", "test_real_data_calib"]:
    g.node(ext, ext.replace("_", " "), shape="plaintext",
           fontname="Consolas", fontsize="8", fontcolor="#94A3B8")

# ═══════════════════════════════════════════════════════════════
# EDGES — internal data flow
# ═══════════════════════════════════════════════════════════════

# File header -> top-level constructs
g.edge("file_hdr", "MuscleParams", label="  defines  ", style="bold")
g.edge("file_hdr", "constants", label="  defines  ", style="bold")
g.edge("file_hdr", "get_muscle", label="  defines  ", style="bold")
g.edge("file_hdr", "ThreeCCrState", label="  defines  ", style="bold")
g.edge("file_hdr", "ThreeCCr_init", label="  defines  ", style="bold")

# MuscleParams internal
g.edge("MuscleParams", "MuscleParams_props", label="  @property  ")

# Constants use MuscleParams
g.edge("MuscleParams", "constants", label="  instantiates  ")

# get_muscle uses MUSCLE_REGISTRY
g.edge("constants", "get_muscle", label="  MUSCLE_REGISTRY  ")

# ThreeCCrState methods
g.edge("ThreeCCrState", "State_methods", label="  methods  ")

# ThreeCCr class method chain
g.edge("ThreeCCr_init", "R_eff", label="  self.params  ", style="dashed")
g.edge("ThreeCCr_init", "baseline_neural_drive", label="  self.kp  ", style="dashed")
g.edge("ThreeCCr_init", "ode_rhs", label="  self.params.F  ", style="dashed")

g.edge("R_eff", "ode_rhs", label="  Reff  ")
g.edge("baseline_neural_drive", "ode_rhs", label="  C(t)  ", style="dashed", color="#DC2626")
g.edge("ode_rhs", "step_euler", label="  dx/dt  ")
g.edge("ode_rhs", "simulate", label="  rhs()  ")
g.edge("ThreeCCrState", "step_euler", label="  state.as_array()  ")
g.edge("ThreeCCrState", "simulate", label="  state0.as_array()  ")
g.edge("step_euler", "ThreeCCrState", label="  from_array(x_new)  ", style="dotted", constraint="false")

g.edge("ThreeCCr_init", "verify_conservation", style="dashed")
g.edge("ThreeCCr_init", "steady_state_work", style="dashed")
g.edge("MuscleParams_props", "steady_state_work", label="  delta_max  ")

# ═══════════════════════════════════════════════════════════════
# CROSS-FILE EDGES — who imports what
# ═══════════════════════════════════════════════════════════════

# Outward arrows: this file's exports -> consumers
g.edge("MuscleParams", "ecbf_filter", label="  MuscleParams, ThreeCCrState  ",
       style="dotted", color="#3B82F6", dir="forward")
g.edge("MuscleParams", "pipeline", label="  MuscleParams, ThreeCCr, ThreeCCrState, get_muscle, SHOULDER  ",
       style="dotted", color="#3B82F6", dir="forward")
g.edge("MuscleParams", "real_data_calib", label="  MuscleParams, ThreeCCr, SHOULDER  ",
       style="dotted", color="#3B82F6", dir="forward")
g.edge("get_muscle", "pettingzoo_wrapper", label="  get_muscle()  ",
       style="dotted", color="#3B82F6", dir="forward")

# Test consumers
g.edge("MuscleParams", "test_three_cc_r", label="  full API  ",
       style="dotted", color="#10B981", dir="forward")
g.edge("constants", "test_pipeline", label="  SHOULDER, ELBOW, GRIP  ",
       style="dotted", color="#10B981", dir="forward")
g.edge("constants", "test_ecbf", label="  SHOULDER + classes  ",
       style="dotted", color="#10B981", dir="forward")
g.edge("constants", "test_real_data_calib", label="  SHOULDER  ",
       style="dotted", color="#10B981", dir="forward")

# ═══════════════════════════════════════════════════════════════
# LAYOUT — rank groupings
# ═══════════════════════════════════════════════════════════════

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("MuscleParams")
    s.node("constants")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("MuscleParams_props")
    s.node("get_muscle")
    s.node("ThreeCCrState")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("State_methods")
    s.node("ThreeCCr_init")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("R_eff")
    s.node("baseline_neural_drive")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("step_euler")
    s.node("simulate")
    s.node("verify_conservation")
    s.node("steady_state_work")

# External consumers at bottom
with g.subgraph() as s:
    s.attr(rank="same")
    for ext in ["ecbf_filter", "pipeline", "real_data_calib", "pettingzoo_wrapper",
                "test_three_cc_r", "test_pipeline", "test_ecbf", "test_real_data_calib"]:
        s.node(ext)

# ═══════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════
legend_html = """<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#F8FAFC" COLOR="#CBD5E1">
<TR><TD COLSPAN="2" BGCOLOR="#334155"><FONT COLOR="white"><B>  LEGEND  </B></FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">solid arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">direct data flow</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">dashed arrow</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#475569">self.attribute access</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#DC2626">red dashed</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#DC2626">neural drive C(t) path</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#3B82F6">blue dotted</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#3B82F6">cross-file import (prod)</FONT></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#10B981">green dotted</FONT></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#10B981">cross-file import (test)</FONT></TD></TR>
</TABLE>"""

g.node("legend", "<" + legend_html + ">")

# Render
g.render("diagrams/flowchart_three_cc_r", cleanup=True)
print("SUCCESS: diagrams/flowchart_three_cc_r.png")
