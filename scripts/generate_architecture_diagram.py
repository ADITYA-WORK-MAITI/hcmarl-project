"""
HC-MARL Architecture Diagram — v2
==================================
Incorporates all supervisor red-pen feedback:
  - Data flow made explicit (what goes where)
  - Constants vs variables clearly tagged with paper sources
  - ALL baselines listed by name (10 methods)
  - ALL performance metrics spelled out (9 metrics)
  - Task allocation sequence shown step-by-step
  - Runs/seeds (5 seeds x 10 methods = 50+ runs) shown
  - Performance study section prominent
  - Dimensionality (N, M, G, n, m) explicit everywhere
  - Offline vs Online phase separated
  - Contributions C1-C5 labelled
  - Inputs grouped and sourced
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import os

# ── colours ─────────────────────────────────────────────────────────────
BG       = "#FAFAFA"
C_INPUT  = "#E3F2FD"   # blue-tinted inputs
C_CONST  = "#BBDEFB"   # darker blue for constants
C_VAR    = "#FFE0B2"   # orange for variables
C_OFFLINE= "#F3E5F5"   # purple offline phase
C_ONLINE = "#E8F5E9"   # green online phase
C_STEP   = "#FFFFFF"   # white step boxes inside online
C_SAFETY = "#FCE4EC"   # pink safety
C_OUTPUT = "#FFF9C4"   # yellow outputs
C_EVAL   = "#EDE7F6"   # lavender evaluation
C_METHOD = "#E0F2F1"   # teal methods
C_NOVEL  = "#FFCDD2"   # red-tinted novel
C_SOURCE = "#F3E5F5"   # purple sourced
C_PERF   = "#DCEDC8"   # lime performance
C_BORDER = "#37474F"
C_TITLE  = "#0D47A1"
C_RED    = "#C62828"
C_GREEN  = "#2E7D32"
C_GREY   = "#78909C"
C_DARK   = "#212121"
C_BLUE   = "#1565C0"

fig, ax = plt.subplots(1, 1, figsize=(40, 72))
ax.set_xlim(0, 40)
ax.set_ylim(0, 72)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)

# ── drawing helpers ─────────────────────────────────────────────────────
def rbox(x, y, w, h, color=C_STEP, border=C_BORDER, lw=1.2, ls="-", zorder=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                       facecolor=color, edgecolor=border, linewidth=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(p)

def T(x, y, s, fs=9, ha="left", va="top", color=C_DARK, wt="normal",
      fam="monospace", zorder=5):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=color,
            weight=wt, family=fam, zorder=zorder,
            linespacing=1.35)

def heading(x, y, s, fs=14):
    T(x, y, s, fs=fs, ha="center", color=C_TITLE, wt="bold", fam="sans-serif")

def arr(x1, y1, x2, y2, color=C_BORDER, lw=1.8, hw=0.25):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f"->,head_width={hw}", color=color, lw=lw),
                zorder=3)

def darr(x1, y1, x2, y2, color=C_BORDER, lw=1.5):
    """Dashed arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle="dashed"), zorder=3)

def hline(y, x1=1, x2=39, color=C_GREY, lw=0.6):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, ls="--", zorder=1)

def tag(x, y, label, color, fs=7.5, bw=None, bh=0.4, border=None):
    if bw is None:
        bw = len(label) * 0.155 + 0.5
    if border is None:
        border = color
    rbox(x, y, bw, bh, color=color, border=border, lw=0.8)
    T(x + bw/2, y + bh/2, label, fs=fs, ha="center", va="center", color="#333")

def bracket_label(x, y, h, label, side="right", fs=8):
    """Draw a vertical bracket with label."""
    if side == "right":
        ax.plot([x, x+0.3, x+0.3, x], [y+h, y+h, y, y],
                color=C_RED, lw=1.5, zorder=4)
        T(x+0.5, y+h/2, label, fs=fs, ha="left", va="center", color=C_RED, wt="bold")
    else:
        ax.plot([x, x-0.3, x-0.3, x], [y+h, y+h, y, y],
                color=C_RED, lw=1.5, zorder=4)
        T(x-0.5, y+h/2, label, fs=fs, ha="right", va="center", color=C_RED, wt="bold")


# ====================================================================
#  TITLE
# ====================================================================
heading(20, 71.5, "HC-MARL: Human-Centric Multi-Agent Reinforcement Learning", fs=18)
T(20, 70.7, "Proposed Plan of Work  ·  Architecture  ·  Mathematical Framework v12",
  fs=11, ha="center", color=C_GREY, fam="sans-serif")
T(20, 70.2, "Aditya Maiti  ·  USATR, GGSIPU  ·  Supervisor: Dr. Amrit Pal Singh",
  fs=10, ha="center", color=C_GREY, fam="sans-serif")
ax.plot([1, 39], [69.8, 69.8], color=C_BORDER, lw=2)

# ====================================================================
#  SECTION 1: INPUTS  (addressing "Data?", "Constant Values",
#                       "Real Research Paper Values", "Set", "Group")
# ====================================================================
heading(20, 69.3, "INPUTS", fs=14)

# ── Input 1: Worker Demos ──
rbox(1, 66.5, 8.5, 2.5, color=C_INPUT)
T(1.3, 68.8, "Worker Demonstrations", fs=10, wt="bold")
T(1.3, 68.1,
  "Trajectories tau = (s0,a0,s1,...)\n"
  "Collected from N workers\n"
  "Used by MMICRL (Section 4)\n"
  "z in {0,..,K-1} latent types", fs=8)
tag(1.3, 66.7, "VARIABLE", C_VAR, fs=7)
tag(4.3, 66.7, "dim: N trajectories", C_INPUT, fs=7)

# ── Input 2: Muscle Parameters ──
rbox(10.5, 66.5, 8.5, 2.5, color=C_CONST)
T(10.8, 68.8, "Muscle Parameters", fs=10, wt="bold")
T(10.8, 68.1,
  "F, R, r per muscle group\n"
  "6 groups: shoulder, ankle,\n"
  "knee, elbow, trunk, grip\n"
  "From Table 1 of [3,4,5,6]", fs=8)
tag(10.8, 66.7, "CONSTANT", C_CONST, fs=7, border=C_BLUE)
tag(14.5, 66.7, "Source: [3,4,5,6]", C_SOURCE, fs=7)

# ── Input 3: Task Profiles ──
rbox(20, 66.5, 8.5, 2.5, color=C_CONST)
T(20.3, 68.8, "Task Profiles", fs=10, wt="bold")
T(20.3, 68.1,
  "T_L,g^(j) in [0,1]  (Eq 34)\n"
  "fraction of MVC per muscle\n"
  "M tasks x G muscle groups\n"
  "e.g. box_lift: sh=0.5,gr=0.6", fs=8)
tag(20.3, 66.7, "CONSTANT", C_CONST, fs=7, border=C_BLUE)
tag(24, 66.7, "dim: M x G matrix", C_INPUT, fs=7)

# ── Input 4: Configuration ──
rbox(29.5, 66.5, 9.5, 2.5, color=C_CONST)
T(29.8, 68.8, "Configuration (Set)", fs=10, wt="bold")
T(29.8, 68.1,
  "N workers, M tasks, G muscles\n"
  "dt=1.0min, kp=10.0, kappa=1.0\n"
  "eps=0.001, alpha1/2=0.05\n"
  "alpha3=0.10, Theta_max per muscle", fs=8)
tag(29.8, 66.7, "CONSTANT", C_CONST, fs=7, border=C_BLUE)
tag(33.5, 66.7, "YAML files", C_INPUT, fs=7)

# Grouping brace
bracket_label(39.3, 66.5, 2.5, "GROUPED\nINPUTS\n(all pre-set\nbefore run)", fs=8)

# Source annotations (addressing "Real Research Paper Values")
T(1.3, 66.1, "Sources: [1] Liu et al. 2002  [2] Xia & Frey-Law 2008  [3] Frey-Law et al. 2012  "
  "[4] Looft et al. 2018  [5] Looft & Frey-Law 2020  [6] Frey-Law & Avin 2010",
  fs=7, color=C_GREY, fam="sans-serif")

# Arrows from inputs down
for cx in [5.25, 14.75, 24.25, 34.25]:
    arr(cx, 66.5, cx, 65.3)

# ====================================================================
#  SECTION 2: OFFLINE PHASE  (MMICRL)
# ====================================================================
hline(65.8)
rbox(1, 62.3, 38, 3.2, color=C_OFFLINE, border="#7B1FA2", lw=2)
T(2, 65.3, "OFFLINE PHASE", fs=12, wt="bold", color="#7B1FA2")

T(2, 64.5, "MMICRL  (Section 4, Eqs 9-11)", fs=11, wt="bold")
T(2, 63.8,
  "Objective:  max_pi [ lambda1 * H[pi(tau)] - lambda2 * H[pi(tau|z)] ]            [Eq 9]\n"
  "Decomposed: = lambda2 * I(tau;z) + (lambda1 - lambda2) * H[pi(tau)]              [Eq 10]\n"
  "Default:    lambda1 = lambda2 = lambda  =>  max lambda * I(tau;z)                 [Eq 11]", fs=9)

# MMICRL output
rbox(2, 62.5, 18, 0.6, color="#CE93D8")
T(2.3, 62.95, "Output: K discovered types + learned Theta_max^(k) per type   (dim: K floats)", fs=8, wt="bold")

# Annotations
tag(26, 64.8, "NOVEL correction C1: lambda1 != lambda2 residual", C_NOVEL, fs=8, bw=11)
tag(26, 64.2, "Borrowed: Qiao et al. [8], MaxEnt IRL [7]", C_SOURCE, fs=8, bw=9.5)
tag(26, 63.6, "VARIABLE: trajectories tau, latent z", C_VAR, fs=8, bw=8.5)
tag(26, 63.0, "CONSTANT: lambda1, lambda2 (default=1.0)", C_CONST, fs=8, bw=9)

arr(20, 62.5, 20, 61.5)

# ====================================================================
#  SECTION 3: ONLINE PHASE — 7-Step Pipeline
#  (addressing "Task allocation Sequence", "S→A")
# ====================================================================
rbox(1, 42.5, 38, 19.2, color=C_ONLINE, border=C_GREEN, lw=2.5)
T(2, 61.5, "ONLINE PHASE  (Section 7.3 — repeats each allocation round)", fs=12, wt="bold", color=C_GREEN)

# Step dimensions reference
rbox(26, 60.5, 12.5, 1.0, color="#C8E6C9", border=C_GREEN)
T(26.3, 61.25, "Per round: N workers x G muscles x 3 states\n"
  "Default: N=4, G=3, M=4 tasks  (configurable)", fs=8, color=C_GREEN, wt="bold")

# ── STEP 1: Observe State ──
sy = 59.5
rbox(3, sy, 34, 1.3, color=C_STEP, border=C_GREEN)
T(3.3, sy+1.15, "Step 1 — Observe State  (Section 3)", fs=10, wt="bold")
T(3.3, sy+0.55, "Read x_i(t) = [M_R, M_A, M_F]^T for each worker i in {1,...,N}, each muscle g in {1,...,G}", fs=9)
# dimension
T(26, sy+0.2, "obs dim = N*G*3 + 1 = 37 (default)", fs=8, color=C_BLUE, wt="bold")
# State explanation
T(3.3, sy+0.15, "M_R=resting fraction  M_A=active fraction  M_F=fatigued fraction  (sum=1 always, Thm 3.2)", fs=7.5, color=C_GREY)
arr(20, sy, 20, sy - 0.3)

# ── STEP 2: Fair Task Allocation (NSWF) ──
sy2 = sy - 2.1
rbox(3, sy2, 34, 1.8, color=C_STEP, border=C_GREEN)
T(3.3, sy2+1.65, "Step 2 — Fair Task Allocation  (Section 6, Eqs 31-33)", fs=10, wt="bold")
T(3.3, sy2+1.05,
  "Nash Social Welfare:  max SUM_{i=1}^{N} ln( U(i, j*(i)) - D_i(M_F^i) )          [Eq 33]\n"
  "Disagreement utility: D_i = kappa * (M_F)^2 / (1 - M_F)  with D_i(0)=0, D_i->inf [Eq 32]\n"
  "Rest option:          U(i,0) = D_i + eps  =>  rest surplus = eps (always feasible) [Eq 31]", fs=8.5)
T(26, sy2+0.15, "Input: U in R^{NxM}, M_F in R^N\nOutput: j*(i) in {0,1,...,M}", fs=8, color=C_BLUE, wt="bold")

# Task allocation sequence detail (addressing supervisor's "Task allocation Sequence (2)")
rbox(3.5, sy2+0.05, 21, 0.6, color="#FFF9C4")
T(3.8, sy2+0.5,
  "Sequence: compute D_i -> build surplus matrix (NxM) -> solve assignment -> output j*(i)", fs=7.5, wt="bold")

arr(20, sy2, 20, sy2 - 0.3)

# ── STEP 3: Load Translation ──
sy3 = sy2 - 1.5
rbox(3, sy3, 34, 1.2, color=C_STEP, border=C_GREEN)
T(3.3, sy3+1.05, "Step 3 — Load Translation  (Section 7.1, Eq 34)", fs=10, wt="bold")
T(3.3, sy3+0.45, "Map assigned task j*(i) to target load vector:  T_L^(j) = [T_{L,g1}, ..., T_{L,gG}] in [0,1]^G\n"
  "If j=0 (rest): T_L = 0 for all muscles   |   Lookup from task demand profile matrix (M x G)", fs=8.5)
T(30, sy3+0.15, "dim: T_L in [0,1]^G", fs=8, color=C_BLUE, wt="bold")
arr(20, sy3, 20, sy3 - 0.3)

# ── STEP 4: Neural Drive Proposal ──
sy4 = sy3 - 1.6
rbox(3, sy4, 34, 1.3, color=C_STEP, border=C_GREEN)
T(3.3, sy4+1.15, "Step 4 — RL Policy / Baseline Neural Drive  (Section 7.2, Eq 35)", fs=10, wt="bold")
T(3.3, sy4+0.55,
  "Baseline controller: C_nom = kp * max(T_L - M_A, 0)   [Eq 35]   kp=10.0 [CONST]\n"
  "RL policy (trained): C_nom = pi_theta(x, task)  =>  replaces baseline after training", fs=8.5)
T(30, sy4+0.15, "dim: C_nom in R^1 per muscle", fs=8, color=C_BLUE, wt="bold")

# S→A annotation (addressing supervisor feedback)
tag(26, sy4+0.2, "STATE -> ACTION mapping", C_VAR, fs=8, bw=6)

arr(20, sy4, 20, sy4 - 0.3)

# ── STEP 5: Safety Filter (ECBF) ──
sy5 = sy4 - 2.8
rbox(3, sy5, 34, 2.5, color=C_SAFETY, border=C_RED)
T(3.3, sy5+2.35, "Step 5 — Safety Filter: Dual-Barrier ECBF  (Section 5, Eqs 12-26)", fs=10, wt="bold", color=C_RED)
T(3.3, sy5+1.7,
  "Barrier 1 (fatigue ceiling, relative degree 2):  h1 = Theta_max - M_F >= 0       [Eq 12]\n"
  "  h1_dot = -F*M_A + R_eff*M_F  (C absent!)  =>  relative degree >= 2              [Eq 13]\n"
  "  h1_ddot = -F*C + F^2*M_A + R_eff*F*M_A - R_eff^2*M_F  (C appears, coeff = -F)  [Eq 14]\n"
  "  psi_1 = h1_dot + alpha1 * h1                                                     [Eq 16]\n"
  "Barrier 2 (resting floor, relative degree 1):  h2 = M_R >= 0                       [Eq 21]\n"
  "  h2_dot = R_eff*M_F - C  (C appears, coeff = -1)                                  [Eq 22]", fs=8)
T(3.3, sy5+0.25,
  "CBF-QP:  C* = argmin ||C - C_nom||^2   s.t.  Eq18 + Eq23 + C>=0   [Eq 20]  Solver: CVXPY/OSQP", fs=8.5, wt="bold")
T(26, sy5+0.05, "Input: C_nom, x in R^3\nOutput: C* in R^1 (safe)", fs=8, color=C_RED, wt="bold")

arr(20, sy5, 20, sy5 - 0.3)

# ── STEP 6: ODE Integration ──
sy6 = sy5 - 1.8
rbox(3, sy6, 34, 1.5, color=C_STEP, border=C_GREEN)
T(3.3, sy6+1.35, "Step 6 — 3CC-r Physiological Plant  (Section 3, Eqs 2-5)", fs=10, wt="bold")
T(3.3, sy6+0.75,
  "dM_A/dt = C*(t) - F*M_A   [Eq 2]     R_eff = R (work) | R*r (rest)  [Eq 5]\n"
  "dM_F/dt = F*M_A - R_eff*M_F  [Eq 3]   Euler: x(t+dt) = x(t) + dt*f(x, C*)\n"
  "dM_R/dt = R_eff*M_F - C*(t)  [Eq 4]   Renormalise: M_R+M_A+M_F = 1 (Thm 3.2)", fs=8.5)
T(30, sy6+0.15, "x(t+dt) in R^3", fs=8, color=C_BLUE, wt="bold")
tag(26, sy6+0.15, "dt = 1.0 min [CONST]", C_CONST, fs=8, bw=5)

# Loop arrow back to Step 1
ax.annotate("", xy=(3, sy+0.65), xytext=(3, sy6+0.75),
            arrowprops=dict(arrowstyle="->,head_width=0.4", color=C_RED, lw=3,
                            connectionstyle="arc3,rad=-0.08"), zorder=4)
T(1.2, (sy + sy6 + 1.4) / 2, "REPEAT\nevery\nround\n(480x\nfor 8hr\nshift)", fs=8.5,
  ha="center", va="center", color=C_RED, wt="bold")

# ====================================================================
#  SECTION 4: OUTPUTS  (addressing "P1, P2, P3, P4" priority)
# ====================================================================
hline(42.0)
heading(20, 41.5, "OUTPUTS  (from each simulation run)", fs=13)

# Output 1
rbox(1, 39.0, 9, 2.0, color=C_OUTPUT, border="#F57F17")
T(1.3, 40.8, "P1: Safety", fs=10, wt="bold", color="#E65100")
T(1.3, 40.2, "violation rate, max M_F,\ncost per episode,\nECBF intervention count", fs=8)
tag(1.3, 39.2, "Priority 1 (most critical)", C_NOVEL, fs=7)

# Output 2
rbox(10.5, 39.0, 9, 2.0, color=C_OUTPUT, border="#F57F17")
T(10.8, 40.8, "P2: Productivity", fs=10, wt="bold", color="#E65100")
T(10.8, 40.2, "episode reward (task\ncompletion), throughput\n(tasks/hour)", fs=8)
tag(10.8, 39.2, "Priority 2", C_OUTPUT, fs=7, border="#F57F17")

# Output 3
rbox(20, 39.0, 9, 2.0, color=C_OUTPUT, border="#F57F17")
T(20.3, 40.8, "P3: Fairness", fs=10, wt="bold", color="#E65100")
T(20.3, 40.2, "Jain's fairness index,\nworker utilisation %,\nrest fraction per worker", fs=8)
tag(20.3, 39.2, "Priority 3", C_OUTPUT, fs=7, border="#F57F17")

# Output 4
rbox(29.5, 39.0, 9.5, 2.0, color=C_OUTPUT, border="#F57F17")
T(29.8, 40.8, "P4: Physiological", fs=10, wt="bold", color="#E65100")
T(29.8, 40.2, "fatigue trajectories\nM_F(t) over 8-hour shift,\nrest/work phase timings", fs=8)
tag(29.8, 39.2, "Priority 4", C_OUTPUT, fs=7, border="#F57F17")

for cx in [5.5, 15, 24.5, 34.25]:
    arr(cx, 42.5, cx, 41.0)

for cx in [5.5, 15, 24.5, 34.25]:
    arr(cx, 39.0, cx, 38.5)

# ====================================================================
#  SECTION 5: PERFORMANCE STUDY  (addressing "Performance Study",
#             "list out all performance measures", all baselines)
# ====================================================================
heading(20, 38.2, "PERFORMANCE STUDY", fs=14)
hline(37.8)

# ── All 9 metrics listed ──
rbox(1, 35.0, 38, 2.5, color=C_PERF, border=C_GREEN, lw=1.5)
T(2, 37.3, "ALL 9 PERFORMANCE METRICS  (measured for every method, every seed)", fs=11, wt="bold", color=C_GREEN)
T(2, 36.6,
  "  #   Metric                    Symbol    Direction   Unit           What it measures\n"
  "  ──  ────────────────────────  ────────  ─────────   ────────────   ─────────────────────────────────────\n"
  "  1.  Episode Reward            R_ep      Maximise    scalar         cumulative task completion reward\n"
  "  2.  Safety Violation Rate     V_rate    Minimise    fraction       episodes where M_F > Theta_max\n"
  "  3.  Cost per Episode          C_ep      Minimise    scalar         sum of constraint violations per ep\n"
  "  4.  Jain's Fairness Index     J_fair    Maximise    [0, 1]         workload balance across N workers\n"
  "  5.  Worker Utilisation        U_work    Maximise    %              fraction of time spent on tasks\n"
  "  6.  Max Fatigue (worst-case)  M_F,max   Minimise    [0, 1]         highest M_F across all workers\n"
  "  7.  Rest Fraction             f_rest    Monitor     %              fraction of time spent resting\n"
  "  8.  ECBF Interventions        N_ecbf    Monitor     count          how often filter clips C_nom\n"
  "  9.  Throughput                T_put     Maximise    tasks/hour     productive tasks completed per hour", fs=7.5)

# ── All 10 methods listed ──
rbox(1, 31.5, 38, 3.2, color=C_METHOD, border="#00695C", lw=1.5)
T(2, 34.5, "ALL 10 METHODS  (each trained with 5 seeds x 5M steps = 25M steps total per method)", fs=11, wt="bold", color="#00695C")

T(2, 33.7,
  "  #   Method         Components                                    Source              Seeds   Steps\n"
  "  ──  ─────────────  ────────────────────────────────────────────  ──────────────────  ─────   ─────\n"
  "  1.  HC-MARL (ours) MAPPO + ECBF + NSWF + MMICRL + 3CC-r + rep   This work           5       5M\n"
  "  2.  MAPPO          Multi-Agent PPO, centralised critic            Yu et al.           5       5M\n"
  "  3.  IPPO           Independent PPO, no parameter sharing          de Witt et al.      5       5M\n"
  "  4.  PPO-Lagrangian PPO + Lagrangian dual variable for cost        OmniSafe [28]       5       5M\n"
  "  5.  CPO            Constrained Policy Optimisation, trust region   Achiam et al. [11]  5       5M\n"
  "  6.  MAPPO-Lag      MAPPO + cost critic + lambda update            This work           5       5M\n"
  "  7.  MACPO          Multi-Agent CPO                                 SafePO              5       5M\n"
  "  8.  FOCOPS         First-Order CPO                                 OmniSafe            5       5M\n"
  "  9.  Fixed-Schedule Round-robin task assignment, no learning        Baseline            5       5M\n"
  " 10.  Greedy-Safe    Greedy assignment with safety heuristic         Baseline            5       5M", fs=7.2)
T(31, 31.8, "Total: 50 training runs", fs=9, wt="bold", color="#00695C")

# ── Comparison matrix ──
rbox(1, 28.2, 38, 3.0, color=C_EVAL, border="#4527A0", lw=1.5)
T(2, 31.0, "COMPARISON MATRIX  (10 methods x 9 metrics x 5 seeds = 450 measurement points)", fs=11, wt="bold", color="#4527A0")

T(2, 30.3,
  "              R_ep     V_rate   C_ep    J_fair   U_work   M_F,max  f_rest   N_ecbf   T_put\n"
  "              ─────    ─────    ─────   ─────    ─────    ─────    ─────    ─────    ─────\n"
  "  HC-MARL     mean±std for each cell (5 seeds)   =>  bold = best, underline = 2nd best\n"
  "  MAPPO       ...\n"
  "  IPPO        ...     (full results in Table 1 of paper: tables/table1_main_results.tex)\n"
  "  PPO-Lag     ...\n"
  "  ...9 more rows...", fs=7.5)

T(27, 28.5, "Format: mean +/- std across 5 seeds\n"
  "Bold best, underline 2nd best", fs=8, color="#4527A0", wt="bold")

# ── Runs / Seeds annotation ── (addressing "r1 r2 ... r15")
rbox(1, 26.5, 38, 1.4, color="#FFF3E0", border="#E65100")
T(2, 27.7, "RUNS & SEEDS  (addressing repeatability)", fs=10, wt="bold", color="#E65100")
T(2, 27.1,
  "Each method:  seed_0, seed_1, seed_2, seed_3, seed_4   (5 independent runs with different random seeds)\n"
  "Report:       mean +/- std across 5 seeds for every metric  |  Total runs: 10 methods x 5 seeds = 50 main runs", fs=8.5)

# ====================================================================
#  SECTION 6: ABLATION STUDIES  (addressing C1, C2, etc.)
# ====================================================================
heading(20, 25.8, "ABLATION STUDIES  (isolate contribution of each component)", fs=13)
hline(25.4)

rbox(1, 22.5, 38, 2.7, color="#FBE9E7", border="#BF360C", lw=1.5)
T(2, 25.0, "5 ABLATIONS  (remove one component at a time, 5 seeds each = 25 ablation runs)", fs=10, wt="bold", color="#BF360C")

T(2, 24.3,
  "  Ablation           What changes                          Tests contribution of    Correction\n"
  "  ─────────────────  ────────────────────────────────────  ──────────────────────   ──────────\n"
  "  HC-MARL − ECBF     No safety filter, raw policy actions   ECBF dual-barrier QP     C2, C3\n"
  "  HC-MARL − NSWF     Round-robin allocation (no fairness)   Nash Social Welfare      C4\n"
  "  HC-MARL − MMICRL   Fixed Theta_max for all (no types)     Constraint inference     C1\n"
  "  HC-MARL − Reperf   R_eff = R always (no rest boost)       Reperfusion switch r>1   --\n"
  "  HC-MARL − Divg.Di  D_i = kappa (constant, not fatigue)    Divergent disagreement   C4", fs=8)

# ====================================================================
#  SECTION 7: SCALING STUDY + SAFETY-GYM
# ====================================================================
heading(20, 22.0, "SCALING STUDY  +  EXTERNAL VALIDATION (Safety-Gymnasium)", fs=13)
hline(21.6)

rbox(1, 19.5, 18, 1.8, color="#E0F2F1", border="#00695C")
T(1.3, 21.1, "SCALING ANALYSIS", fs=10, wt="bold", color="#00695C")
T(1.3, 20.5,
  "N = {3, 4, 6, 8, 12} workers\n"
  "5 seeds each = 25 scaling runs\n"
  "Metrics vs N with error bands", fs=8.5)

rbox(20, 19.5, 19, 1.8, color="#E0F2F1", border="#00695C")
T(20.3, 21.1, "SAFETY-GYM VALIDATION", fs=10, wt="bold", color="#00695C")
T(20.3, 20.5,
  "2 envs: PointGoal, AntVelocity\n"
  "4 methods x 2 envs x 5 seeds = 40 runs\n"
  "External benchmark comparison", fs=8.5)

T(1, 19.1, "Grand total: 50 (main) + 25 (ablation) + 25 (scaling) + 40 (safety-gym) = 140 training runs", fs=9, wt="bold", color=C_RED)

# ====================================================================
#  SECTION 8: PAPER DELIVERABLES
# ====================================================================
hline(18.5)
heading(20, 18.1, "PAPER DELIVERABLES  ·  NeurIPS 2026 Format", fs=13)

rbox(1, 15.5, 12, 2.2, color="#DCEDC8")
T(1.3, 17.5, "6 FIGURES", fs=10, wt="bold")
T(1.3, 16.9,
  "Fig 1: Training curves (R,C vs t)\n"
  "Fig 2: Main 10-method bar chart\n"
  "Fig 3: Ablation comparison\n"
  "Fig 4: Scaling (metrics vs N)\n"
  "Fig 5: M_F(t) trajectories\n"
  "Fig 6: Pareto frontier", fs=8)

rbox(14, 15.5, 12, 2.2, color="#DCEDC8")
T(14.3, 17.5, "3 TABLES", fs=10, wt="bold")
T(14.3, 16.9,
  "Table 1: 10 methods x 9 metrics\n"
  "  (mean +/- std, bold best)\n"
  "Table 2: 6 ablation variants\n"
  "  (full vs each component removed)\n"
  "Table 3: Safety-Gym validation\n"
  "  (4 methods x 2 envs)", fs=8)

rbox(27, 15.5, 12, 2.2, color="#DCEDC8")
T(27.3, 17.5, "PAPER STRUCTURE", fs=10, wt="bold")
T(27.3, 16.9,
  "Sec 1: Introduction\n"
  "Sec 2: Related work (4 areas)\n"
  "Sec 3-6: Method (4 modules)\n"
  "Sec 7: Pipeline + interface\n"
  "Sec 8: Experiments + results\n"
  "Sec 9: Discussion + conclusion", fs=8)

# ====================================================================
#  SECTION 9: NOVELTY vs BORROWED
# ====================================================================
hline(15.0)
heading(20, 14.6, "NOVELTY (5 CORRECTIONS)  vs  BORROWED (7 SOURCE FAMILIES)", fs=13)

rbox(1, 11.2, 18.5, 3.0, color=C_NOVEL, border=C_RED, lw=1.5)
T(1.3, 14.0, "THIS WORK (NOVEL)", fs=11, wt="bold", color=C_RED)
T(1.3, 13.3,
  "C1. MMICRL: l1!=l2 residual exposed\n"
  "C2. Nagumo rest-phase proof (Thm 5.7)\n"
  "    replaces flawed first-derivative arg\n"
  "C3. Dual barrier h1+h2 (resting floor)\n"
  "C4. NSWF framing + divergent D_i -> inf\n"
  "C5. Action-to-neural-drive interface\n"
  "    (Section 7, Eqs 34-35, new)", fs=8.5)

rbox(20.5, 11.2, 18.5, 3.0, color=C_SOURCE, border="#7B1FA2", lw=1.5)
T(20.8, 14.0, "BORROWED (WITH ATTRIBUTION)", fs=11, wt="bold", color="#7B1FA2")
T(20.8, 13.3,
  "3CC-r ODE:    Liu et al. [1] (2002)\n"
  "Submaximal:   Xia & Frey-Law [2] (2008)\n"
  "Parameters:   Frey-Law et al. [3,6]\n"
  "Reperfusion:  Looft et al. [4,5]\n"
  "ECBF theory:  Nguyen [12], Xiao [13]\n"
  "CBF/QP:       Ames et al. [14,15]\n"
  "MMICRL base:  Qiao [8], MaxEnt [7]\n"
  "NSWF axioms:  Nash [17], Kaneko [21]", fs=8.5)

# ====================================================================
#  SECTION 10: THREE SAFETY GUARANTEES
# ====================================================================
hline(10.8)
heading(20, 10.4, "THREE INDEPENDENT SAFETY MECHANISMS  (no single point of failure)", fs=12)

rbox(1, 7.7, 12, 2.4, color=C_SAFETY, border=C_RED)
T(1.3, 9.9, "Mechanism 1:", fs=10, wt="bold", color=C_RED)
T(1.3, 9.4, "ECBF QP Filter", fs=9, wt="bold")
T(1.3, 8.8, "psi1_dot + a2*psi1 >= 0\nenforced at EVERY timestep\nGuarantee: M_F <= Theta_max\nper muscle, per worker", fs=7.5)

rbox(14, 7.7, 12, 2.4, color=C_SAFETY, border=C_RED)
T(14.3, 9.9, "Mechanism 2:", fs=10, wt="bold", color=C_RED)
T(14.3, 9.4, "Nagumo Invariance", fs=9, wt="bold")
T(14.3, 8.8, "Thm 5.7: during rest,\nS={x: M_F<=Theta_max}\nis forward-invariant\nif Theta_max >= F/(F+Rr)", fs=7.5)

rbox(27, 7.7, 12, 2.4, color=C_SAFETY, border=C_RED)
T(27.3, 9.9, "Mechanism 3:", fs=10, wt="bold", color=C_RED)
T(27.3, 9.4, "NSWF Rest Forcing", fs=9, wt="bold")
T(27.3, 8.8, "D_i -> inf as M_F -> 1\nNo finite U overcomes D_i\nPlanner auto-assigns rest\nBurnout impossible", fs=7.5)

# ====================================================================
#  LEGEND
# ====================================================================
ax.plot([1, 39], [7.2, 7.2], color=C_BORDER, lw=2)
T(20, 6.8, "LEGEND", fs=11, ha="center", wt="bold", color=C_TITLE, fam="sans-serif")

legend_data = [
    ("Constant (pre-set)", C_CONST),
    ("Variable (changes)", C_VAR),
    ("Novel (this work)", C_NOVEL),
    ("Borrowed (cited)", C_SOURCE),
    ("Dimension info", C_INPUT),
    ("Safety-critical", C_SAFETY),
    ("Pipeline step", C_ONLINE),
    ("Output", C_OUTPUT),
    ("Evaluation", C_EVAL),
    ("Performance", C_PERF),
]
for i, (label, color) in enumerate(legend_data):
    lx = 1 + (i % 5) * 7.5
    ly = 5.8 - (i // 5) * 0.7
    rbox(lx, ly, 3.0, 0.55, color=color)
    T(lx + 3.3, ly + 0.3, label, fs=7.5, va="center", fam="sans-serif")

# Dotted = optional, Solid = mandatory
ax.plot([1, 3], [4.7, 4.7], color=C_BORDER, lw=1.5, ls="-")
T(3.3, 4.8, "Solid border = mandatory path", fs=7.5, va="center", fam="sans-serif")
ax.plot([12, 14], [4.7, 4.7], color=C_BORDER, lw=1.5, ls="--")
T(14.3, 4.8, "Dashed border = optional/periodic", fs=7.5, va="center", fam="sans-serif")

# Footer
T(20, 4.0,
  "HC-MARL v12  ·  175 unit tests  ·  389 total files  ·  140 training runs  ·  "
  "28 references  ·  Feb-Apr 2026",
  fs=10, ha="center", color=C_GREY, fam="sans-serif")

T(20, 3.4,
  "Corrections from supervisor review incorporated:  explicit data sources  ·  all baselines named  ·  "
  "all metrics listed  ·  task sequence shown  ·  runs/seeds explicit  ·  outputs prioritised",
  fs=8, ha="center", color=C_GREY, fam="sans-serif")

# ── save ────────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
out = "figures/hcmarl_architecture_diagram.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG, pad_inches=0.5)
print(f"Saved: {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
plt.close(fig)
