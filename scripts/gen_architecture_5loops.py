"""
Generate 5 architecture diagrams (one per loop) for HC-MARL.
Each diagram is a clean column-wise flow (left-to-right).
Where an inner loop runs, it appears as a labelled sub-row.

Output: diagrams/loop_L0.png ... diagrams/loop_L4.png
        diagrams/loop_overview.png (master nesting diagram)
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

OUT = r"C:\Users\admin\Desktop\hcmarl_project\diagrams"


def draw_box(ax, cx, cy, w, h, text, fill="#FFFFFF", stroke="#333",
             fontsize=9, bold_title=None, text_color="#111"):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=3,rounding_size=6",
        facecolor=fill, edgecolor=stroke, linewidth=1.4))
    if bold_title:
        ax.text(cx, cy - h/2 + 14, bold_title, fontsize=fontsize + 1,
                ha="center", va="top", fontweight="bold", color=text_color)
        ax.text(cx, cy + 4, text, fontsize=fontsize,
                ha="center", va="center", color=text_color, linespacing=1.35)
    else:
        ax.text(cx, cy, text, fontsize=fontsize,
                ha="center", va="center", color=text_color, linespacing=1.35)


def h_arrow(ax, x1, y, x2, color="#222", lw=1.8, label=None, label_above=True):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if label:
        yoff = -12 if not label_above else 10
        ax.text((x1 + x2)/2, y - yoff, label, fontsize=8, ha="center",
                va="bottom" if label_above else "top", color=color,
                fontstyle="italic")


def v_arrow(ax, x, y1, y2, color="#222", lw=1.8, label=None):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if label:
        ax.text(x + 8, (y1 + y2)/2, label, fontsize=8, ha="left",
                va="center", color=color, fontstyle="italic")


def return_arrow(ax, x_start, y_start, x_end, y_end, y_offset,
                 color="#C62828", lw=1.6, label=None):
    """Right-angle return arrow going down then left then up."""
    y_mid = y_start + y_offset
    xs = [x_start, x_start, x_end, x_end]
    ys = [y_start, y_mid, y_mid, y_end]
    for i in range(len(xs) - 2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=color, lw=lw)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if label:
        ax.text((xs[1] + xs[2])/2, y_mid - 10, label, fontsize=8,
                ha="center", va="bottom", color=color, fontstyle="italic")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.close(fig)
    print(f"  Wrote: {path}")


# ============================================================
# LOOP 0: Offline Calibration + MMICRL
# ============================================================
def draw_L0():
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.set_xlim(0, 1400)
    ax.set_ylim(400, 0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(700, 25, "L0: Offline Calibration & Constraint Inference  (runs once before training)",
            fontsize=13, fontweight="bold", ha="center", color="#6A1B9A")

    # Column 1: Real Data
    draw_box(ax, 130, 150, 220, 120,
             "WSD4FESDRM dataset\n34 subjects\nBorg-RPE-20 shoulder\n30-60% MVIC\nZenodo 8415066",
             fill="#FFF8E1", stroke="#F57F17",
             bold_title="1. Real Dataset")

    # Column 2: Calibration
    draw_box(ax, 420, 150, 240, 120,
             "Grid-search (F, R) per subject\nminimise |ET_obs - ET_pred|\nforward-simulate 3CC-r ODE\n"
             "Output: (F, R, r) per muscle\n[Table 1, Eq.6]",
             fill="#FFF8E1", stroke="#F57F17",
             bold_title="2. 3CC-r Calibration (Path G)")

    # Column 3: Demo generation
    draw_box(ax, 700, 150, 210, 120,
             "Run calibrated ODE\nper subject at %MVIC\nCollect tau = {(o_i, a_i)}\nPer-agent rollouts",
             fill="#FFF8E1", stroke="#F57F17",
             bold_title="3. Demo Collection")

    # Column 4: MMICRL
    draw_box(ax, 990, 150, 260, 120,
             "EM loop (n_iterations epochs):\n  M-step: train CFDE flow p(s,a|z)\n  E-step: reassign z via p(z|tau)\n"
             "CFDE = MADE/MAF norm. flows\nlambda_1 = lambda_2 => pure MI\n[Eq.9-11]",
             fill="#FFF8E1", stroke="#F57F17",
             bold_title="4. MMICRL Constraint Inference")

    # Column 5: Output
    draw_box(ax, 1280, 150, 200, 120,
             "Per-type theta_max(z)\nPersonalised safety\nceilings per muscle\n"
             "Constraint network (BCE)\nType assignments",
             fill="#E8F5E9", stroke="#2E7D32",
             bold_title="5. Output to L1-L4")

    # Arrows
    h_arrow(ax, 240, 150, 300, label="ET_obs, %MVIC")
    h_arrow(ax, 540, 150, 595, label="(F, R, r)")
    h_arrow(ax, 805, 150, 860, label="tau demos")
    h_arrow(ax, 1120, 150, 1180, label="theta_max(z)")

    # EM loop annotation
    ax.add_patch(FancyBboxPatch((870, 80), 250, 155,
                                boxstyle="round,pad=0", facecolor="none",
                                edgecolor="#6A1B9A", linewidth=1.2,
                                linestyle="--"))
    ax.text(995, 78, "EM loop", fontsize=9, ha="center", color="#6A1B9A",
            fontstyle="italic")

    # Param table
    ax.text(130, 290, "Calibrated parameters [Table 1, Frey-Law+ 2012]:", fontsize=9,
            fontweight="bold")
    params = (
        "Shoulder: F=0.01820, R=0.00168, r=15  |  Ankle: F=0.00589, R=0.00058, r=15  |  "
        "Knee: F=0.01500, R=0.00149, r=15\n"
        "Elbow: F=0.00912, R=0.00094, r=15  |  Trunk: F=0.00755, R=0.00075, r=15  |  "
        "Grip: F=0.00980, R=0.00064, r=30"
    )
    ax.text(130, 320, params, fontsize=8.5, family="monospace")

    # Section refs
    ax.text(130, 370, "Math doc refs:  data = Sci. Data 2024 (Zenodo 8415066)  |  "
            "calibration = Section 3.5  |  MMICRL = Section 4, Eq.9-11  |  "
            "CFDE = Qiao+ NeurIPS 2023", fontsize=8, color="#555")

    save(fig, "loop_L0_calibration.png")


# ============================================================
# LOOP 1: Training Loop (outermost online)
# ============================================================
def draw_L1():
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    ax.set_xlim(0, 1400)
    ax.set_ylim(550, 0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(700, 25, "L1: Training Loop  (while global_step < 5M)",
            fontsize=13, fontweight="bold", ha="center", color="#1565C0")

    # Row of columns
    draw_box(ax, 130, 160, 200, 110,
             "From L0:\ntheta_max(z) per worker\n(F, R, r) per muscle\nConstraint net weights",
             fill="#E3F2FD", stroke="#1565C0",
             bold_title="Inputs")

    draw_box(ax, 380, 160, 200, 110,
             "env.reset()\nInit MR=1, MA=0, MF=0\nper worker per muscle\nobs, global_state",
             fill="#E3F2FD", stroke="#1565C0",
             bold_title="Episode Reset")

    draw_box(ax, 660, 160, 240, 110,
             "L2: Episode loop\n(max_steps env steps)\n\n"
             "Contains L3 (NSWF+ECBF)\nSee diagram L2 for detail",
             fill="#BBDEFB", stroke="#0D47A1",
             bold_title="L2: Run Episode")

    draw_box(ax, 950, 160, 240, 110,
             "L4: PPO update\n10 epochs x minibatches\n\n"
             "Actor + Value critic +\nCost critic + lambda update\nSee diagram L4 for detail",
             fill="#C8E6C9", stroke="#2E7D32",
             bold_title="L4: Policy Update")

    draw_box(ax, 1250, 160, 200, 110,
             "Log 9 metrics\nCheckpoint if best\nEval if interval hit\nglobal_step += ep_steps",
             fill="#E3F2FD", stroke="#1565C0",
             bold_title="Log & Checkpoint")

    # Forward arrows
    h_arrow(ax, 230, 160, 280, label="params")
    h_arrow(ax, 480, 160, 540, label="obs_0")
    h_arrow(ax, 780, 160, 830, label="buffer full")
    h_arrow(ax, 1070, 160, 1150, label="updated theta")

    # Return arrow: checkpoint back to episode reset
    return_arrow(ax, 1250, 215, 380, 215, 120, color="#C62828",
                 label="next episode (if global_step < 5M)")

    # Episode count
    ax.text(700, 400, "Repeats: ~episodes until 5M total env steps across all episodes\n"
            "5 seeds run independently (separate train.py invocations)",
            fontsize=9, ha="center", color="#555")

    # Nesting callout
    ax.add_patch(FancyBboxPatch((540, 98), 460, 140,
                                boxstyle="round,pad=0", facecolor="none",
                                edgecolor="#D32F2F", linewidth=1.3,
                                linestyle="--"))
    ax.text(770, 96, "Inner loops L2, L3, L4 (see separate diagrams)",
            fontsize=9, ha="center", color="#D32F2F", fontstyle="italic")

    ax.text(130, 480, "Math doc refs:  training pipeline = Section 7.3 (steps 1-7)  |  "
            "PPO = Achiam+ 2017 (CMDP)  |  9 metrics = O1-O9",
            fontsize=8, color="#555")

    save(fig, "loop_L1_training.png")


# ============================================================
# LOOP 2: Episode Loop (env steps)
# ============================================================
def draw_L2():
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=150)
    ax.set_xlim(0, 1500)
    ax.set_ylim(650, 0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(750, 25, "L2: Episode Loop  (for step in range(max_steps), inside L1)",
            fontsize=13, fontweight="bold", ha="center", color="#00695C")

    y_main = 170

    # Col 1: Observe
    draw_box(ax, 120, y_main, 190, 130,
             "Read x_i(t) per worker\n[MR, MA, MF] x 6 muscles\nMR+MA+MF = 1  [Eq.1]\n\n"
             "o_i in R^19 (local)\ns in R^{18N+1} (global)\nN=6 => R^109",
             fill="#E0F2F1", stroke="#00695C",
             bold_title="1. Observe")

    # Col 2: Allocate (L3 sub-row)
    draw_box(ax, 360, y_main, 210, 130,
             "L3: NSWF Allocator\nmax sum_i ln(U(i,j*(i))-D_i)\nD_i = kappa*MF^2/(1-MF)\n"
             "[Eq.33, Eq.32, Eq.31]\nRuns if should_reallocate()\n"
             "=> j*(i) per worker",
             fill="#FFF3E0", stroke="#E65100",
             bold_title="2. Allocate (L3)")

    # Col 3: Load translate
    draw_box(ax, 575, y_main, 180, 130,
             "j*(i) => T_L^(j) in [0,1]^6\n6 tasks x 6 muscles\nlookup table\n[Eq.34]",
             fill="#E0F2F1", stroke="#00695C",
             bold_title="3. Load Translate")

    # Col 4: Actor
    draw_box(ax, 770, y_main, 190, 130,
             "Decentralised actor\npi_theta(C | o_i, task)\n2x64 Tanh, o_i in R^19\n\n"
             "Baseline: C_nom =\nk_p * (T_L - MA)_+\nRL replaces baseline [Eq.35]",
             fill="#E0F2F1", stroke="#00695C",
             bold_title="4. Actor => C_nom")

    # Col 5: ECBF (part of L3)
    draw_box(ax, 990, y_main, 220, 130,
             "CBF-QP per muscle per worker:\nmin ||C - C_nom||^2  s.t.\n"
             "(i) h=Theta_max-MF >= 0 [rd 2]\n(ii) h2=MR >= 0 [rd 1]\n"
             "alpha_1,2,3 = 0.5 (config)\nTheta_max >= F/(F+Rr) [Eq.25]",
             fill="#FFF3E0", stroke="#E65100",
             bold_title="5. ECBF Filter (L3)")

    # Col 6: ODE
    draw_box(ax, 1230, y_main, 210, 130,
             "Euler integrate 3CC-r:\ndMA/dt = C* - F*MA  [Eq.2]\n"
             "dMF/dt = F*MA - R_eff*MF [3]\ndMR/dt = R_eff*MF - C* [4]\n"
             "R_eff = R (work) | R*r (rest)\ndt = 1 min",
             fill="#E0F2F1", stroke="#00695C",
             bold_title="6. ODE => x(t+1)")

    # Row 2: Reward/Cost
    y2 = 380
    draw_box(ax, 400, y2, 290, 90,
             "R_t = sum_i ln( U(i,j*(i)) - D_i(MF_i) )  [Eq.33]\nNSWF log-surplus by construction\n"
             "C_t = sum_{i,g} max(0, MF_{i,g} - Theta_{max,i,g})",
             fill="#E8F5E9", stroke="#2E7D32",
             bold_title="7. Reward R_t & Cost C_t")

    draw_box(ax, 800, y2, 250, 90,
             "Store (o_i, s, a, log_pi, R, C, done)\nin per-agent rollout buffer\nfor PPO update at episode end",
             fill="#E8F5E9", stroke="#2E7D32",
             bold_title="8. Buffer Store")

    # Arrows row 1
    h_arrow(ax, 215, y_main, 255, label="x_i(t)")
    h_arrow(ax, 465, y_main, 485, label="j*(i)")
    h_arrow(ax, 665, y_main, 675, label="T_L")
    h_arrow(ax, 865, y_main, 880, label="C_nom")
    h_arrow(ax, 1100, y_main, 1125, label="C*")

    # ODE down to reward
    v_arrow(ax, 1230, y_main + 65, y2 - 45, color="#222", label="x(t+1), MF")

    # Reward to buffer
    h_arrow(ax, 545, y2, 675, label="R_t, C_t")

    # Return arrow: ODE output back to observe
    return_arrow(ax, 1335, y_main, 120, y_main, 250, color="#C62828",
                 label="Loop 2 feedback: x(t+1) => next step observation")

    # L3 highlight
    ax.add_patch(FancyBboxPatch((253, y_main - 80), 870, 195,
                                boxstyle="round,pad=0", facecolor="none",
                                edgecolor="#E65100", linewidth=1.5,
                                linestyle="--"))
    ax.text(688, y_main - 82, "L3: Control pipeline (NSWF allocate + ECBF filter) — runs every env step",
            fontsize=9, ha="center", color="#E65100", fontstyle="italic")

    # Buffer to L4 annotation
    ax.text(800, y2 + 70, "=> Buffer sent to L4 (PPO update) at episode end",
            fontsize=9, ha="center", color="#2E7D32", fontstyle="italic")

    ax.text(120, 540, "Math doc refs:  observation = Sec 3.1  |  allocation = Sec 6, Eq.31-33  |  "
            "load = Sec 7.1, Eq.34  |  actor = Sec 7.2, Eq.35  |  ECBF = Sec 5, Eq.18-23  |  "
            "ODE = Sec 3.2, Eq.2-5", fontsize=8, color="#555")

    ax.text(120, 570, "Key: delta_max = R/(F+R) [sustainability bound, Eq.6] is NOT Theta_max.  "
            "Theta_max is a design parameter with lower bound F/(F+Rr) [Def. 5.4, Eq.25].",
            fontsize=8, color="#C62828", fontweight="bold")

    save(fig, "loop_L2_episode.png")


# ============================================================
# LOOP 3: Control Pipeline (NSWF + ECBF per step)
# ============================================================
def draw_L3():
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.set_xlim(0, 1400)
    ax.set_ylim(500, 0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(700, 25, "L3: Control Pipeline — NSWF Allocation + ECBF Safety  (per env step, inside L2)",
            fontsize=12, fontweight="bold", ha="center", color="#E65100")

    y = 160

    # Col 1: fatigue read
    draw_box(ax, 120, y, 180, 120,
             "Read MF_i per worker\nfrom env state\nmax(MF) per worker\nfor allocator input",
             fill="#FFF3E0", stroke="#E65100",
             bold_title="1. Fatigue Read")

    # Col 2: NSWF
    draw_box(ax, 340, y, 220, 120,
             "max sum_i ln(S_i)\nS_i = U(i,j*(i)) - D_i(MF_i)\nD_i = kappa*MF^2/(1-MF) [Eq.32]\n"
             "U(i,0) = D_i + eps [Eq.31]\nAssignment: 1 task per worker\nMultiple workers can rest",
             fill="#FFF3E0", stroke="#E65100",
             bold_title="2. NSWF Solve [Eq.33]")

    # Col 3: Load
    draw_box(ax, 560, y, 160, 120,
             "j*(i) => T_L^(j)_g\nper-muscle MVC\nfraction [Eq.34]\n6 muscles x demand",
             fill="#FFF3E0", stroke="#E65100",
             bold_title="3. Load Map")

    # Col 4: nominal drive
    draw_box(ax, 740, y, 170, 120,
             "C_nom = pi_theta(o_i)\nor baseline:\nk_p*(T_L - MA)_+\nk_p = 1.0 [Eq.35]",
             fill="#FFF3E0", stroke="#E65100",
             bold_title="4. Nominal Drive")

    # Col 5: ECBF QP
    draw_box(ax, 970, y, 240, 120,
             "Per muscle g, per worker i:\nmin ||C - C_nom||^2  s.t.\n"
             "Fatigue ceiling [Eq.18]:\n  ECBF rel-deg 2 constraint\n"
             "Resting floor [Eq.23]:\n  CBF rel-deg 1 constraint\nC >= 0",
             fill="#FFF3E0", stroke="#E65100",
             bold_title="5. ECBF-QP [Eq.20]")

    # Col 6: output
    draw_box(ax, 1250, y, 180, 120,
             "C* (safe drive)\nper muscle per worker\n\n"
             "If QP infeasible:\nC* = 0 (forced rest)\n[Remark 5.13]",
             fill="#E8F5E9", stroke="#2E7D32",
             bold_title="6. Safe Output C*")

    # Arrows
    h_arrow(ax, 210, y, 230, label="MF_i")
    h_arrow(ax, 450, y, 480, label="j*(i)")
    h_arrow(ax, 640, y, 655, label="T_L")
    h_arrow(ax, 825, y, 850, label="C_nom")
    h_arrow(ax, 1090, y, 1160, label="C*")

    # Design constraint box
    draw_box(ax, 970, 340, 340, 70,
             "Design constraint [Def. 5.4]:  Theta_max >= F/(F+Rr)\n"
             "Shoulder: 62.7%  |  Ankle: 2.1%  |  Knee: 36.4%\n"
             "Safety across mode transitions: Nagumo invariance [Thm 5.7]",
             fill="#FFEBEE", stroke="#C62828", fontsize=8.5)

    v_arrow(ax, 970, 305, 340, color="#C62828", label="")

    # NSWF properties
    draw_box(ax, 340, 340, 260, 70,
             "Nash Social Welfare (not Nash Bargaining) [Kaneko+ 1979]\n"
             "D_i diverges as MF -> 1: math guarantee against burnout\n"
             "Rest option: ln(eps) << 0, strongly disfavoured but feasible",
             fill="#FFF8E1", stroke="#F57F17", fontsize=8.5)

    ax.text(120, 450, "Math doc refs:  NSWF = Sec 6, Eq.31-33  |  ECBF = Sec 5, Eq.12-30  |  "
            "dual barrier = Sec 5.3  |  rest-phase safety = Thm 5.7  |  QP = Boyd 2004, CVXPY/OSQP",
            fontsize=8, color="#555")

    save(fig, "loop_L3_control.png")


# ============================================================
# LOOP 4: PPO-Lagrangian Update
# ============================================================
def draw_L4():
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    ax.set_xlim(0, 1400)
    ax.set_ylim(550, 0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(700, 25, "L4: MAPPO-Lagrangian Policy Update  (10 epochs, end of each episode in L1)",
            fontsize=12, fontweight="bold", ha="center", color="#2E7D32")

    y = 170

    # Col 1: Buffer
    draw_box(ax, 120, y, 180, 130,
             "Rollout buffer\n(o_i, s, a, log_pi,\nR_t, C_t, done)\nfrom L2 episode\n\n"
             "Flat: T*N samples\n(T=steps, N=workers)",
             fill="#C8E6C9", stroke="#2E7D32",
             bold_title="1. Buffer")

    # Col 2: GAE
    draw_box(ax, 340, y, 210, 130,
             "GAE advantages:\ndelta_t = R_t + gamma*V(s_{t+1}) - V(s_t)\nA_t = sum (gamma*lam)^k delta_{t+k}\n"
             "gamma = 0.99, lam_GAE = 0.95\n\nSame for cost advantages\n(separate cost-critic V_c)",
             fill="#C8E6C9", stroke="#2E7D32",
             bold_title="2. Compute GAE")

    # Col 3: PPO epoch (inner loop)
    draw_box(ax, 620, y, 260, 130,
             "For each of 10 epochs:\n"
             "  ratio r = pi_new / pi_old\n"
             "  L_actor = -min(r*A, clip(r)*A)\n"
             "            + lam * max(cost surr)\n"
             "            - 0.01 * H(pi)\n"
             "  Backprop + grad clip (0.5)\n"
             "  batch=256, lr=3e-4",
             fill="#A5D6A7", stroke="#1B5E20",
             bold_title="3. PPO Epochs (inner)")

    # Col 4: Critics
    draw_box(ax, 920, y, 210, 130,
             "Value critic V(s):\n  2x128 Tanh => scalar\n  L_critic = MSE(V, returns)\n\n"
             "Cost critic V_c(s):\n  2x128 Tanh => scalar\n  L_ccrit = MSE(V_c, cost_ret)",
             fill="#C8E6C9", stroke="#2E7D32",
             bold_title="4. Critic Updates")

    # Col 5: Lambda
    draw_box(ax, 1170, y, 210, 130,
             "Dual variable lambda:\nlog_lam in nn.Parameter\n\n"
             "cost_ema = EMA of ep cost\n"
             "delta_lam ~ -(cost_ema - budget)\nbudget = 0.10\n\n"
             "lam UP if MF > Theta_max\nlam DOWN otherwise",
             fill="#C8E6C9", stroke="#2E7D32",
             bold_title="5. Lambda Update")

    # Arrows
    h_arrow(ax, 210, y, 235, label="transitions")
    h_arrow(ax, 445, y, 490, label="A_t, returns")
    h_arrow(ax, 750, y, 815, label="actor grad")
    h_arrow(ax, 1025, y, 1065, label="V, V_c")

    # Return to actor
    return_arrow(ax, 1170, y + 65, 620, y + 65, 130, color="#C62828",
                 label="Loop 4: 10 epochs (n_epochs iterations)")

    # Output arrow
    draw_box(ax, 700, 400, 280, 60,
             "Updated weights: theta (actor), phi (critic)\n"
             "=> back to L2 next episode (closes L1 iteration)",
             fill="#E8F5E9", stroke="#1B5E20", fontsize=9)
    v_arrow(ax, 700, y + 65, 370, color="#1B5E20", label="output")

    ax.text(120, 490, "Math doc refs:  MAPPO-Lagrangian = Achiam+ 2017 (CMDP) + PPO  |  "
            "GAE = Schulman+ 2015  |  dual ascent on lambda  |  "
            "cost surrogates: max(r1,r2) errs toward safety under ratio truncation",
            fontsize=8, color="#555")

    save(fig, "loop_L4_learning.png")


# ============================================================
# OVERVIEW: Nesting diagram
# ============================================================
def draw_overview():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.set_xlim(0, 1400)
    ax.set_ylim(700, 0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(700, 30, "HC-MARL Architecture — Loop Nesting Overview",
            fontsize=14, fontweight="bold", ha="center")
    ax.text(700, 55, "5 loops, 3 nesting levels. Each loop has its own detailed diagram.",
            fontsize=10, ha="center", color="#555")

    # L0 (offline, full width)
    ax.add_patch(Rectangle((60, 80), 1280, 80, facecolor="#FFF8E1",
                           edgecolor="#F57F17", linewidth=2, alpha=0.6))
    ax.text(700, 100, "L0: Offline Calibration & MMICRL  (runs once)",
            fontsize=12, fontweight="bold", ha="center", color="#E65100")
    ax.text(700, 125, "Real Data -> 3CC-r Calibration (Path G) -> Demo Collection -> MMICRL (EM) -> "
            "theta_max(z), (F,R,r)", fontsize=9, ha="center")

    # Arrow L0 -> L1
    v_arrow(ax, 700, 160, 195, color="#333", label="params")

    # L1 (training, full width)
    ax.add_patch(Rectangle((60, 200), 1280, 450, facecolor="#E3F2FD",
                           edgecolor="#1565C0", linewidth=2, alpha=0.4))
    ax.text(90, 220, "L1: Training Loop  (while global_step < 5M, 5 seeds)",
            fontsize=12, fontweight="bold", color="#1565C0")

    # Inside L1: episode reset
    draw_box(ax, 200, 290, 180, 50, "Episode Reset\nobs_0, global_state_0",
             fill="#BBDEFB", stroke="#1565C0", fontsize=9)

    # L2 box (env steps)
    ax.add_patch(Rectangle((320, 260), 700, 220, facecolor="#E0F2F1",
                           edgecolor="#00695C", linewidth=2, alpha=0.5))
    ax.text(340, 278, "L2: Episode Loop  (for step in max_steps)",
            fontsize=11, fontweight="bold", color="#00695C")

    # L3 inside L2
    ax.add_patch(Rectangle((340, 300), 660, 80, facecolor="#FFF3E0",
                           edgecolor="#E65100", linewidth=1.5, alpha=0.6))
    ax.text(360, 315, "L3: Control Pipeline (per env step)",
            fontsize=10, fontweight="bold", color="#E65100")
    ax.text(670, 350,
            "Observe x_i -> NSWF allocate j*(i) -> Load T_L -> Actor C_nom -> ECBF-QP C* -> ODE x(t+1)",
            fontsize=9, ha="center")

    # Reward/cost/buffer
    draw_box(ax, 670, 430, 350, 35,
             "R_t = NSWF log-surplus [Eq.33]  |  C_t = sum max(0, MF-Theta_max)  |  Store to buffer",
             fill="#C8E6C9", stroke="#2E7D32", fontsize=8.5)

    # L2 feedback arrow
    return_arrow(ax, 960, 340, 390, 340, 60, color="#00695C",
                 label="x(t+1) => next step")

    # Arrow L2 -> L4
    h_arrow(ax, 1020, 400, 1090, color="#333", label="buffer")

    # L4 box (PPO)
    ax.add_patch(Rectangle((1060, 260), 260, 220, facecolor="#C8E6C9",
                           edgecolor="#2E7D32", linewidth=2, alpha=0.5))
    ax.text(1080, 278, "L4: PPO-Lagrangian",
            fontsize=11, fontweight="bold", color="#2E7D32")
    ax.text(1190, 340,
            "10 PPO epochs\nActor + Critics\n+ Lambda dual ascent\n\n"
            "L_actor = -min(rA, clip*A)\n+ lam*max(cost_surr)\n- beta*H",
            fontsize=8.5, ha="center")

    # L4 return to L2 actor
    return_arrow(ax, 1190, 480, 200, 315, 130, color="#C62828",
                 label="L1 loop: next episode with updated theta")

    # Metrics out
    draw_box(ax, 200, 570, 250, 50,
             "9-Metric Logger: O1-O9\n(violation, safety, fairness, tasks, reward, fatigue, ...)",
             fill="#FAFAFA", stroke="#555", fontsize=8.5)
    v_arrow(ax, 670, 447, 570, color="#555", label="ep metrics")

    # Footer
    ax.text(120, 660,
            "Section refs:  L0 = Sec 3.5, 4  |  L2 = Sec 7.3 (pipeline steps 1-7)  |  "
            "L3 = Sec 5-6  |  L4 = CMDP (Achiam+ 2017)  |  "
            "All errors from v1 corrected: Theta_max vs delta_max, Trunk F, reward log-surplus, global-state dim",
            fontsize=8, color="#555")

    save(fig, "loop_overview.png")


# ============================================================
if __name__ == "__main__":
    print("Generating 5-loop architecture diagrams...")
    draw_L0()
    draw_L1()
    draw_L2()
    draw_L3()
    draw_L4()
    draw_overview()
    print("Done. All PNGs in diagrams/")
