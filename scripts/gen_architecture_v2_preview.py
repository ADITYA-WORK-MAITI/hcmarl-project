"""Render a PNG preview of HC_MARL_ARCHITECTURE_v2.drawio using matplotlib.

Layout mirrors the drawio file 1:1 so teacher/user can validate topology
before opening draw.io. Canvas is 1500x900 px at 100% zoom.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches

W, H = 1500, 900
fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=150)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # flip y so draw.io coords match
ax.set_aspect("equal")
ax.axis("off")


def panel(x, y, w, h, face, edge, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                           linewidth=1.2, alpha=0.35))
    ax.text(x + 8, y + 14, label, fontsize=9, fontweight="bold",
            color=edge, va="center")


def box(x, y, w, h, text, stroke="#333", fill="#FFFFFF", fontsize=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=2,rounding_size=4",
                                facecolor=fill, edgecolor=stroke, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fontsize,
            ha="center", va="center", wrap=True)


def arrow(x1, y1, x2, y2, color="#222", lw=1.3, ls="-", label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + 4, (y1 + y2) / 2, label, fontsize=7,
                color=color, va="center")


def rail(points, color, ls, label=None, label_xy=None):
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if i == len(points) - 2:
            arrow(x1, y1, x2, y2, color=color, lw=1.6, ls=ls)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.6,
                    linestyle=ls)
    if label and label_xy:
        ax.text(*label_xy, label, fontsize=8, fontstyle="italic",
                color=color, ha="center")


# Title
ax.text(W / 2, 20, "HC-MARL · End-to-End Architecture  (v2, math-verified)",
        fontsize=13, fontweight="bold", ha="center")

# Phase panels
panel(260, 40, 520, 220, "#FFF5CC", "#B8860B",
      "CALIBRATION  (offline, pre-training)")
panel(260, 270, 520, 400, "#E3F0FF", "#1F4E79",
      "CONTROL / ENV LOOP  (Loop 1: per-step)")
panel(260, 680, 520, 180, "#E6F5E1", "#2E7D32",
      "LEARNING  (Loop 2: per-batch)")

# Spine boxes (x=285, w=470)
BX, BW = 285, 470
rows = [
    # (y, h, text, stroke)
    (60, 55,
     "1. Real Dataset · WSD4FESDRM\n34 subj · Borg-RPE-20 shoulder rotation · 30–60% MVIC\nZenodo 8415066 · Sci. Data 2024",
     "#B8860B"),
    (125, 65,
     "2. 3CC-r Calibration (Path G)   §3.5 · Eq.(6)\nGrid-search (F,R) per subject vs ET_obs\n"
     "Shoulder 0.01820/0.00168 · Ankle 0.00589/0.00058 · Knee 0.01500/0.00149\n"
     "Elbow 0.00912/0.00094 · Trunk 0.00755/0.00075 · Grip 0.00980/0.00064  (r=15; grip r=30)",
     "#B8860B"),
    (200, 55,
     "3. MMICRL · Multi-Modal Inverse Constrained RL   §4 · Eq.(9)–(11)\n"
     "Obj: λ₁H[π(τ)] − λ₂H[π(τ|z)] → recommend λ₁=λ₂ (pure MI)\n"
     "CFDE flows (MADE/MAF) | z  →  personalised Θ_max(z) per worker",
     "#B8860B"),
    (290, 60,
     "4. State Observation   §3.1 · Eq.(1)\n"
     "x_i(t) = [MR, MA, MF] per muscle · MR+MA+MF = 1\n"
     "Local o_i ∈ ℝ¹⁹  ·  Global s ∈ ℝ^{18N+1} (N=6 → ℝ¹⁰⁹)",
     "#1F4E79"),
    (360, 60,
     "5. NSWF Task Allocator (central planner)   §6 · Eq.(33)\n"
     "max_{a_{ij}} Σ_i ln( U(i, j*(i)) − D_i(MF_i) ) · D_i = κ·MF²/(1−MF)  [Eq.32]\n"
     "Rest U(i,0)=D_i+ε  [Eq.31] · κ=1.0 · 6 tasks (incl. rest)",
     "#1F4E79"),
    (430, 55,
     "6. Load Translation   §7.1 · Eq.(34)\n"
     "j*(i) → T_L^(j) ∈ [0,1]⁶   (6 tasks × 6 muscles lookup)\n"
     "e.g. heavy_lift = shoulder 0.45, knee 0.40, trunk 0.50, grip 0.55, …",
     "#1F4E79"),
    (495, 55,
     "7. Decentralised Actor  π_θ(C | o_i, task)   §7.2 · Eq.(35)\n"
     "Baseline C_nom = k_p·(T_L − MA)_+ , k_p=1.0  →  RL: C_nom = f_θ(x, task)\n"
     "2×64 Tanh · input o_i ∈ ℝ¹⁹ (local only)",
     "#1F4E79"),
    (560, 60,
     "8. ECBF Safety Filter · CBF-QP (dual barrier)   §5 · Eq.(18),(20),(23)\n"
     "min_C ‖C − C_nom‖²  s.t.  (i) h = Θ_max − MF ≥ 0  (rel-deg 2)   (ii) h₂ = MR ≥ 0\n"
     "α₁=α₂=α₃=0.5 (config) · Θ_max ≥ F/(F+Rr)  [Def. 5.4, Eq.25]",
     "#1F4E79"),
    (630, 35,
     "9. 3CC-r ODE Integration  →  x(t+1)   §3.2 · Eq.(2)–(5)\n"
     "dMA/dt=C*−F·MA · dMF/dt=F·MA−R_eff·MF · dMR/dt=R_eff·MF−C*  ·  R_eff=R (work) | R·r (rest)  ·  δ_max=R/(F+R)",
     "#1F4E79"),
    (700, 55,
     "10. Reward & Safety Cost   §6,§5 · Eq.(33),(12)\n"
     "R_t = Σ_i ln( U(i, j*(i)) − D_i(MF_i) )   [= NSWF log-surplus]\n"
     "C_t = Σ_{i,g} max(0, MF_{i,g} − Θ_{max,i,g})   [→ Lagrangian]",
     "#2E7D32"),
    (765, 70,
     "11. Centralised Learner · MAPPO-Lagrangian   (Achiam+ '17, CMDP)\n"
     "V_φ(s) 2×128 Tanh · GAE γ=0.99 λ=0.95 · V_c(s) cost-critic · log λ ∈ nn.Parameter (dual ascent)\n"
     "L_actor = −min(rÂ, clip(r,1±0.2)Â) + λ·max(ĉ_r1, ĉ_r2) − 0.01·H\n"
     "batch 256 · epochs 10 · lr 3e-4 · 5M steps · 5 seeds",
     "#2E7D32"),
]
y_centres = {}
for i, (y, h, text, stroke) in enumerate(rows, start=1):
    box(BX, y, BW, h, text, stroke=stroke, fill="#FFFFFF", fontsize=7.2)
    y_centres[i] = (BX + BW / 2, y + h / 2, y, y + h)

# Forward arrows between spine boxes
for i in range(1, 11):
    _, _, _, y_bot = y_centres[i]
    _, _, y_top_next, _ = y_centres[i + 1]
    x_mid = BX + BW / 2
    arrow(x_mid, y_bot, x_mid, y_top_next, color="#222", lw=1.3)

# Left panel: logger
box(40, 290, 200, 190,
    "9-Metric Episode Logger\n\n"
    "O1 violation_rate\nO2 safety_rate\nO3 jain_fairness\n"
    "O4 tasks_completed\nO5 cum_reward\nO6 peak_fatigue\n"
    "O7 forced_rest_rate\nO8 constraint_recovery_t\nO9 cum_cost",
    stroke="#555", fill="#FAFAFA", fontsize=7.5)

box(40, 490, 200, 180,
    "Legend\n"
    "──  forward data / control\n"
    "– –  Loop 1: env-state feedback\n"
    "· · ·  Loop 2: gradient feedback\n\n"
    "Phases\n"
    "■ yellow = offline calibration\n"
    "■ blue = online env loop\n"
    "■ green = policy learning\n\n"
    "§X = math-doc section ref",
    stroke="#555", fill="#FAFAFA", fontsize=7.5)

box(40, 680, 200, 155,
    "Notes\n"
    "• α₁=α₂=α₃=0.5 in config.\n"
    "  Env currently hardcodes\n"
    "  (0.05, 0.05, 0.10) — flagged.\n"
    "• Θ_max is a DESIGN parameter\n"
    "  (not = δ_max). Lower bound:\n"
    "  Θ_max ≥ F/(F+Rr)  [Eq.25].\n"
    "• N = n_workers  (6 default)",
    stroke="#C67C00", fill="#FFF4E5", fontsize=7.5)

# Right side: Equation sheet
eq_text = (
    "Equation Sheet  (referenced by §)\n\n"
    "§3 Physiology (3CC-r)\n"
    "(1)  MR + MA + MF = 1\n"
    "(2)  dMA/dt = C − F·MA\n"
    "(3)  dMF/dt = F·MA − R_eff·MF\n"
    "(4)  dMR/dt = R_eff·MF − C\n"
    "(5)  R_eff = R (work) | R·r (rest)\n"
    "(6)  δ_max = R/(F+R) ;  C_max = FR/(F+R)\n\n"
    "§4 MMICRL\n"
    "(9)   max_π  λ₁H[π(τ)] − λ₂H[π(τ|z)]\n"
    "(10)  = λ₂ I(τ;z) + (λ₁−λ₂) H[π(τ)]\n"
    "(11)  λ₁=λ₂ ⇒ pure MI maximisation\n\n"
    "§5 ECBF (dual barrier)\n"
    "(12)  h(x) = Θ_max − MF ≥ 0\n"
    "(18)  −F·C + F²MA + R_eff·F·MA − R_eff²·MF\n"
    "         + α₁·h + α₂·ψ₁ ≥ 0   [rel-deg 2]\n"
    "(20)  C* = argmin ‖C−C_nom‖²  s.t. (18),(23),C≥0\n"
    "(21)  h₂(x) = MR = 1 − MA − MF ≥ 0\n"
    "(23)  C ≤ R_eff·MF + α₃·(1 − MA − MF)\n"
    "(25)  Θ_max ≥ F / (F + R·r)   [Def. 5.4]\n"
    "(29)  Δt* ≥ F⁻¹ · ln( F·MA(t_s) / β(t_r) )_+\n\n"
    "§6 NSWF Allocation\n"
    "(31)  U(i,0) = D_i + ε        (rest option)\n"
    "(32)  D_i(MF) = κ · MF² / (1 − MF)\n"
    "(33)  max_{a} Σ_i ln( U(i,j*(i)) − D_i(MF_i) )\n\n"
    "§7 Interface\n"
    "(34)  T_L^{(j)}_g ∈ [0,1]  (task-g MVC demand)\n"
    "(35)  C(t) = k_p · (T_L − MA)_+   (baseline)"
)
ax.add_patch(FancyBboxPatch((830, 40), 640, 820,
                            boxstyle="round,pad=2,rounding_size=4",
                            facecolor="#FFFFFF", edgecolor="#333",
                            linewidth=1.1))
ax.text(838, 48, eq_text, fontsize=7.8, family="monospace",
        va="top", ha="left")

# Feedback rails
# Loop 1 (env state): from b9 right side → up → into b4 right side
b9_cx, b9_cy, b9_y0, b9_y1 = y_centres[9]
b4_cx, b4_cy, b4_y0, b4_y1 = y_centres[4]
rail([(BX + BW, b9_cy), (800, b9_cy), (800, b4_cy), (BX + BW, b4_cy)],
     color="#1F4E79", ls="--", label="Loop 1\n(env step)",
     label_xy=(807, (b9_cy + b4_cy) / 2))

# Loop 2 (grad): from b11 right → up → into b7 right
b11_cx, b11_cy, b11_y0, b11_y1 = y_centres[11]
b7_cx, b7_cy, b7_y0, b7_y1 = y_centres[7]
rail([(BX + BW, b11_cy), (790, b11_cy), (790, b7_cy), (BX + BW, b7_cy)],
     color="#2E7D32", ls=":", label="Loop 2\n(∇θ update)",
     label_xy=(770, (b11_cy + b7_cy) / 2))

# Lateral arrow to logger from box 10
b10_cx, b10_cy, _, _ = y_centres[10]
arrow(BX, b10_cy, 40 + 200, b10_cy - 150, color="#555", lw=1.1,
      label="metrics")

# Caption
cap = (
    "Caption. Real demonstrations (WSD4FESDRM, §3.5) calibrate the 3CC-r ODEs (Path G) and train MMICRL (§4), "
    "which outputs per-worker Θ_max(z). In Loop 1, each worker observes x_i=[MR,MA,MF]; NSWF (§6, Eq.33) assigns "
    "j*(i); load-translation (§7.1) gives T_L; actor π_θ proposes C_nom (Eq.35); the dual-barrier ECBF-QP "
    "(§5, Eq.18+23) projects to C*; 3CC-r integrates to x(t+1). R_t (NSWF log-surplus) and C_t (Θ_max violations) "
    "drive MAPPO-Lagrangian (Loop 2) with dual-ascended λ. All four prior errors (Θ_max vs δ_max, trunk F, reward "
    "log-surplus, global-state dim) are corrected; section refs on every box are auditable against "
    "MATHEMATICAL MODELLING.pdf."
)
ax.add_patch(FancyBboxPatch((40, 860), 1430, 35,
                            boxstyle="round,pad=2,rounding_size=4",
                            facecolor="#FFFFFF", edgecolor="#888",
                            linewidth=1.0))
ax.text(48, 863, cap, fontsize=7.2, va="top", ha="left", wrap=True)

out = r"C:/Users/admin/Desktop/hcmarl_project/diagrams/HC_MARL_ARCHITECTURE_v2_preview.png"
plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1,
            facecolor="white")
print(f"Wrote: {out}")
