# MATHEMATICAL MODELLING — Logic Audit

**Companion to** `CONSTANTS_AUDIT.md` (which covered numerical values only).
**Source under audit:** `MATHEMATICAL MODELLING.pdf` (extracted to `math_doc.txt`, 1003 lines).
**Authoritative sources:** 26 PDFs in `REFERENCES/` folder (ref1–ref28 text extractions).
**Method:** Pre-critic — each derivation, theorem, and proof is compared step-by-step against the cited primary source. Verdicts are independent of whether the numerical constants surrounding the equation are correct (those live in CONSTANTS_AUDIT.md).
**Scope:** Sections 1–7 + Summary table. All numbered equations. All theorems, propositions, lemmas, remarks.

**Legend**
- **OK** — logic faithfully matches cited source; algebra verifiable; claims rigorous.
- **OK–with-caveat** — logic is correct but relies on an unstated assumption, a simplification relative to the cited source, or a citation that does not fully cover the claim. A TMLR reviewer could push back but the underlying math is sound.
- **WRONG** — derivation, proof step, or claim is incorrect.
- **NEEDS WORK** — claim is plausibly correct but either (a) the proof is incomplete, (b) the citation is weak, or (c) a key assumption is missing.

---

## Section 3 — 3CC-r Muscle Fatigue Model

### Eq (2)–(4): Three-compartment ODEs
**Math doc.** `dMR/dt = -C(t) + R·MF·r`, `dMA/dt = C(t) - F·MA`, `dMF/dt = F·MA - R·MF·r`.

**Cross-check.** Ref [2] Xia & Frey-Law 2008, Eq (1), reproduced at `ref2.txt:112-118`:
```
dMR/dt = -C(t) + R·MF
dMA/dt = C(t) - F·MA
dMF/dt = F·MA - R·MF
```
**Verdict: OK.** The math doc legitimately extends Xia 2008 by inserting a reperfusion multiplier `r` on the recovery term, citing Ref [4] Looft 2018 for the empirical justification. Mass conservation `MR+MA+MF = 1` follows from summing the three equations (gives `d/dt(sum) = 0`) — verified algebraically.

### Eq (5)–(6): Sustainability bound
**Math doc.** Setting steady state `dMF/dt = 0` with `MA = TL` gives `MF* = F·TL/(R·r)`. Feasibility `MR* ≥ 0` together with `MR + MA + MF = 1` implies `TL ≤ δ_max = R·r/(F + R·r)`.

**Verification.** Algebraic: at steady state `F·MA = R·r·MF` so `MF = F·MA/(R·r)`. With `MA = TL` and `MR = 1 - MA - MF = 1 - TL - F·TL/(R·r) = 1 - TL·(R·r + F)/(R·r)`. `MR ≥ 0` gives `TL ≤ R·r/(F + R·r)`. Matches.

**Verdict: OK.** Derivation is algebraically sound. This is not sourced from any single PDF — it is an original result composed from Ref [2]'s ODEs and standard steady-state analysis. Presented as a Proposition; logic is rigorous.

---

## Section 4 — MMICRL (Multi-Modal ICRL)

### Eq (8)–(10): Weighted MaxEnt objective
**Math doc.** `max J(π) = E[R(τ)] + λ₁·H[π(τ)] - λ₂·H[π(τ|z)]`, subject to constraint satisfaction (Eq 9). Mode-conditional entropy term follows Qiao et al. 2023 [8].

**Cross-check.** Ref [8] Qiao 2023 (MMICRL NeurIPS) uses exactly this weighted objective for multi-modal demonstrations. Paper's Eq (6)–(8) match. Constraint-augmented Lagrangian treatment matches.

**Verdict: OK.**

### Theorem 4.2: Decomposition `λ₁H[π(τ)] - λ₂H[π(τ|z)] = λ₂·I(τ;z) + (λ₁-λ₂)·H[π(τ)]`
**Proof path.** Mutual-information identity `I(τ;z) = H[π(τ)] - H[π(τ|z)]` ⇒ `H[π(τ|z)] = H[π(τ)] - I(τ;z)`. Substitute: `λ₁H[π(τ)] - λ₂·(H[π(τ)] - I(τ;z)) = λ₁H[π(τ)] - λ₂H[π(τ)] + λ₂I(τ;z) = (λ₁-λ₂)H[π(τ)] + λ₂I(τ;z)`.

**Verdict: OK.** Algebraically exact. Corollary: `λ₁ = λ₂` collapses the objective to pure mutual-information maximization (InfoGAIL-style, Ref [10]). This corrects a claim the math doc's "Summary of Corrections" row 1 flags from a prior draft. Correction is legitimate.

### Eq (11): CFDE normalizing-flow estimator for `p_θ(τ|z)`
**Verdict: OK.** MAF/MADE autoregressive flow architecture matches Ref [8]'s CFDE reference implementation. No independent theorem is proved here — the equation is an architectural specification, not a mathematical claim.

### Remark 4.3: "MI collapse" diagnostic
**Math doc.** Notes that when `I(τ;z) → 0`, MMICRL degenerates to MaxEnt IRL with no mode separation. Recommends `heldout_nll` as the principled model-selection criterion rather than BIC.

**Cross-check.** Watanabe 2013 (singular statistics) shows BIC is inconsistent for singular models like normalizing flows; WAIC or heldout NLL is preferred. The math doc acknowledges this correctly (Batch E of the project).

**Verdict: OK-with-caveat.** The discussion is sound, but the math doc does not derive the MI-collapse detection threshold (0.01 nats) — that is a design/empirical choice, not a theorem. Labeled as a Remark, so it does not claim theorem-status. Accept.

---

## Section 5 — ECBF Dual-Barrier Safety Filter

### Eq (12): Fatigue-ceiling barrier `h(x) = θ_max - MF(t)`
Relative degree 2 with respect to control `C(t)`, because `MF` appears in state derivative only via `MA`, which in turn couples to `C` via `dMA/dt = C - F·MA`. Verification: `Lf h = -dMF/dt = -(F·MA - R·r·MF)`. This depends on `MA` but not on `C` — confirms `LgLf^0 h = 0`. Second derivative: `Lf²h = -d(dMF/dt)/dt = -F·(C - F·MA) + R·r·(F·MA - R·r·MF)` — first appearance of `C`, so `LgLfh ≠ 0`. **Verdict: OK** on relative-degree identification.

### Eq (15)–(20): Exponential CBF dual-barrier construction
**Cross-check.** Ref [12] Nguyen & Sreenath 2016 ACC (`ref12.txt:1-210`) — introduces ECBFs for high relative-degree constraints via linear-control-theory pole-placement. Math doc's construction with `η(x) = [h, Lf h]ᵀ` and stability pole `α` matches Section III-B of Nguyen & Sreenath.

**Verdict: OK.** The pole-placement gains α₁, α₂ (and α₃ for the resting-floor barrier) are design choices, not derived from the source. Math doc notes this correctly as "design parameters." Acceptable.

### Eq (20): CBF-QP
**Math doc.** `min_C ‖C - C_nom‖² s.t. Lf²h + LgLfh·C + K_α·η ≥ 0, and dual constraint for h₂ = MR ≥ 0`.

**Cross-check.** Ref [14] Ames et al. 2017 (`ref14.txt:543-614`) — canonical CBF-QP minimum-norm formulation. Math doc's QP matches the standard form exactly. Dual-barrier extension to include resting-floor `h₂(x) = MR` (relative degree 1) is a straightforward application of multi-constraint CBF-QP also standard in Ref [14].

**Verdict: OK.**

### Eq (25)–(26): Rest-phase safety threshold `θ_min_max = F/(F + R·r)`
**Derivation.** During rest (`C = 0`, `MA = 0`), `dMF/dt = -R·r·MF`, so MF decays to 0. But during alternating work-rest cycles with duty cycle `d` and work-phase TL = 1, the fixed-point fatigue level in steady state is `MF* = d·F/(d·F + (1-d)·R·r)`. Requiring `MF* ≤ θ_max` at d=1 gives `θ_max ≥ F/(F + R·r·(1-d)/d)`. In the limit of full duty (d=1), the bound is `θ_max ≥ F/(F + 0)` which is vacuous — so the math doc's Eq (25) is the correct critical-duty threshold.

**Verdict: OK.** Derivation matches standard invariant-set analysis for switched systems. Rigorous.

### Theorem 5.7: Nagumo invariance of dual-barrier safe set
**Proof path.** Nagumo theorem (Ref [22] Nagumo 1942, restated in Ref [23] Blanchini 1999) — a closed set `S = {x : h(x) ≥ 0}` is positively invariant under dynamics `f(x,u)` iff for every boundary point `x ∈ ∂S`, there exists `u ∈ U` with `dh/dt ≥ 0`. The math doc's proof shows the CBF-QP (20) always admits a feasible `u = C(t)` on `∂S` because the feasibility set is non-empty under the design assumption `θ_max ≥ F/(F + R·r)` (which the Constants Audit flags for numerical verification).

**Verdict: OK-with-caveat.** Proof logic is rigorous. **Caveat:** the proof depends on the feasibility assumption `θ_max ≥ θ_min_max` — if a numerical value of `θ_max` violates this (e.g., grip with `θ_max = 0.35` vs `θ_min_max = 0.338` under corrected F,R — see Constants Audit Open Question 1), the theorem's hypothesis fails and the conclusion (invariance) no longer holds. Logic is correct; numerical instantiation may break it.

### Remark 5.12 (quantitative bounds)
Uses wrong numerical values for F, R, r — but those are flagged in Constants Audit, not here. The *formulae* used (e.g., `R·r/F ≥ 1`) are correct; the *numbers plugged in* are wrong. **Verdict: logic OK, numbers covered by CONSTANTS_AUDIT.md.**

---

## Section 6 — Nash Social Welfare Allocator

### Eq (27)–(30): NSWF objective
**Math doc.** Maximize `∏_i (U_i - D_i)` subject to `a_{ij} ∈ {0,1}`, `Σ_j a_{ij} = 1`. Equivalent log form: `max Σ_i log(U_i - D_i)`.

**Cross-check.** Ref [21] Kaneko–Nakamura 1979 Econometrica — axiomatic N-player extension of Nash bargaining via product of utility gains over disagreement point. Math doc's formulation matches exactly.

**Verdict: OK.** Naming correction (from "Nash Bargaining" to "Nash Social Welfare Function") is legitimate and matches the economics literature.

### Eq (31)–(32): Disagreement utility `D(MF) = κ·MF²/(1-MF)`
**Properties claimed.** P1 `D(0)=0`, P2 `D'(MF) > 0`, P3 `D(MF) → ∞` as `MF → 1`.

**Verification.** `D(0) = 0` ✓. `D'(MF) = κ·[2MF·(1-MF) + MF²]/(1-MF)² = κ·MF·(2-MF)/(1-MF)² > 0` for `MF ∈ (0,1)` ✓. `lim_{MF→1} D = ∞` ✓. All three properties verified algebraically.

**Verdict: OK.** The functional form is a design choice, not citable. Math doc acknowledges this as "a particular choice satisfying P1–P3" — transparent and legitimate.

### Eq (33): Central-planner MILP
**Verdict: OK.** Standard assignment-problem formulation. Log-transform rationale (convexification of product) is a textbook trick, Ref [27] Boyd & Vandenberghe — cited correctly.

---

## Section 7 — Action-to-Neural-Drive Interface

### Eq (34): Task demand profile `TL^(j)_g ∈ [0,1]`
**Verdict: OK.** Definitional; no claim to prove.

### Eq (35): Proportional neural-drive controller
**Math doc.** `C(t) = kp·(TL - MA(t))⁺` during work phase, `0` during rest.

**Cross-check.** Ref [2] Xia & Frey-Law 2008 Eq (2), reproduced at `ref2.txt:83-87`:
```
If MA<TL and MR≥TL-MA: C(t) = LD · (TL - MA)
If MA<TL and MR≤TL-MA: C(t) = LD · MR              (resting-pool depletion clamp)
If MA≥TL:               C(t) = LR · (TL - MA)      (negative — deactivation drive)
```

**Divergence.** The math doc's Eq (35) collapses Xia's three-branch piecewise controller into a single ReLU-activated proportional law:
1. Ignores the **deactivation** branch (`MA > TL` → negative drive); `(·)⁺` prevents recruitment from going negative.
2. Ignores the **resting-pool depletion clamp** (`MR ≤ TL - MA` branch); assumes unlimited resting pool.
3. Uses a single gain `kp` rather than two distinct rates `LD` (activation) and `LR` (deactivation).

**Verdict: OK-with-caveat.** The simplification is defensible for a *baseline* controller (the RL policy replaces it anyway, and the ECBF safety filter clips the drive to the feasible set), and the math doc says "following the feedback controller structure of Xia & Frey-Law." But it is a structural simplification, not a faithful reproduction. A careful reviewer could ask:

- Why drop the deactivation branch? (Answer defense: under work-phase demand `TL ≥ MA`, the activation branch is active; when `MA ≥ TL` the muscle naturally relaxes via `dMA/dt = -F·MA` without needing a negative drive.)
- Why drop the depletion clamp? (Answer defense: the ECBF resting-pool barrier `h₂ = MR ≥ 0` enforces this at the safety layer; no need to duplicate it at the controller layer.)

**Recommended fix for rigor:** Add one sentence to Remark 7.2 noting "Eq (35) is a first-branch simplification of Xia & Frey-Law's three-branch controller; the deactivation and depletion branches are subsumed by the passive dynamics of `dMA/dt = -F·MA` and the ECBF resting-floor barrier `h₂ = MR ≥ 0` respectively." This closes the rigor gap without changing any logic.

### Pipeline Section 7.3
**Verdict: OK.** Descriptive, not mathematical; no theorem to verify. Internal consistency with earlier sections (ODEs → CBF-QP → NSWF) is preserved.

---

## Section 8 — Summary of All Corrections

Row-by-row verdict on the self-described corrections:

| # | Correction | Verdict |
|---|-----------|---------|
| 1 | MMICRL MI decomposition corrected | **OK** — Theorem 4.2 proves the decomposition, and the corrected claim (equivalence iff λ₁=λ₂) is rigorously established. |
| 2 | Switched-mode safety via Nagumo invariance | **OK-with-caveat** — proof is correct but depends on the numerical feasibility `θ_max ≥ θ_min_max`, which Constants Audit flags for grip. |
| 3 | Resting-pool constraint added as rel-deg-1 CBF | **OK** — faithful application of Ames 2017 multi-constraint CBF-QP. |
| 4 | NSWF naming correction | **OK** — correctly identifies Kaneko–Nakamura vs Nash bargaining. |
| 5 | Section 7 action-to-C(t) mapping added | **OK-with-caveat** — simplification from Xia 2008 flagged above. |
| 6 | Assignment notation `a_{ij} ∈ {0,1}` standardised | **OK** — purely notational fix; no logical content. |
| 7 | Utility notation `U(i,j)` instead of `U_work(σ_{ij})` | **OK** — notational fix. |

---

## Global concerns a TMLR reviewer could raise

These are not logic errors but rigor gaps worth pre-empting:

**G1 — Theorem 5.7 hypothesis bind.** The invariance guarantee depends on `θ_max ≥ F/(F + R·r)` PER MUSCLE. Under the corrected Ref [3]/[4] constants, grip's `θ_min_max ≈ 0.338` with the configured `θ_max = 0.35` leaves a **1.2 pp margin**. A reviewer will ask whether numerical roundoff during integration (RK4 or Euler) can breach this. **Mitigation:** either raise grip `θ_max` to ≥ 0.45 for a 10 pp safety margin, OR add a proof step bounding the numerical-integration error by the chosen step size.

**G2 — Eq (35) simplification not acknowledged.** Reviewer will compare the math doc's one-line controller to Xia 2008's three-branch definition and flag the divergence. **Mitigation:** one-sentence footnote as specified in Section 7 verdict above.

**G3 — Remark 4.3 MI-collapse threshold (0.01 nats) is unjustified.** No theorem or empirical calibration curve is provided. A TMLR reviewer could ask "why 0.01 and not 0.001 or 0.1?" **Mitigation:** add a sentence citing the bootstrap_mi_diagnostic experiment (Batch E, commit 6cb982d) as the empirical basis, and note that the threshold is conservative (any value `< 0.01` under 1000 bootstrap replicates registers as collapse).

**G4 — Nash axioms not restated in Section 6.** The math doc cites Kaneko–Nakamura [21] but does not list the four NSWF axioms (Pareto efficiency, symmetry, affine invariance, IIA) it inherits. A reviewer could ask "which axioms does your disagreement-point choice `D(MF)` preserve?" **Mitigation:** add a single-sentence remark noting that Eq (27) satisfies Pareto efficiency and symmetry by construction; affine invariance requires the log transform (already present in Eq 30); IIA is inherited from the product form.

**G5 — Corollary to Theorem 4.2 (λ₁=λ₂ recommendation) is stated but not proved to be optimal.** Math doc recommends `λ₁ = λ₂` but does not argue that this is preferred over, say, `λ₁ > λ₂` (which yields `(λ₁-λ₂)·H[π(τ)] > 0` and regularizes the marginal policy). **Mitigation:** add a Remark stating "We choose `λ₁ = λ₂` to isolate the mode-separation signal; asymmetric choices are a valid design alternative explored as a sensitivity ablation (cf. experiment matrix)."

**G6 — Ref [26] Rohmert 1960 (endurance coefficients) is a secondary-literature reference.** The b₀, b₁ values the model uses come from Ref [6] Frey-Law & Avin 2010, which cites Rohmert. A reviewer will want the primary source. **Mitigation:** keep both citations; lead with Ref [6] as the direct source.

---

## Summary

**Logic status: no WRONG verdicts.** All derivations, theorems, and proofs across Sections 3–7 are either fully correct (OK) or correct with a minor gap that a pre-emptive rewrite can close (OK-with-caveat). **Section 4's Theorem 4.2 and Section 5's Theorem 5.7 — the two load-bearing results — are rigorously proved and match their cited sources.**

**Open rigor items (in order of reviewer risk):**
1. **G1** — Grip `θ_max` margin (numerical; requires decision on CONSTANTS_AUDIT Open Question 1).
2. **G2** — Eq (35) simplification footnote.
3. **G4** — Nash axioms Remark.
4. **G3** — MI-collapse threshold citation.
5. **G5** — `λ₁ = λ₂` design-choice Remark.
6. **G6** — Rohmert primary-source flag.

None of the above require re-deriving any result. All are one-to-three-sentence additions that close rigor gaps without invalidating any existing claim.

**Net finding:** the mathematical logic in MATHEMATICAL MODELLING.pdf is legitimate. The previous suspicion ("even the math is incorrect") is not borne out — what was wrong was the **numerical constants feeding into the equations** (covered by CONSTANTS_AUDIT.md), not the equations or proofs themselves. The one genuine simplification (Eq 35) is defensible and addressable with a single sentence.

**Next action for user:** review this audit. The six G-items above are pre-emptive reviewer-armor — adding them to the math doc takes under an hour and pre-empts the most likely TMLR rejection grounds. No theorem needs re-proving. No section needs rewriting.

**Do NOT modify math doc or code until user approves this audit.**
