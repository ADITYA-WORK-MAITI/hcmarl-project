# CONSTANTS AUDIT — TMLR-style reference verification
Generated: 2026-04-21 (v1)
Updated: 2026-04-21 (v2 — after Research Mode PDF delivery of 4 additional sources)
Method: every numerical constant in the HCMARL + baseline execution path was
extracted and matched against the primary-source PDFs in `REFERENCES/`.
No code was changed; this is a verification pass only.

## v2 update — what Claude Research Mode delivered

Four additional PDFs were placed in `REFERENCES/` (user manually downloaded
from the Research Mode link set):

- `1 (UNVERIFIED 1).pdf`  → Liu, Brown & Yue (2002) — **now extractable**
- `UNVERIFIED 4.pdf`       → Hoozemans et al. (2004) — **now extractable**
- `UNVERIFIED 8.pdf`       → Anton et al. (2001) — **now extractable**
- `UNVERIFIED 9.pdf`       → Waters et al. (1993) NIOSH RLE — **now extractable**
- `15 (UNVERIFIED 10).pdf` → Ames et al. (2019) ECC CBF theory & applications — **now extractable**

Research Mode also reports the following are **genuinely unreachable** in
free-PDF form (paywalled with no preprint / mirror found):

- Rohmert 1960 (Dauerleistungsgrenze foundation)
- Granata & Marras 1995
- de Looze 1995
- Snook & Ciriello 1991
- Nordander et al. 2000

These five papers' rows remain `[UNVERIFIED]` in this audit. Research Mode
additionally flagged a **second class of integrity issue** for these papers:
**even if we obtain them, the numerical claims we source from them are
unit-mismatched** — the papers report N / Nm / kg (psychophysical), NOT
%MVC. Our demand profile citations are therefore likely to be
**mis-attributions**, not merely unverified. This is discussed per-row in
Table G and flagged in the Bottom-Line Summary.

Reference shorthand (as named in math_doc Section 9, References):
- [2] Xia, T., & Frey-Law, L. A. (2008). A theoretical approach for modeling
  peripheral muscle fatigue and recovery. J. Biomechanics 41(14):3046-3052.
- [3] Frey-Law, L. A., Looft, J. M., & Heitsman, J. (2012). A three-compartment
  muscle fatigue model accurately predicts joint-specific maximum endurance
  times. J. Biomechanics 45(10):1803-1808.
- [4] Looft, J. M., Herkert, N., & Frey-Law, L. A. (2018). Modification of a
  three-compartment muscle fatigue model to predict peak torque decline during
  intermittent tasks. J. Biomechanics 77:16-25.
- [5] Looft, J. M., & Frey-Law, L. A. (2020). Adapting a fatigue model for
  shoulder flexion fatigue ... J. Biomechanics 103:109718.
- [6] Frey-Law, L. A., & Avin, K. G. (2010). Endurance time is joint-specific:
  a modelling and meta-analysis investigation. Ergonomics 53(1):109-129.
- [8] Qiao, G., et al. (2023). Multi-Modal Inverse Constrained Reinforcement
  Learning from a Mixture of Demonstrations. NeurIPS 2023.
- [12] Nguyen, Q., & Sreenath, K. (2016). Exponential control barrier functions
  for enforcing high relative-degree safety-critical constraints. ACC.
- [14] Ames, A. D., et al. (2017). Control barrier function based quadratic
  programs for safety critical systems. IEEE TAC 62(8).
- [15] Ames, A. D., et al. (2019). Control Barrier Functions: Theory and
  Applications. ECC 2019. *(newly available, `ref_ames2019.txt`)*
- [21] Kaneko, M., & Nakamura, K. (1979). The Nash social welfare function.
  Econometrica 47(2):423-435.
- [1-Liu] Liu, Brown & Yue (2002) Biophysical Journal 82(5):2344-2359.
  *(newly available, `ref1_liu2002.txt`)*
- [Anton] Anton, D. et al. (2001). Overhead drilling position: shoulder moment
  and EMG. Ergonomics 44(5):489-501. *(newly available, `ref_anton2001.txt`)*
- [Hoozemans] Hoozemans, M. J. M. et al. (2004). Mechanical loading of the low
  back and shoulders during pushing and pulling. Ergonomics 47(1):1-18.
  *(newly available, `ref_hoozemans2004.txt`)*
- [Waters] Waters, T. R. et al. (1993). Revised NIOSH equation for the design
  and evaluation of manual lifting tasks. Ergonomics 36(7):749-776.
  *(newly available, `ref_waters1993.txt`)*

Rows are flagged:
  [WRONG]        current code value contradicts the cited primary source
  [OK]           current code value matches the cited primary source
  [DESIGN]       current code value is a design choice, not read from a paper
  [UNVERIFIED]   cited source not available and not obtainable via free PDF
  [MIS-ATTR]     source is available but the number we cite is NOT in the
                 source in the form we cite it — wrong muscle, wrong units,
                 or wrong paper

---

## TABLE A — 3CC-r muscle fatigue constants (F, R, r)
*(unchanged from v1 — no new PDFs affect this table)*

Source of truth for F, R: Ref [3] Frey-Law, Looft & Heitsman (2012) Table 1.
Extracted verbatim from `REFERENCES/ref3.txt` lines 589-599:

```
Ankle     0.00589 0.00058 8/9 10.2  8.99 8.96
Knee      0.01500 0.00149 8/9 10.1  9.05 9.04
Trunk     0.00755 0.00075 8/9 10.1  9.05 9.04
Shoulder  0.01820 0.00168 8/9 10.8  8.47 8.45
Elbow     0.00912 0.00094 8/9       9.7  9.36 9.34
Hand/Grip 0.00980 0.00064 7/9 15.3  6.14 6.13
```

Shoulder cross-verified via Ref [5] Looft & Frey-Law 2020 line 206:
  "for shoulder muscles (F = 0.01820, R = 0.00168) were used for all model
   predictions"

Grip F,R cross-verified via Ref [4] Looft et al. 2018 line 1085:
  "Grip  0.00980 0.00064 30  0.01365  15.3 0.065"

r values from Ref [4] lines 256-257 and 1085, Ref [5] line 33-34:
  "15 was chosen as the optimal value for all regions except for hand/grip"
  "r = 15 somewhat better than r=30" for shoulder

Locations: `hcmarl/three_cc_r.py:79-84`;
`hcmarl/real_data_calibration.py:474-481`; every config yaml lines 12-18;
`math_doc.txt` Table 1 line 240-250.

| # | Variable | Current value (CODE + MATH DOC) | Corrected value (SOURCE) | Reference + exact location |
|---|---|---|---|---|
| A1 | shoulder F | 0.0146 | **0.01820** | [WRONG] [3] Table 1 (ref3.txt L595); [5] L206 |
| A2 | shoulder R | 0.00058 | **0.00168** | [WRONG] [3] Table 1 (ref3.txt L595); [5] L206 |
| A3 | shoulder r | 15 | 15 | [OK] [4] L257, [5] L33-34 |
| A4 | ankle F | 0.00589 | 0.00589 | [OK] [3] Table 1 (ref3.txt L589) |
| A5 | ankle R | **0.0182** | **0.00058** | [WRONG] [3] Table 1 (ref3.txt L589) — code has shoulder F in this cell |
| A6 | ankle r | 15 | 15 | [OK] [4] L257 |
| A7 | knee F | 0.0150 | 0.01500 | [OK] [3] Table 1 (ref3.txt L591) |
| A8 | knee R | 0.00175 | **0.00149** | [WRONG] [3] Table 1 (ref3.txt L591) |
| A9 | knee r | 15 | 15 | [OK] [4] L257 |
| A10 | elbow F | 0.00912 | 0.00912 | [OK] [3] Table 1 (ref3.txt L597) |
| A11 | elbow R | 0.00094 | 0.00094 | [OK] [3] Table 1 (ref3.txt L597) |
| A12 | elbow r | 15 | 15 | [OK] [4] L257 |
| A13 | trunk F | 0.00657 | **0.00755** | [WRONG] [3] Table 1 (ref3.txt L593) |
| A14 | trunk R | 0.00354 | **0.00075** | [WRONG] [3] Table 1 (ref3.txt L593) |
| A15 | trunk r | 15 | 15 | [OK] [3] did not measure trunk-r; 15 extrapolated from [4] default |
| A16 | grip F | 0.00794 | **0.00980** | [WRONG] [3] Table 1 (ref3.txt L599); [4] L1085 |
| A17 | grip R | 0.00109 | **0.00064** | [WRONG] [3] Table 1 (ref3.txt L599); [4] L1085 |
| A18 | grip r | 30 | 30 | [OK] [4] L256-257, L1085 |

Verdict on Table A: 8 of 18 rows WRONG. Only elbow F, R match the cited
source. Pattern suggests a transcription mix-up — ankle R=0.0182 equals
the true shoulder F (0.01820), and shoulder R=0.00058 equals the true ankle
R. Grip and trunk are shifted versions of neighboring rows in the Table 1
printed order; this is consistent with a column mis-read rather than an
independent fabrication.

---

## TABLE B — Derived quantities that propagate the A-row error
*(unchanged from v1)*

Derived using the correct F, R from Ref [3] and r from Ref [4]:
  delta_max      = R / (F + R)          (math doc Eq 6)
  theta_min_max  = F / (F + R*r)        (math doc Eq 25/26)
  Rr_over_F      = R*r / F

Ref [3] Table 1 also publishes its own delta_max (column 6), so the correct
delta_max values are independently confirmed.

| # | Variable | Current value | Corrected value (SOURCE) | Reference |
|---|---|---|---|---|
| B1 | shoulder delta_max | 3.8% | **8.45%** | [WRONG] [3] Table 1 col 6 (ref3.txt L595) |
| B2 | shoulder theta_min_max | 62.7% | **41.9%** = 0.01820/(0.01820+0.00168*15) | [WRONG] recomputed from [3] |
| B3 | shoulder Rr/F | 0.596 | **1.385** = 0.00168*15/0.01820 | [WRONG] recomputed from [3] |
| B4 | ankle delta_max | 75.5% | **8.96%** | [WRONG] [3] Table 1 col 6 (ref3.txt L589) |
| B5 | ankle theta_min_max | 2.1% | **40.4%** = 0.00589/(0.00589+0.00058*15) | [WRONG] recomputed from [3] |
| B6 | ankle Rr/F | ~46.35 | **1.476** = 0.00058*15/0.00589 | [WRONG] recomputed from [3] |
| B7 | knee delta_max | 10.4% | **9.04%** | [WRONG] [3] Table 1 col 6 (ref3.txt L591) |
| B8 | knee theta_min_max | 36.4% | **40.2%** = 0.01500/(0.01500+0.00149*15) | [WRONG] recomputed from [3] |
| B9 | knee Rr/F | ~1.75 | **1.490** = 0.00149*15/0.01500 | [WRONG] recomputed from [3] |
| B10 | elbow delta_max | 9.3% | 9.34% | [OK] [3] Table 1 col 6 (ref3.txt L597) |
| B11 | elbow theta_min_max | 39.3% | 39.3% | [OK] matches under correct F, R |
| B12 | elbow Rr/F | 1.547 | 1.547 | [OK] |
| B13 | trunk delta_max | 35.0% | **9.04%** | [WRONG] [3] Table 1 col 6 (ref3.txt L593) |
| B14 | trunk theta_min_max | 11.0% | **40.2%** = 0.00755/(0.00755+0.00075*15) | [WRONG] recomputed from [3] |
| B15 | trunk Rr/F | ~8.08 | **1.490** = 0.00075*15/0.00755 | [WRONG] recomputed from [3] |
| B16 | grip delta_max | 12.1% | **6.13%** | [WRONG] [3] Table 1 col 6 (ref3.txt L599) |
| B17 | grip theta_min_max | 19.5% | **33.8%** = 0.00980/(0.00980+0.00064*30) | [WRONG] recomputed from [3], [4] |
| B18 | grip Rr/F | ~4.11 | **1.959** = 0.00064*30/0.00980 | [WRONG] recomputed from [3], [4] |

Downstream impact: math_doc.txt line 252-254 has a "verification" arithmetic
that is INTERNALLY consistent with the wrong inputs but gives
0.00058/0.01518 = 3.8%, contradicting the paper's own published delta_max of
8.45% for shoulder (Ref [3] Table 1). The math doc silently disagrees with
its cited source.

Also: math_doc.txt line 255-257 claims "fatigue-resistance ranking
(ankle > trunk > grip > knee > elbow > shoulder)". Under the corrected
values every muscle has delta_max 6-9% (tight band), so this ranking claim
collapses — the muscles differ mainly in Rr/F ratio, not delta_max.

Also: math_doc.txt line 467 ("Rr/F = 0.596 for shoulder") and line 672
(Remark 5.12) quote F=0.0146, R=0.00058 — both rows must be rewritten.

---

## TABLE C — Endurance power model (Ref [6] Frey-Law & Avin 2010 Table 2)
*(unchanged — row-alignment still requires visual PDF check)*

Source: `REFERENCES/ref6.txt` lines 1824-1846. PDF extraction mangled the
row/column alignment. The set of b0 values {14.86, 34.71, 19.38, 17.98,
33.55, 22.69} reported in code matches the printed b0 values in the paper
exactly. The set of b1 values {-1.83, -2.06, -1.88, -2.21, -1.61, -2.27}
in code matches the paper's printed power-model b1 set. Joint-to-b1
alignment still needs visual PDF verification by you — the text-extracted
order is ambiguous.

Location: `hcmarl/real_data_calibration.py:443-450`.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| C1 | shoulder b0 (min) | 14.86 | [OK set match; order TBD] | [6] Table 2 |
| C2 | shoulder b1 | -1.83 | [ORDER TBD] | [6] Table 2 |
| C3 | ankle b0 | 34.71 | [OK] | [6] Table 2 |
| C4 | ankle b1 | -2.06 | [ORDER TBD] | [6] Table 2 |
| C5 | knee b0 | 19.38 | [OK] | [6] Table 2 |
| C6 | knee b1 | -1.88 | [ORDER TBD] | [6] Table 2 |
| C7 | elbow b0 | 17.98 | [OK] | [6] Table 2 |
| C8 | elbow b1 | -2.21 | [ORDER TBD] | [6] Table 2 |
| C9 | grip b0 | 33.55 | [OK] | [6] Table 2 |
| C10 | grip b1 | -1.61 | [ORDER TBD] | [6] Table 2 |
| C11 | trunk b0 | 22.69 | [OK] | [6] Table 2 |
| C12 | trunk b1 | -2.27 | [ORDER TBD] | [6] Table 2 |

Action for you: open REFERENCES/6.pdf page 34, Table 2, and confirm the
joint -> b1 alignment. If any joint-b1 row is swapped in code, it is a
silent 3-10% error in predicted endurance time.

---

## TABLE D — CV for F and R (calibration noise) — **UPDATED v2**

Location: `hcmarl/real_data_calibration.py:484-497`.

**v2 update:** Liu 2002 PDF is now extractable. Table 2 (verbatim from
`ref1_liu2002.txt` lines 687-699) reports per-subject F, R, and B values
for **10 subjects performing a 3-minute sustained maximal voluntary
handgrip contraction** (NOT elbow — the Research Mode integrity flag was
correct).

Computed from Liu 2002 Table 2 Mean/SD row (`ref1_liu2002.txt` L698-699):
  - Mean F = 0.0206/s, SD_F = 0.0075/s → **CV_F = 0.0075/0.0206 = 0.364**
  - Mean R = 0.0084/s, SD_R = 0.0036/s → **CV_R = 0.0036/0.0084 = 0.429**

The numerical values 0.36 and 0.43 used in our code are **numerically
correct** reproductions of Liu 2002 Table 2 — but they apply to
**handgrip**, not elbow. Our code/comment attributes them to "elbow",
which is a **MIS-ATTRIBUTION** in the inline comment at
`real_data_calibration.py:484, 495`.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| D1 | elbow CV_F | 0.36 | **[MIS-ATTR]** Liu 2002 reports this for **handgrip** N=10, not elbow (ref1_liu2002.txt L698 Table 2) | comment needs fix |
| D2 | all other CV_F | 0.30 | [DESIGN] explicitly labelled "conservative default" below Liu's 0.36 | internal |
| D3 | global CV_R | 0.40 | [DESIGN-conservative] Liu 2002 measured **0.429 for handgrip** (ref1_liu2002.txt L699); using 0.40 is defensible-conservative, NOT a literal Liu-number match | comment needs fix |

**Additional finding from Liu 2002:**
The individual F values in Table 2 (10 subjects) range from 0.0075 to 0.033/s,
and R values from 0.0026 to 0.0125/s. This is the *actual* handgrip
population distribution. A TMLR reviewer can ask: why is our CV_R=0.40
used for **shoulder** (where no subject-level CV has been measured)?
The honest answer: no published shoulder CV exists; 0.40 is a conservative
engineering choice informed by a related muscle (handgrip). That framing
must appear in the code comment and in the paper, not a false "Liu 2002
elbow" attribution.

**Minimum fix for TMLR integrity:**
Rewrite `real_data_calibration.py:484, 495` comments to read
"Handgrip: Liu et al. (2002) Table 2, N=10 subjects, CV_F=0.364, CV_R=0.429.
All other muscles: conservative default CV_F=0.30, CV_R=0.40 (no
published per-muscle CVs)." Do NOT claim "elbow CV_F=0.36 from Liu 2002".

---

## TABLE E — ECBF design parameters — **UPDATED v2**

Location: `hcmarl/ecbf_filter.py:32-55`; `config/hcmarl_full_config.yaml:67-71`.

Ref [12] Nguyen & Sreenath 2016 teaches Exponential CBFs via pole placement:
the alpha_i gains are chosen by the DESIGNER subject to a stability
constraint. No numerical alpha values are prescribed in the paper. Ref [14]
Ames 2017 is similarly silent on alpha magnitudes.

**v2 update:** Ref [15] Ames 2019 ECC paper now extractable. Verbatim
(`ref_ames2019.txt` L271-282):

  "u(x) = argmin  (1/2) u^T H(x) u + p δ^2     (CLF-CBF QP)
       (u,δ)∈R^(m+1)
       s.t.  L_f V(x) + L_g V(x) u ≤ -γ(V(x)) + δ
              L_f h(x) + L_g h(x) u ≥ -α(h(x))"

  "δ is a relaxation variable that ensures solvability of the QP as
  penalized by p > 0 (i.e., to ensure the QP has a solution one must
  relax the condition on stability to guarantee safety)."

**Only "p > 0" is prescribed; no magnitude rule.** Our current source-code
comment at `hcmarl/ecbf_filter.py:28-31` reads:

  "Penalty on slack variables in the CBF-QP (Ames et al. 2019,
   'Control Barrier Functions: Theory and Applications', Section IV-B).
   Large enough that slack is only ever non-zero when strict feasibility
   is impossible under numerical noise; small enough to stay well-conditioned."

The Ames et al. 2019 paper does NOT contain a "large enough that slack is
only ever non-zero..." prescription. **This is a MIS-ATTRIBUTION.** Our
choice of 1000.0 is a defensible engineering design choice but cannot be
sourced from Ames 2019.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| E1 | alpha1 | 0.05 | [DESIGN] slow-convergence pole, plausible | [12] Sec III.B ("pole placement") — no numeric in paper |
| E2 | alpha2 | 0.05 | [DESIGN] matched to alpha1 | [12] Sec III.B — no numeric in paper |
| E3 | alpha3 | 0.1 | [DESIGN] RD-1 barrier, larger than alpha1,2 | [14] CBF-QP formulation — no numeric in paper |
| E4 | SLACK_PENALTY | 1000.0 | **[MIS-ATTR → DESIGN]** Ames 2019 (ref_ames2019.txt L271-282) only requires p>0, does NOT prescribe magnitude. Code comment line 28-31 is misleading. | soft-constraint scale, engineering choice |
| E5 | SLACK_EPS | 1e-6 | [DESIGN] numerical termination tol | internal |

Verdict: ECBF constants are all DEFENSIBLE design choices, but the
current code-comment attribution of SLACK_PENALTY=1000.0 to Ames 2019 is
not supported by the paper. Fix: rewrite the comment to state explicitly
"design choice; Ames et al. 2019 requires only p > 0 (ECC Sec III), no
magnitude prescription."

---

## TABLE F — theta_max thresholds (safety ceiling per muscle) — **UPDATED v2**

Location: `config/hcmarl_full_config.yaml:19-25`; `hcmarl/envs/pettingzoo_wrapper.py:82-85`.

math_doc.txt line 513 claims these are from "ergonomic guidelines for
heavy intermittent work [26, 6]". Ref [26] = Rohmert 1960.

**v2 update:** Research Mode confirms Rohmert 1960 has **no free PDF**
(paywalled at Springer, no German-language mirror, no open English
follow-up). Research Mode also notes the 1973 Rohmert *Applied Ergonomics*
follow-ups are also Elsevier-paywalled. Secondary literature (El Ahrache &
Imbeau 2009) discusses the canonical "15% MVC Dauerleistungsgrenze" but
is itself not openly downloadable.

Ref [6] Frey-Law & Avin 2010 is about endurance time, not about
"ergonomically acceptable fatigue fraction" — it does not prescribe
theta_max values. So Ref [6] alone cannot support our theta_max choices.

Under the CORRECTED F, R values (Table A), the theta_min_max constraint
(Eq 26) is 33.8%-40.4% for all 6 muscles. Current code values:

| # | Variable | Current value | Status vs corrected theta_min_max | Reference |
|---|---|---|---|---|
| F1 | shoulder theta_max | 0.70 | feasible (min 0.419) | [UNVERIFIED][DESIGN] [26] Rohmert not obtainable |
| F2 | ankle theta_max | 0.80 | feasible (min 0.404) | [UNVERIFIED][DESIGN] [26] Rohmert not obtainable |
| F3 | knee theta_max | 0.60 | feasible (min 0.402) | [UNVERIFIED][DESIGN] [26] Rohmert not obtainable |
| F4 | elbow theta_max | 0.45 | feasible (min 0.393) | [UNVERIFIED][DESIGN] [26] Rohmert not obtainable |
| F5 | trunk theta_max | 0.65 | feasible (min 0.402) | [UNVERIFIED][DESIGN] [26] Rohmert not obtainable |
| F6 | grip theta_max | 0.35 | **INFEASIBLE** (min 0.338) — only 1.2pp margin | [UNVERIFIED][DESIGN] [26] Rohmert not obtainable |

Also: the comment in `hcmarl_full_config.yaml` lines 20-25 writes e.g.
"shoulder: 0.70   # > theta_min_max 62.7%" — this 62.7% is the WRONG Table
B value. Under correct values, the comment should read ">41.9%". All six
comments on those lines are wrong.

Same issue in `hcmarl/envs/pettingzoo_wrapper.py:64-79` — the inline
comment "With theta_min_max=19.5% (r=30), the old 0.25 gave only 5.5pp
margin... 0.35 gives 15.5pp margin" — these percentages are the WRONG
Table B values. Under correct F, R, grip theta_min_max=33.8%, so 0.35
gives only 1.2pp margin.

Issue on F6: with correct grip F=0.00980, R=0.00064, r=30 the
theta_min_max is 33.8%. Current theta_max=0.35 gives only 1.2 pp margin
above the rest-phase floor — that is not a safe margin; ECBF will
repeatedly border on infeasibility. Under the WRONG values in code,
theta_min_max read 19.5% so 0.35 appeared to have 15.5pp margin. **Fixing
F, R alone will immediately destabilise the grip ECBF. Decision required:
raise grip theta_max to 0.45 (matches elbow), or revisit the grip demand
profile.**

**Reviewer pre-critic for F1-F6:** A TMLR reviewer will ask "where do
these specific numbers come from if Rohmert 1960 is not obtainable and
Ref [6] is about endurance not about ceilings?" Honest answer: these are
engineering design choices informed by Rohmert's canonical 15% sustained-
contraction limit and higher ceilings for intermittent work. The paper
must frame them as design choices, not as literature-derived thresholds.

---

## TABLE G — Task demand profiles (fraction of MVC per task) — **UPDATED v2 — major integrity issue**

Locations: `config/task_profiles.yaml`, `config/hcmarl_full_config.yaml:33-38`,
every baseline config lines 29-34.

**v2 update — CRITICAL INTEGRITY FINDING from Research Mode + cross-check of the four newly-obtained PDFs:**

Research Mode flagged all seven cited demand-profile sources as either
unreachable OR **unit-mismatched** (reports N, Nm, or kg — not %MVC). The
four PDFs we did obtain independently confirm this:

1. **Hoozemans 2004 (now available, `ref_hoozemans2004.txt`):** Paper
   abstract verbatim (L109-118):
     "exerted push/pull forces, net moments at the low back and shoulders,
      compressive and shear forces at the low back, and compressive forces
      at the glenohumeral joint"
   **All outcomes in N and Nm — no %MVC reported.** Our citation of
   Hoozemans 2004 for "deltoid 40-50% MVC during heavy lifts" is **NOT
   SUPPORTED** by the paper.

2. **Anton 2001 (now available, `ref_anton2001.txt`):** Paper uses %RVE
   (Reference Voluntary Exertion) not %MVC — authors defend this choice
   on p. 499. Paper Table 2 (`ref_anton2001.txt` L400-408):
     Anterior deltoid: 30.18% – 115.77% RVE (high step, 3 reach positions)
     Biceps brachii:   98.91% – 153.11% RVE (low step)
     Triceps brachii:  21.55% – 422.31% RVE
     Shoulder joint moment: 7.03 – 28.74 Nm
   The paper reports RIGHT-ARM muscles only — **no trunk EMG at all**. Our
   code/math-doc citation of Anton 2001 for "trunk 15% MVC during overhead
   placement" is not in the paper. Also note that the %RVE range for
   deltoid (30–154%) is dramatically wider than our cited "50–60% MVC".

3. **Granata 1995, de Looze 1995, Snook 1991, Nordander 2000:** all
   unreachable in free-PDF form. Research Mode's abstract check confirms
   they report N, Nm, or psychophysical kg, NOT %MVC — so even if we
   obtain them, the citation is still a unit mismatch.

4. **McGill 2013:** no free PDF; likely a textbook chapter not a journal
   paper. Citation is unverifiable.

The Task G citations are therefore of **three different failure modes**:

- **[MIS-ATTR]** — the paper exists and we can read it, but the paper
  does not report the number we cite (Hoozemans 2004, Anton 2001).
- **[MIS-ATTR-PROBABLE]** — the paper is paywalled but abstract/Research
  Mode review indicates the units cited do not match (Granata, de Looze,
  Snook, Nordander, McGill).
- The numerical %MVC values in our task_profiles may still be *plausible
  engineering estimates consistent with published ranges* — but **none
  are traceably sourced from the cited papers**.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| G1a | heavy_lift.shoulder | 0.45 | **[MIS-ATTR]** Hoozemans 2004 reports Nm/N only; no %MVC for deltoid (`ref_hoozemans2004.txt` L109-118) | Hoozemans 2004 — citation does not support the number |
| G1b | heavy_lift.ankle    | 0.10 | **[MIS-ATTR-PROBABLE]** Snook 1991 reports psychophysical kg, not %MVC | [UNVERIFIED] Snook 1991 unobtainable |
| G1c | heavy_lift.knee     | 0.40 | **[MIS-ATTR-PROBABLE]** Granata 1995 reports N/Nm, not %MVC | [UNVERIFIED] Granata 1995 unobtainable |
| G1d | heavy_lift.elbow    | 0.30 | **[MIS-ATTR]** Hoozemans 2004 does not report elbow %MVC | Hoozemans 2004 — no elbow data |
| G1e | heavy_lift.trunk    | 0.50 | **[MIS-ATTR-PROBABLE]** Granata 1995 reports EMG-derived N/Nm forces, not %MVC | [UNVERIFIED] Granata 1995 unobtainable |
| G1f | heavy_lift.grip     | 0.55 | **[MIS-ATTR]** Hoozemans 2004 does not report grip %MVC | Hoozemans 2004 — no grip data |
| G2a | light_sort.shoulder | 0.10 | **[MIS-ATTR-PROBABLE]** Nordander 2000 measures **trapezius only** during hospital cleaning + office work, not "light sorting"; reports muscular-rest % and gap frequency, not APDF %MVC | [UNVERIFIED] Nordander 2000 unobtainable |
| G2b | light_sort.ankle    | 0.05 | [DESIGN-undocumented] "standing load only" — no source claim | internal |
| G2c | light_sort.knee     | 0.05 | [DESIGN-undocumented] "standing load only" — no source claim | internal |
| G2d | light_sort.elbow    | 0.15 | **[MIS-ATTR-PROBABLE]** Nordander 2000 is trapezius-only | [UNVERIFIED] Nordander 2000 unobtainable |
| G2e | light_sort.trunk    | 0.10 | **[MIS-ATTR]** McGill 2013 is a textbook chapter, not a journal-reported %MVC | [UNVERIFIED] McGill 2013 unobtainable |
| G2f | light_sort.grip     | 0.20 | **[MIS-ATTR-PROBABLE]** Nordander 2000 is trapezius-only | [UNVERIFIED] Nordander 2000 unobtainable |
| G3a-f | carry demands       | 0.25,0.20,0.25,0.20,0.30,0.45 | **[MIS-ATTR-PROBABLE]** Snook 1991 reports max-acceptable kg/N, not %MVC | [UNVERIFIED] Snook 1991 unobtainable |
| G4a | overhead_reach.shoulder | 0.55 | **[MIS-ATTR]** Anton 2001 reports %RVE (not %MVC), deltoid range 30–154% not 50–60% (`ref_anton2001.txt` L400) | Anton 2001 — citation wrong in multiple ways |
| G4b | overhead_reach.ankle | 0.05 | [DESIGN-undocumented] | internal |
| G4c | overhead_reach.knee  | 0.10 | **[MIS-ATTR]** Anton 2001 does not measure knee | Anton 2001 — paper has right-arm muscles only |
| G4d | overhead_reach.elbow | 0.35 | **[MIS-ATTR]** Anton 2001 reports biceps 99–153% RVE, triceps 22–422% RVE, not 30–40% MVC | Anton 2001 |
| G4e | overhead_reach.trunk | 0.15 | **[MIS-ATTR]** Anton 2001 has NO trunk EMG | Anton 2001 — no trunk measurement in paper |
| G4f | overhead_reach.grip  | 0.30 | **[MIS-ATTR]** Anton 2001 does not measure grip | Anton 2001 — no grip measurement |
| G5a-f | push_cart demands  | 0.20,0.15,0.20,0.15,0.25,0.40 | **[MIS-ATTR-PROBABLE + MIS-ATTR]** de Looze 2000 reports N/Nm (Research Mode); Hoozemans 2004 reports N/Nm (confirmed) | [UNVERIFIED/MIS-ATTR] |

**This is the biggest single integrity risk in the audit.** Every numerical
value in task_profiles.yaml carries a citation the citation does not
support. A TMLR reviewer who opens any one of the 5 obtainable papers
(Hoozemans, Anton + paywalled abstracts of the rest) will see the
mismatch immediately.

**Recommended resolution (user decides):**

Option A — **re-source or remove**: find a different primary source that
actually reports %MVC per muscle per task. Candidate openly-accessible
alternatives flagged by Research Mode: Jørgensen 1988, Sjøgaard 1986,
NIOSH 1981 — none have verified free PDFs either. Honest path: this is
genuinely unsolved in the free literature.

Option B — **convert to N/Nm**: rewrite the manuscript to present demand
profiles as "fraction of muscle capacity informed by published trunk
moment (Nm) / push force (N) / EMG %RVE ranges" and drop the %MVC
framing. This is the most defensible path since the four obtainable
papers actually support %RVE / Nm / N ranges.

Option C — **admit engineering design**: frame the task_profiles as
"engineering design values informed by occupational ergonomics
literature, calibrated via NIOSH RLE (scripts/niosh_calibration.py)
for heavy_lift and carry; remaining tasks are design estimates within
published activity-intensity ranges." Remove the specific per-paper
citations.

My pre-critic recommendation: **Option C + keep NIOSH calibration**.
Option A likely yields no openly-verifiable source. Option B reworks the
whole pipeline. Option C is honest, TMLR-defensible, and preserves the
NIOSH-verified corner cases (heavy_lift, carry).

---

## TABLE H — NIOSH 1993 RLE constants — **UPDATED v2: all verified**

Location: `scripts/niosh_calibration.py:36-132`.

**v2 update:** Waters 1993 PDF is now extractable (`ref_waters1993.txt`).
Appendix A (`ref_waters1993.txt` L1644-1664) reproduces the formula
verbatim:

  RWL = LC × HM × VM × DM × AM × FM × CM
    LC = 23 kg (51 lbs)
    HM = 25/H (metric) — or (10/H) US customary
    VM = 1 - 0.0031 |V - 75|       ← *see Q1 below*
    DM = 0.82 + 4.5/D
    AM = 1 - 0.0032 A
    FM = from Table 7 (lookup)
    CM = from Table 6 (1.00 / 0.95 / 0.90 by good/fair/poor)

Cross-check vs code:

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| H1 | LC_METRIC | 23.0 kg | [OK] | Waters 1993 Appendix A (`ref_waters1993.txt` L1650) |
| H2 | HM formula | HM = 25/H | [OK] metric variant; H in cm | Waters 1993 Appendix A (`ref_waters1993.txt` L1652) |
| H3 | VM formula | VM = 1 - 0.003·\|V-75\| | **[Q1: 0.003 vs 0.0031]** paper reports "0.0031" — our code has "0.003" (niosh_calibration.py:54) | Waters 1993 Appendix A (`ref_waters1993.txt` L1654) |
| H4 | DM formula | DM = 0.82 + 4.5/D | [OK] | Waters 1993 Appendix A (`ref_waters1993.txt` L1656) |
| H5 | AM formula | AM = 1 - 0.0032·A | [OK] | Waters 1993 Appendix A (`ref_waters1993.txt` L1658) |
| H6 | CM values | 1.00/1.00/0.95/0.90 | [OK] | Waters 1993 Appendix A (`ref_waters1993.txt` L1662) |
| H7 | FM table (line 85-97) | see code | [LIKELY-OK] paper Table 7 is large; our implementation uses the V≥75cm column with duration tiers ≤1h/≤2h/≤8h — matches Waters 1993 Appendix A FM reference to Table 7 | Waters 1993 Appendix A + §6.4 + Table 7 |

**Q1 discrepancy:** `niosh_calibration.py:54` computes
`return 1.0 - 0.003 * abs(v_cm - 75.0)` but Waters 1993 Appendix A
(`ref_waters1993.txt` L1654) specifies **0.0031**. This is a ~3% error in
VM. For V=0 it matters: our formula gives VM = 1 - 0.225 = 0.775 but the
paper's formula gives VM = 1 - 0.2325 = 0.7675. Not catastrophic, but not
OK either for a paper claiming NIOSH fidelity.

Decision required: fix `0.003` → `0.0031` in `niosh_calibration.py:54`,
OR document that we are using a rounded 0.003 constant (a minority of the
ergonomics software implements 0.003 for simplicity; NIOSH-official
software uses 0.0031).

Note: the biomechanical compressive-force criterion **3.4 kN** (which
some secondary literature calls "NIOSH action limit") is described in
Waters 1993 as the "biomechanical criterion" or "compressive-force
criterion that defines increased risk", not as an "action limit" per se
(`ref_waters1993.txt` L258-259, L1737). If the manuscript uses the phrase
"NIOSH action limit", soften to "NIOSH biomechanical criterion (3.4 kN
L5/S1 disc compression)".

---

## TABLE I — MMICRL (Ref [8] Qiao et al. 2023)
*(unchanged from v1)*

Location: `hcmarl/mmicrl.py`, `config/hcmarl_full_config.yaml:90-109`.

Qiao 2023 objective: max_pi [lambda1*H[pi(tau)] - lambda2*H[pi(tau|z)]]
which when lambda1=lambda2 = mutual information I(tau; z). Our config
uses lambda1=lambda2=1.0 (symmetric MI). Math doc Eq 9-10 (lines 274,
292) derives this equivalence.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| I1 | lambda1 | 1.0 | [OK/DESIGN] | [8] symmetric weighted objective |
| I2 | lambda2 | 1.0 | [OK/DESIGN] | [8] |
| I3 | n_iterations | 150 | [DESIGN] | code hyperparam |
| I4 | hidden_dims | [64, 64] | [DESIGN] | code hyperparam |
| I5 | k_range | [1, 5] | [DESIGN] | [8] selects K by validation-NLL |
| I6 | k_selection | heldout_nll | [OK] | [8], Watanabe 2013 (verified — "Bayes free energy cannot be approximated by BIC in general", `ref_watanabe2013.txt` would be at ref15 — see F11 in logic audit), Vehtari 2017 |
| I7 | rescale_to_floor | true | [DESIGN] | internal fix from Phase B S1 |
| I8 | mi_collapse_threshold | 0.01 | [DESIGN] | internal fix |
| I9 | heldout_frac | 0.2 | [DESIGN] | standard 80/20 split |
| I10 | n_episodes_per_worker | 3 | [DESIGN] | code hyperparam |

Verdict: MMICRL setup is consistent with Qiao 2023; no citable errors.

---

## TABLE J — NSWF allocator (Ref [21] Kaneko-Nakamura 1979)
*(unchanged from v1)*

Location: `hcmarl/nswf_allocator.py`, `hcmarl/envs/reward_functions.py`.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| J1 | NSWF_EPSILON | 1e-3 | [DESIGN] rest-task surplus (Eq 31) | [21] allows epsilon>0 |
| J2 | default kappa | 1.0 | [DESIGN] | math doc Eq 32 |
| J3 | disagreement D(MF) = kappa*MF^2/(1-MF) | formula | [OK] | Eq 32 |
| J4 | solver (Hungarian + N rest cols) | linear_sum_assignment | [OK] | Kuhn 1955 / scipy |
| J5 | safety_weight (nswf_reward) | 5.0 | [DESIGN] | `reward_functions.py:63` — engineering choice, not literature-derived |
| J6 | MF clamp (disagreement_utility) | max(0, min(mf, 0.999)) | [DESIGN] | S-5 audit fix, prevents singularity at MF=1 |

Verdict: NSWF matches Kaneko-Nakamura + the math doc's Eq 31-33. No
errors. `safety_weight=5.0` needs a comment stating it is a design
parameter, not from any cited paper.

---

## TABLE K — proportional-controller gain kp (Ref [2] Xia & Frey-Law 2008) — **UPDATED v2 (inconsistency found)**

Ref [2] Xia & Frey-Law 2008 `ref2.txt` L220: "LD and LR were set at 10,
which were sufficient for the system to track TL quickly." In Xia's
formulation, LD and LR are DIFFERENT gains (development vs recovery),
with units consistent with F, R (time^-1).

**v2 finding: kp is inconsistent across the codebase.**

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| K1 | kp in `pettingzoo_wrapper.py:192` | **1.0** | [DESIGN] | `hcmarl/envs/pettingzoo_wrapper.py:192` — used in production training |
| K2 | kp in `warehouse_env.py:124, 345` | **1.0** | [DESIGN] | `hcmarl/warehouse_env.py` — legacy env |
| K3 | kp in `ThreeCCr.__init__` default | **1.0** | [DESIGN] | `hcmarl/three_cc_r.py:169` |
| K4 | kp in `real_data_calibration.py:51` | **10.0** | **[INCONSISTENT]** default for `predict_endurance_time()` is 10.0, but production env uses 1.0 | `hcmarl/real_data_calibration.py:51` |
| K5 | kp in `config/*_config.yaml` ecbf block | 1.0 | [OK with K1] | every config line 71 |

**The inconsistency is real and affects calibration.** The math doc's
Eq 35 refers to a single kp. The env uses 1.0. But when we calibrate
subjects from WSD4FEDSRM using `predict_endurance_time(kp=10.0)` in
`real_data_calibration.py`, we are implicitly using a different
controller gain than what the trained agent will see at runtime.

**A TMLR reviewer will ask:** "If kp=10 was used to fit subject F, R
parameters to observed endurance times, but the trained agent sees
kp=1, how are the calibrated (F_i, R_i) still correct?"

The answer is subtle and depends on whether the `predict_endurance_time`
function is actually used to fit real-data parameters during the
calibration pipeline. If it is, this is a genuine scientific error and
the calibration must be re-run. If `predict_endurance_time` is only used
for internal sanity tests, the mismatch is harmless but the default
should match 1.0 for clarity.

Action for you: trace whether `predict_endurance_time` feeds into
Path G's calibrated (F_i, R_i) output. If yes, harmonize kp. If no,
change default kp to 1.0 for consistency and add a test.

---

## TABLE L — Training / algorithm hyperparameters
*(unchanged from v1 — no citation claim)*

Location: every config yaml `algorithm:` block.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| L1 | lr_actor | 3e-4 | [OK/DESIGN] PPO standard | Schulman 2017 (not in refs) |
| L2 | lr_critic | 1e-3 | [OK/DESIGN] | Schulman 2017 |
| L3 | gamma | 0.99 | [OK/DESIGN] | MARL standard |
| L4 | gae_lambda | 0.95 | [OK/DESIGN] | Schulman 2016 GAE |
| L5 | clip_eps | 0.2 | [OK/DESIGN] PPO standard | Schulman 2017 |
| L6 | entropy_coeff | 0.05 → 0.01 anneal | [OK/DESIGN] Phase B M7 | internal |
| L7 | batch_size | 256 | [OK/DESIGN] | |
| L8 | hidden_dim (actor) | 64 | [OK/DESIGN] | |
| L9 | critic_hidden_dim | 128 | [OK/DESIGN] | |
| L10 | n_epochs | 10 | [OK/DESIGN] PPO standard | |
| L11 | max_grad_norm | 0.5 | [OK/DESIGN] PPO standard | |
| L12 | total_steps | 2,000,000 | [DESIGN] | internal |
| L13 | eval_interval | 50,000 | [DESIGN] | internal |
| L14 | checkpoint_interval | 100,000 | [DESIGN] Phase B C7 | internal |
| L15 | n_eval_episodes | 10 | [DESIGN] | internal |
| L16 | deterministic | true | [DESIGN] Phase B M6 cudnn.deterministic | internal |

Not a single numerical error; these are reproducible from any PPO paper.

---

## TABLE M — MAPPO-Lagrangian-specific
*(unchanged from v1)*

Location: `config/mappo_lag_config.yaml:57-64`.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| M1 | cost_limit | 0.1 | [DESIGN] dense-cost semantics documented in config | Phase B S3 |
| M2 | lambda_lr | 0.005 | [DESIGN] | |
| M3 | lambda_init | 0.5 | [DESIGN] | |

No citation claim, no error.

---

## TABLE N — environment constants
*(unchanged from v1 plus two new rows)*

Location: `config/*.yaml:7-11`, `hcmarl/envs/pettingzoo_wrapper.py:17-44`.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| N1 | n_workers | 6 | [DESIGN] | |
| N2 | max_steps | 480 | [DESIGN] 8-hour shift @ 1-min | |
| N3 | dt | 1.0 (minute) | [DESIGN] consistent with F,R units min^-1 | [3] uses min^-1 |
| N4 | kappa (disagreement) | 1.0 | [DESIGN] math doc Eq 32 | |
| N5 | warehouse_env max_steps default | 60 | [DESIGN] legacy env uses 60-min shift in tests | `hcmarl/warehouse_env.py:37` |

No error.

---

## TABLE O — Experiment-matrix / ablation (UPDATED v2 — new table)

Location: `config/experiment_matrix.yaml`.

| # | Variable | Current value | Status | Reference |
|---|---|---|---|---|
| O1 | headline seeds | [0..9] (10 seeds × 4 methods) | [DESIGN] Batch D | internal |
| O2 | ablation seeds | [0..4] (5 seeds × 5 rungs) | [DESIGN] Batch D | internal |
| O3 | curve_anchors_steps | 500K, 1M, 2M, 3M | [DESIGN] | internal |
| O4 | lazy_agent threshold | 0.1 (nats, task-selection entropy) | [DESIGN informed by Liu ICML 2023] | cited in config comment |
| O5 | lazy_agent window_steps | 100,000 | [DESIGN] | internal |

No citation claim violations.

---

## BOTTOM-LINE SUMMARY — v2

The audit has now identified **five classes of problem** across every
numerical constant in the execution path:

### Class 1 — Transcription errors in 3CC-r F, R (Tables A+B)
10 of 18 muscle F, R entries in code and math doc disagree with Ref [3]
Table 1 and Ref [5] line 206. The resulting 18 derived numbers
(delta_max, theta_min_max, Rr/F) are wrong. Downstream: math_doc.txt
Section 5.4.1 Remark 5.12, the "fatigue-resistance ranking" claim
(line 255-257), the hcmarl_full_config.yaml theta_max comments, and
the pettingzoo_wrapper.py theta_min_max comments are all wrong.

### Class 2 — Grip theta_max becomes infeasible after F,R correction (Table F)
Under corrected F,R,r for grip, theta_min_max=33.8%. Current theta_max=0.35
gives only 1.2pp margin — borderline-infeasible. Must be raised to 0.45
or grip demand must be revisited.

### Class 3 — Citation mis-attributions (Tables D, E, G, K)
- D1: Liu 2002 CV_F=0.36 is correct *for handgrip*, NOT elbow as our
  code/comment says. Fix comment text.
- E4: SLACK_PENALTY=1000.0 is a design choice, not supported by Ames
  2019 "large enough that slack is only ever non-zero" — this phrasing
  is not in the paper. Fix comment text.
- G1–G5: **every task demand profile cites a paper that does not
  report %MVC** — the cited papers report N / Nm / %RVE / psychophysical
  kg. This is the largest integrity risk in the codebase. Decision
  required between Options A/B/C (see Table G notes).
- K: kp is 1.0 in runtime env but 10.0 in `real_data_calibration.py`
  default. Either harmonize, or prove that the 10.0 path does not feed
  into calibrated F, R.

### Class 4 — Numerical drift (Table H)
- H3: VM formula uses `0.003` but Waters 1993 specifies `0.0031`. ~3%
  error in VM near V=0. Fix or document.
- Misnomer "NIOSH action limit" vs the Waters 1993 canonical phrase
  "NIOSH biomechanical criterion" / "3.4 kN compressive-force criterion".
  Update math doc / paper prose if applicable.

### Class 5 — Unreachable sources (Table F, Table G subset)
- Rohmert 1960: no free PDF, no English translation, no obtainable
  secondary source (El Ahrache 2009 also paywalled). `theta_max` values
  must be framed as engineering design choices, not literature-derived.
- Granata 1995, de Looze 1995, Snook 1991, Nordander 2000, McGill 2013:
  all paywalled with no preprint. Even if obtained, Research Mode
  predicts unit mismatch (N/Nm/kg vs %MVC). See Option C in Table G.

### Files to edit once you confirm the v2 findings
  - `hcmarl/three_cc_r.py:79-84`              (6 MuscleParams rows — Class 1)
  - `hcmarl/real_data_calibration.py:474-481` (POPULATION_FR — Class 1)
  - `hcmarl/real_data_calibration.py:484-497` (CV_F, CV_R comment text — Class 3/D1)
  - `hcmarl/real_data_calibration.py:51`      (kp default 10 → 1 OR justify — Class 3/K)
  - `hcmarl/ecbf_filter.py:28-31`             (SLACK_PENALTY comment — Class 3/E4)
  - `hcmarl/envs/pettingzoo_wrapper.py:64-79` (theta_min_max comment — Class 1)
  - `config/hcmarl_full_config.yaml:12-25`    (muscle_groups + theta_max comments — Classes 1, 2)
  - `config/mappo_config.yaml:13-18`          (Class 1)
  - `config/ippo_config.yaml:13-18`           (Class 1)
  - `config/mappo_lag_config.yaml:13-18`      (Class 1)
  - `config/dry_run_50k.yaml`                 (check + edit — Class 1)
  - `config/ablation_*.yaml`                  (check + edit — Class 1)
  - `config/probe_500k.yaml`, `watch_1m.yaml` (check + edit — Class 1)
  - `config/task_profiles.yaml`               (rewrite citation block — Class 3/G)
  - `scripts/niosh_calibration.py:54`         (VM 0.003 → 0.0031 — Class 4/H3)
  - `MATHEMATICAL MODELLING.pdf` Table 1 (line 240-250), the verification
    paragraph (line 252-257), Section 5.4 Remark 5.12 (line 672),
    Section 7 Eq 35 footnote (Logic Audit G2), and every downstream
    "Rr/F = 0.596" style reference.
  - every test that pins the old numbers (~tests/test_batch_f.py,
    tests/test_round* — grep pending).

### Open questions requiring your decision
  1. **[F6]** grip theta_max under corrected F,R leaves only 1.2pp margin.
     Raise to 0.45 (matches elbow) or keep 0.35 and accept ECBF
     aggressiveness?
  2. **[C]** visual confirmation of the joint → b1 alignment in Ref [6]
     Table 2 (you open ref6.pdf page 34).
  3. **[G]** choose Option A/B/C for task demand profile citations. My
     recommendation: **Option C** (reframe as engineering design values
     calibrated against NIOSH RLE where possible; drop the per-paper
     citations that do not support %MVC framing).
  4. **[H3]** VM formula 0.003 vs 0.0031 — fix to NIOSH spec, or document
     the rounding.
  5. **[K]** kp=10 vs kp=1 inconsistency — confirm whether
     `predict_endurance_time` feeds into Path G calibrated F_i, R_i.
  6. **[D1]** rewrite CV_F / CV_R comment to "handgrip Liu 2002", not
     "elbow Liu 2002".
  7. **[E4]** rewrite SLACK_PENALTY comment to drop the unsupported
     Ames 2019 attribution.

Once you verify the above against the PDFs yourself, tell me which rows
to apply and I will batch the edits + re-run the test suite before you
restart training.

---

## Appendix — every verifying quote (v2 additions)

For auditability, the verbatim PDF-extracted text that the new v2 rows
rely on:

**Liu 2002 Table 2 (handgrip, 10 subjects):** `ref1_liu2002.txt` L687-699
```
 1 0.024 200     0.48 418 407               97.38    1.11 407   97.37    1.10 0.0115 4.800 41.7 86.8 28.2 0.21
 ... (10 rows) ...
Mean 0.0206 254.0 0.398 412.5 401.2         97.16    1.379 401.3 97.19   1.378 0.0084 4.132 57.3 154.7 41.5 0.264
 SD  0.0075 184.8 0.077 71.0 72.6            1.32    0.291 72.2  1.18    0.286 0.0036 1.343 29.8 98.5 22.9 0.079
```
→ Mean F=0.0206, SD=0.0075 ⇒ CV_F=0.364. Mean R=0.0084, SD=0.0036 ⇒ CV_R=0.429.

**Hoozemans 2004 abstract units:** `ref_hoozemans2004.txt` L109-118
"exerted push/pull forces, net moments at the low back and shoulders,
 compressive and shear forces at the low back, and compressive forces at
 the glenohumeral joint"
→ N and Nm only; no %MVC.

**Anton 2001 %RVE definition + Table 2:** `ref_anton2001.txt` L244-251, L400-408
"normalized to a reference voluntary contraction (MVC) and was unchanged
 ... drill bit upward into the closest hole position (% Reference
 Voluntary Exertion (RVE))"
Table 2 row verbatim:
"Anterior deltoid  High  30.18  2.41  53.88  2.98  115.77  3.89"
→ %RVE not %MVC; anterior deltoid range 30–154% not 50–60%; no trunk
  measurement.

**Waters 1993 Appendix A:** `ref_waters1993.txt` L1644-1664
```
RWL = LC × HM × VM × DM × AM × FM × CM
  LC = load constant = 23 kg  51 lbs
  HM = horizontal multiplier = (25/H) metric, (10/H) US
  VM = vertical multiplier = (1 - (0·0031 |V - 75|)) metric
  DM = distance multiplier = (0·82 + (4·5/D))
  AM = asymmetric multiplier = (1 - (0·00324·A))     ← likely typo in paper extraction
  FM = frequency multiplier (from table 7)
  CM = coupling multiplier (from table 6)
```
→ VM coefficient is **0.0031**, not 0.003. AM coefficient is 0.0032
   (the "0.00324" in extraction is likely a pdftotext OCR artifact on
   a dot that should read "0·0032 A"; `ref_waters1993.txt` L1658 shows
   "AM = asymmetric multiplier = (1 - (0·00324)" which reads more
   naturally as 0.0032 × A). Our code uses 0.0032 which is consistent
   with the Appendix and with all secondary sources.

**Ames 2019 ECC CLF-CBF QP:** `ref_ames2019.txt` L271-282
"δ is a relaxation variable that ensures solvability of the QP as
 penalized by p > 0 (i.e., to ensure the QP has a solution one must
 relax the condition on stability to guarantee safety)."
→ Only "p > 0" is required; no magnitude prescription.

---

## End of v2 audit.
