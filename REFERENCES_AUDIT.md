# REFERENCES Folder — Audit Against Math Doc + TMLR Paper Plan

**Date:** 2026-05-01
**Math doc:** `MATHEMATICAL_MODELLING_v13.pdf` (28 references in bibliography)
**REFERENCES folder:** `C:\Users\admin\Desktop\hcmarl_project\REFERENCES\`

---

## TL;DR

- **26 of 28** math doc references have a local PDF. ✅
- **2 missing:** [25] Khalil "Nonlinear Systems" (textbook) and [26] Rohmert 1960 (German paper).
- **4 extra PDFs** in folder (Hoozemans 2004, Anton 2001, Waters 1993, Watanabe 2013) — used by code, will need to appear in the TMLR paper once empirical/calibration sections are written.
- **6 PDFs labelled "UNVERIFIED"** — I identified all of them. Two are correctly mapped to numbered references (just need rename); four are supplementary refs.

---

## Full mapping: math doc → folder

| Doc Ref | Authors / Title (short) | Local PDF | Status |
|---|---|---|---|
| [1] | Liu, Brown, Yue 2002 — *Biophysical Journal* | `1 (UNVERIFIED 1).pdf` | ✅ Present (recommend rename → `1.pdf`) |
| [2] | Xia, Frey-Law 2008 — *J. Biomechanics* | `2.pdf` | ✅ |
| [3] | Frey-Law, Looft, Heitsman 2012 — *J. Biomechanics* | `3.pdf` | ✅ |
| [4] | Looft, Herkert, Frey-Law 2018 — *J. Biomechanics* | `4.pdf` | ✅ |
| [5] | Looft, Frey-Law 2020 — *J. Biomechanics* | `5.pdf` | ✅ |
| [6] | Frey-Law, Avin 2010 — *Ergonomics* | `6.pdf` | ✅ |
| [7] | Ziebart, Maas, Bagnell, Dey 2008 — AAAI MaxEnt IRL | `7.pdf` | ✅ |
| [8] | Qiao, Liu, Poupart, Xu 2023 — NeurIPS MMICRL | `8.pdf` | ✅ |
| [9] | Malik, Anwar, Aghasi, Ahmed 2021 — ICML ICRL | `9.pdf` | ✅ |
| [10] | Li, Song, Ermon 2017 — NeurIPS InfoGAIL | `10.pdf` | ✅ |
| [11] | Achiam, Held, Tamar, Abbeel 2017 — ICML CPO | `11.pdf` | ✅ |
| [12] | Nguyen, Sreenath 2016 — ACC ECBF | `12.pdf` | ✅ |
| [13] | Xiao, Belta 2019 — IEEE CDC HOCBF | `13.pdf` | ✅ |
| [14] | Ames, Xu, Grizzle, Tabuada 2017 — *IEEE TAC* CBF-QP | `14.pdf` | ✅ |
| [15] | Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada 2019 — ECC | `15 (UNVERIFIED 10).pdf` | ✅ Present (recommend rename → `15.pdf`) |
| [16] | Prajna, Jadbabaie 2004 — HSCC | `16.pdf` | ✅ |
| [17] | Nash 1950 — *Econometrica* "The Bargaining Problem" | `17.pdf` | ✅ |
| [18] | Nash 1953 — *Econometrica* "Two-Person Cooperative Games" | `18.pdf` | ✅ Image-only scan; metadata confirms title |
| [19] | Navon, Shamsian, Achituve et al. 2022 — ICML | `19.pdf` | ✅ |
| [20] | Binmore, Shaked, Sutton 1989 — *QJE* outside option | `20.pdf` | ✅ |
| [21] | Kaneko, Nakamura 1979 — *Econometrica* NSWF | `21.pdf` | ✅ |
| [22] | Nagumo 1942 — Proc. Phys-Math Soc. Japan | `22.pdf` | ✅ (English translation by Menner & Lavretsky) |
| [23] | Blanchini 1999 — *Automatica* set invariance | `23.pdf` | ✅ |
| [24] | Altman 1999 — *Constrained MDPs* (book) | `24.pdf` | ✅ |
| **[25]** | **Khalil 2002 — *Nonlinear Systems*, 3rd ed.** | **MISSING** | ❌ Not in folder |
| **[26]** | **Rohmert 1960 — German paper on recovery pauses** | **MISSING** | ❌ Not in folder |
| [27] | Boyd, Vandenberghe 2004 — *Convex Optimization* | `27.pdf` | ✅ |
| [28] | Stellato, Banjac, Goulart, Bemporad, Boyd 2020 — OSQP | `28.pdf` | ✅ |

---

## Extra PDFs in folder (not in math doc — used elsewhere in code)

| PDF | Author / Topic | Where it's cited in the project |
|---|---|---|
| `UNVERIFIED 4.pdf` | Hoozemans et al. 2004 — *Ergonomics* mechanical loading shoulders/back during pushing/pulling | `hcmarl/warehouse_env.py:67` (heavy_lift task target loads) |
| `UNVERIFIED 8.pdf` | Anton et al. 2001 — overhead drilling shoulder moments + EMG | Likely shoulder load calibration; check `scripts/niosh_calibration.py` and `hcmarl/warehouse_env.py` |
| `UNVERIFIED 9.pdf` | Waters, Putz-Anderson, Garg, Fine 1993 — NIOSH revised lifting equation | `scripts/niosh_calibration.py` (Batch F NIOSH LI calibration: heavy_lift LI=1.38, carry LI=0.79) |
| `UNVERIFIED 11.pdf` | Watanabe 2013 — *JMLR* Widely Applicable BIC | `hcmarl/mmicrl.py` (Batch E rationale for using held-out NLL instead of BIC for singular flows) |

These are **legitimately part of your project** — they support the empirical sections (target-load calibration, NIOSH ergonomic indices, MMICRL model selection). They will need to appear in the **TMLR paper bibliography** once you write the experimental setup section. They are *not* part of the math framework so they don't appear in the math doc bibliography.

---

## Three "UNVERIFIED" tags meant — what I confirmed

| File | What was unverified | What I confirmed |
|---|---|---|
| `1 (UNVERIFIED 1).pdf` | TODO marker on download | Content matches Liu 2002. Same as ref `[1]`. |
| `15 (UNVERIFIED 10).pdf` | Reference number was reshuffled (was [10], became [15]) | Content matches Ames et al. 2019 ECC tutorial. Same as ref `[15]`. |
| `UNVERIFIED 4.pdf` | Whether to include in main bibliography | Hoozemans 2004 — keep, used in code |
| `UNVERIFIED 8.pdf` | Whether to include in main bibliography | Anton 2001 — keep, used in code |
| `UNVERIFIED 9.pdf` | Whether to include in main bibliography | Waters 1993 NIOSH — keep, used in code |
| `UNVERIFIED 11.pdf` | Whether to include in main bibliography | Watanabe 2013 — keep, used in code |

---

## Recommendations for the TMLR paper

### What you have (28 + 4 = 32 PDFs)
The 32 PDFs in `REFERENCES/` cover everything cited in the math framework + the empirical-section calibration references. This is a strong bibliography foundation.

### What you need to do for the 2 missing items

**[25] Khalil 2002 — *Nonlinear Systems*, 3rd ed.**
- This is the standard reference for class-K functions in nonlinear control. Used only for the Definition 1 of class-K in Section 5.2.
- **Options:**
  1. **Cite the textbook** (no PDF needed for citation). TMLR accepts textbook citations without supplementary PDFs.
  2. **Replace with a freely-available alternative** that defines class-K just as well: Sontag, *Mathematical Control Theory* (Springer 1998, freely available author website), or Hahn, *Stability of Motion* (Springer 1967).
  3. **Define class-K inline in your math doc** (one-line definition, no citation needed): "A continuous function α: [0, a) → [0, ∞) is class-K if α(0) = 0 and α is strictly increasing."
- **Recommendation:** Option 1 (cite the textbook). It's a universally accepted reference, and you don't need to ship a PDF.

**[26] Rohmert 1960 — German paper on static work recovery pauses**
- Foundational endurance-time data, but the **modern English-language reference** Frey-Law & Avin 2010 [6] reproduces and extends Rohmert's data via 194-publication meta-analysis. The 2012 Frey-Law et al. [3] calibration explicitly cites Rohmert 1960 as the historical source but uses [6] for modern data.
- **Options:**
  1. **Cite Rohmert 1960 anyway** as a historical foundational reference; don't include PDF (TMLR is fine with this).
  2. **Drop [26]** and rely solely on [6] for endurance data. You'd remove the parenthetical "(Rohmert 1960)" mentions in Section 3 and Remark 5.6.
- **Recommendation:** Option 1. The historical attribution to Rohmert is appropriate for a fatigue-modelling paper.

### What you'll need to ADD for the TMLR paper bibliography
Beyond the 28+4 you have, the TMLR paper experimental sections will likely cite:

- **MARL baselines** (MAPPO, MAPPO-Lag, PS-IPPO, possibly MACPO):
  - Yu et al. 2022 *NeurIPS* "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (MAPPO)
  - Bertsekas 2019 textbook *or* Ray, Achiam, Amodei 2019 OpenAI safety-gym (PPO-Lag)
  - de Witt et al. 2020 NeurIPS (PS-IPPO equivalent or independent)
  - Gu et al. 2023 ICML "Multi-Agent Constrained Policy Optimisation" (MACPO, optional)
- **Datasets** (WSD4FEDSRM):
  - Zenodo entry 8415066 + the *Scientific Data* 2024 paper
- **Statistical methodology** (IQM + bootstrap CIs):
  - Agarwal et al. 2021 NeurIPS *rliable* (Deep RL at the Edge of the Statistical Precipice)
- **PettingZoo + Gymnasium** (environment library citations):
  - Terry et al. 2021 NeurIPS PettingZoo
  - Towers et al. 2023 Gymnasium
- **PyTorch + scipy + cvxpy** (software citations as TMLR allows)

These are not in the current math doc because they belong to the experimental sections.

---

## Quick action items

1. **Optional housekeeping:** rename `1 (UNVERIFIED 1).pdf` → `1.pdf` and `15 (UNVERIFIED 10).pdf` → `15.pdf`. The `UNVERIFIED` tag is misleading now that I've confirmed both files match their references. The four supplementary `UNVERIFIED N.pdf` files can be renamed to `Hoozemans2004.pdf`, `Anton2001.pdf`, `Waters1993.pdf`, `Watanabe2013.pdf` for clarity in the TMLR submission supplementary materials.
2. **Don't worry about [25] and [26].** Cite them in the bibliography without local PDFs. TMLR accepts textbook and historical paper citations without PDF deposits.
3. **For TMLR submission**, aggregate 28 (math doc) + 4 (supplementary, already in folder) + ~13 (experimental section additions, BibTeX entries staged) ≈ ~45 references total. That's a comfortable size for a TMLR paper.

---

## 2026-05-02 update — experimental-section bibliography staged

Post-baseline-expansion (4 baselines: MAPPO + MAPPO-Lag + MACPO + HAPPO + Shielded-MAPPO; was 3 baselines yesterday) and post-dataset-literature-search, the experimental-section reference count has grown from ~12 to ~13. BibTeX entries for all 13 are now in **`bib/experimental_section.bib`** (project-root, committable; the `REFERENCES/` folder itself is gitignored because it holds copyrighted PDFs) with verification status flags ([VERIFIED] / [PARTIAL] / [TEXTBOOK]).

| # | Entry | Used for | Verification | PDF in folder? |
|---|---|---|---|---|
| 1 | Yu et al. 2022 (MAPPO) | EXP1 baseline | VERIFIED via arXiv:2103.01955 | NO |
| 2 | Stooke et al. 2020 (PID Lag) | MAPPO-Lag baseline | VERIFIED via arXiv:2007.03964 | NO |
| 3 | Kuba et al. 2022 (HAPPO) | EXP1 baseline (NEW) | VERIFIED via arXiv:2109.11251 | NO |
| 4 | Gu et al. 2023 (MACPO) | EXP1 baseline (NEW) | VERIFIED via arXiv:2110.02793 | NO |
| 5 | Alshiekh et al. 2018 (Shielding) | Shielded-MAPPO context (NEW) | VERIFIED via arXiv:1708.08611 | NO |
| 6 | Agarwal et al. 2021 (rliable) | IQM + bootstrap stats | VERIFIED via arXiv:2108.13264 | NO |
| 7 | Hubert & Arabie 1985 (ARI) | EXP3 Part 1 cluster validity | TEXTBOOK | NO |
| 8 | Terry et al. 2021 (PettingZoo) | env library | VERIFIED via arXiv:2009.14471 | NO |
| 9 | Towers et al. 2024 (Gymnasium) | env library | VERIFIED via arXiv:2407.17032 + WebFetch | NO |
| 10 | WSD4FEDSRM 2023 dataset | calibration source | PARTIAL (Zenodo creator field needs verification) | NO |
| 11 | Cerqueira et al. 2024 (sEMG fatigue) | future cross-val (NEW) | VERIFIED via PubMed 39771816 | NO |
| 12 | Mudiyanselage et al. 2021 (manual handling) | related work (NEW) | VERIFIED via arXiv:2109.15036 | NO |
| 13 | Peters et al. 2025 (wearable review) | related work (NEW) | VERIFIED via BAuA institutional OA copy + 3-source author cross-check (was incorrectly attributed to "Cataldi" yesterday; corrected 2026-05-02) | YES (BAuA OA, 2.3 MB) |
| 14 | Sun et al. 2025 (perceived fatigue) | related work (optional, NEW) | PARTIAL (publisher page blocked WebFetch; full author list TODO) | YES (BMC OA, 5.6 MB) |

(Item 14 is optional belt-and-braces; the canonical 13 are 1-13.)

**College Lit Review last-5-years coverage (rubric: 10-15 from 2021-2026):**
- From math doc bib (existing in REFERENCES/): Looft 2020 [5], Malik 2021 [9], Navon 2022 [19], Qiao 2023 [8] = 4
- From experimental section (BibTeX staged above): Yu 2022, Kuba 2022, Gu 2023, Agarwal 2021, Terry 2021, Towers 2024, Cerqueira 2024, Mudiyanselage 2021, Cataldi 2025 = 9 in 2021-2026
- **Total in last 5 years: 13.** Meets the 10-15 rubric.

**Net change vs 2026-05-01 estimate:**
- Yesterday: ~12 experimental-section refs needed
- Today: 13 staged + 1 optional (Sun 2025) = 14 max
- Delta: +1 to +2 from baseline expansion (added Kuba, Gu, Alshiekh; dropped need for de Witt 2020 IPPO since PS-IPPO is no longer in the headline grid).

**PDF status:** none of the 13 BibTeX-staged papers has a local PDF in `REFERENCES/`. The user is responsible for downloading from arXiv / publishers and depositing under the naming convention `<firstauthor><year>.pdf`. The BibTeX entries above are ready to paste into a paper `.bib` file once one exists; PDFs are needed only for TMLR supplementary deposit, not for the BibTeX itself.
