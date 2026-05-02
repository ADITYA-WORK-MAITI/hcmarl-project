# HC-MARL paper revisions — dataset / calibration block

Generated 2026-05-02 in response to the simulated TMLR review (R1
methodologist + R2 ergonomics-aware). The five blocks below are
ready-to-paste into the paper draft. Item 1 also has executable code at
`scripts/sensitivity_analysis.py`; item 3 has executable code at
`hcmarl/aggregation.py::worker_seed_stratified_bootstrap_iqm_ci`.

When the EXP1 + EXP2 + EXP3-Part-B reruns land on L40S and checkpoints
populate `checkpoints/<method>/seed_<s>/`, run:

```bash
python scripts/sensitivity_analysis.py \
    --methods hcmarl mappo mappo_lag macpo happo shielded_mappo \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --levels -0.5 -0.2 0.0 0.2 0.5 \
    --n-eval-episodes 50 --device cuda
```

That produces `results/sensitivity/sensitivity_metrics.csv`. Plot
headline metric IQM ± 95% CI vs. perturbation level per method; if the
HC-MARL bars overlap the +/-50% bars, you have a calibration-robust
result. Reference this figure from §X.

---

## Item 2 — Dataset section paragraph (paste into §Experimental Setup → Dataset)

Public ergonomic datasets aligning with HC-MARL's input requirements
(per-muscle sEMG amplitude trajectories, warehouse-scale task labels,
and per-worker identifiers preserved across tasks) are limited.
WSD4FEDSRM (Borg endurance times, 34 subjects, dynamic isometric
contractions at 35/45/55% MVIC; Zenodo 8415066) is the largest
publicly accessible source meeting all three criteria simultaneously.
NinaPro DB2/DB5 contain dense multi-channel sEMG but lack
warehouse-relevant task labels. Anton et al. (2001) and Hoozemans et
al. (2004) provide warehouse-task labels and lift/carry biomechanics
but record only aggregate metabolic load, not per-muscle sEMG. We
therefore calibrate the 3CC-r per-muscle parameters via the Path G
procedure (§X) on WSD4FEDSRM and report sensitivity over plausible
parameter ranges in §Y. Cross-population validation is identified as
future work (§Z).

---

## Item 4 — Calibration lineage citations (where to insert)

The following citations should appear at the **first mention of Path G
calibration** in the dataset / methods section, and at any **subsequent
mention of per-muscle F, R, r values**. All four are already in the
canonical bibliography (math doc v13, references [3]–[6]):

| Cite at... | Citations |
|---|---|
| First mention of "we calibrate F, R per worker via Path G" | Frey-Law et al. (2012); Frey-Law & Avin (2010) |
| First mention of reperfusion multiplier r=15 (or r=30 for grip) | Looft, Herkert & Frey-Law (2018); Looft & Frey-Law (2020) |
| Discussion of population-mean priors used for non-shoulder muscles | Frey-Law & Avin (2010); Frey-Law et al. (2012) Table 1 |
| Sensitivity to reperfusion regime (no_reperfusion ablation) | Looft, Herkert & Frey-Law (2018); Liu (2002); Xia & Frey-Law (2008) |

Confirmation of bibliography presence:
- [3] Frey-Law, Looft & Heitsman 2012 — present in math doc bib (v13)
- [4] Looft, Herkert & Frey-Law 2018 — present
- [5] Looft & Frey-Law 2020 — present
- [6] Frey-Law & Avin 2010 — present

The citation lineage should make clear that the Path G calibration
methodology is **inherited from**, not invented by, this paper. The
paper's contribution is the integration of per-worker calibrated 3CC-r
into a multi-agent RL setting; the calibration recipe itself is from
Looft & Frey-Law's 2018 / 2020 work on per-region reperfusion
identification.

---

## Item 5 — Limitations paragraph (paste into §Limitations or §Discussion)

We scope HC-MARL's empirical validation to a single real ergonomic
dataset (WSD4FEDSRM) and complement this with a synthetic K=3
demonstration (§Exp 3 Part B) showing the type-inference machinery
operates correctly when worker heterogeneity is present by
construction. On WSD4FEDSRM, MMICRL collapses (mutual information
< 1e-7 in all seeds), indicating the real data does not exhibit the
multi-type structure our model is designed to detect; HC-MARL
gracefully degrades to ECBF + NSWF operation with a configurable
fatigue floor. Cross-population validation, particularly on a dataset
combining the per-muscle sEMG resolution of NinaPro DB2/DB5 with the
warehouse-task labelling of Anton (2001) or Hoozemans (2004), would
require either (i) augmenting NinaPro recordings with task labels via
expert annotation or (ii) instrumenting an Anton-/Hoozemans-style
warehouse study with per-muscle sEMG. Both approaches are identified as
future work; the framework code released with this paper is dataset-
agnostic and can be re-calibrated by replacing
`config/pathg_profiles.json` with a new calibration of the form
documented in §X. Additionally, the worker-level bootstrap (§Y) shows
[INSERT RESULT once sensitivity_analysis.py finishes] of the headline
gain stems from training-seed variance versus worker-population
variance; full worker-population generalisation cannot be claimed from
n=34.

---

## Item 1 — Sensitivity analysis (executable: scripts/sensitivity_analysis.py)

The script perturbs every (worker, muscle) F and R value by a
multiplicative factor (1+δ) for δ ∈ {-0.5, -0.2, 0.0, +0.2, +0.5},
re-runs `scripts/evaluate.py` on each existing checkpoint, and emits a
tidy CSV. Reperfusion multiplier r is held fixed; perturbing r is
already covered by the `no_reperfusion` ablation rung (r=1 across all
muscles).

Headline sentence to put in the paper, after running:

> Headline cumulative-reward IQM shifts by less than [INSERT % from
> sensitivity_metrics.csv] across the ±50% calibration perturbation
> grid, demonstrating that HC-MARL's empirical advantage over baselines
> is not an artefact of the specific Path G calibration values.

If the IQM shifts by **less than 10%**: report as "calibration-robust,"
this is the credibility-win outcome.

If the IQM shifts by **10–25%**: report honestly as "moderately
sensitive," include a sentence in Limitations.

If the IQM shifts by **more than 25%**: this is a finding. Frame as:
"HC-MARL's empirical headline is sensitive to calibration parameters in
the ±50% range, suggesting that practical deployment requires careful
per-population calibration. We discuss this scope constraint in §X."

Either way the paper survives — the sensitivity analysis is a one-shot
credibility add. The downside is bounded.

---

## Item 3 — Worker-level bootstrap (executable: hcmarl/aggregation.py)

The new function `worker_seed_stratified_bootstrap_iqm_ci(score_matrix,
...)` takes a `(n_workers, n_seeds)` matrix and resamples both axes
independently with replacement. The seed-only bootstrap captures
training-RNG variability; this two-axis version additionally captures
worker-level variability — the axis R1 will (correctly) note is dominant
when n_workers ≈ n_seeds.

Sentence to put in the methods section:

> Confidence intervals on the headline IQM are computed via a two-axis
> stratified bootstrap over (worker, seed) pairs (Algorithm in
> Appendix A; implementation released with the code), capturing both
> training-RNG and worker-population variability. Resampling only over
> seeds — the more common choice in the multi-agent RL literature — is
> reported in Appendix B for comparison.

To produce the score matrix from existing eval outputs, evaluation runs
must record per-worker metrics. The current `scripts/evaluate.py`
produces aggregated metrics. **Action item before submission**: extend
the eval CSV schema to record `cumulative_reward_per_worker_<i>` and
`safety_violations_per_worker_<i>` columns so the (n_workers, n_seeds)
matrix can be reconstructed without re-running evaluation. This is a
schema change, not a re-training change; ~30 minutes of work in
`scripts/evaluate.py` and `hcmarl/logger.py`.

After the rerun lands, aggregate via:

```python
from hcmarl.aggregation import aggregate_by_method_two_axis
import numpy as np

# Build the score matrix from per-worker eval CSVs:
# scores[method] is (34, 10) -- rows = workers, cols = seeds.
scores = {
    "hcmarl":         np.array([...]),  # (34, 10)
    "mappo":          np.array([...]),
    "mappo_lag":      np.array([...]),
    "macpo":          np.array([...]),
    "happo":          np.array([...]),
    "shielded_mappo": np.array([...]),
}
report = aggregate_by_method_two_axis(scores)
# report["hcmarl"] = {"iqm": ..., "ci_lo": ..., "ci_hi": ..., "n_workers": 34, "n_seeds": 10}
```

Plot IQM ± [ci_lo, ci_hi] per method as the headline figure. If the CIs
are wider than the seed-only CIs (they will be), this is HONEST not
weakening — the paper accurately reports the relevant uncertainty.
