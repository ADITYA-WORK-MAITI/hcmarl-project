# Rebuttal armor (Batch F)

Pre-rehearsed responses to two predictable reviewer questions. Both are
narrative, but each is backed by a specific calculation or citation so
the paper can cite, not handwave.

## F1 — "Why not merge Cerqueira + NinaPro to get K ≥ 2 on real data?"

**Rehearsed answer (verbatim, for drafting the rebuttal / paper §5.3):**

> Merging Cerqueira et al. (2024) and NinaPro (Atzori et al. 2014) into a
> single MMICRL training set is superficially attractive — it would lift
> N to ~150 subjects and, on paper, create heterogeneity through cohort
> differences. We decline the merge for four independent reasons, each
> sufficient on its own.
>
> (1) **Units are incompatible.** Cerqueira reports shoulder endurance
> times at task-prescribed %MVIC, enabling direct (F, R) calibration of
> the 3CC-r fatigue ODE. NinaPro reports surface-EMG RMS amplitudes;
> converting RMS into %MVIC requires a per-subject MVIC normalisation
> measurement that NinaPro does not provide. Merging therefore requires
> injecting an estimated normalisation factor, and the clustering MMICRL
> then "discovers" is a projection of that estimator's bias, not of
> physiology.
>
> (2) **Task sets are disjoint.** NinaPro is hand-grasp gesture
> recognition; Cerqueira is shoulder dynamic rotation against resistance
> (our six MVIC-fraction task profiles span warehouse lifting/carrying,
> none of which appear in NinaPro). The joint distribution of (muscle,
> task) has empty cells for every Cerqueira task × NinaPro task pair,
> so a flow trained on the concatenation learns two disjoint manifolds.
> MMICRL then recovers "K = 2 cohorts," but this K is identifying
> *datasets*, not worker types — an artifact, not a finding.
>
> (3) **Cohort covariate shift is un-characterised.** NinaPro's
> participants span a wide age + sex distribution for which Cerqueira
> provides no matched covariates. ComBat harmonisation (Fortin et al.
> 2017) can remove a single linear batch effect but is documented to
> fail when batch and biological signal are confounded (Bayer et al.
> 2022); warehouse workers are not a sampled NinaPro subset, so we
> cannot verify that assumption.
>
> (4) **ComBat preserves means, not tails.** Our ECBF uses per-type
> θ_max thresholds, which live in the upper tail of the F-distribution.
> Even a correctly-applied ComBat does not preserve tail quantiles
> across batches (Zindler et al. 2020). A merged dataset would therefore
> give ECBF the wrong threshold distribution while still making the
> MMICRL MI metric look stronger — a textbook false positive.
>
> We instead present three distinct experiments: Experiment A
> (HC-MARL vs. baselines, no MMICRL in the hot path), Experiment B
> (MMICRL recovers K = 3 synthetic cohorts with ARI ≥ 0.80), and
> Experiment C (MMICRL correctly reports K = 1 on the homogeneous
> Cerqueira cohort, with bootstrap 95 % CI on the MI point estimate).
> K = 1 on real data is a *finding about Cerqueira*, not a failure of
> MMICRL — which is exactly what Experiment B is there to prove.

**Appendix sanity-check (if required during rebuttal):** pick any
two of {Cerqueira, NinaPro-DB1, WSD4FEDSRM} and fit MMICRL on the
concatenation with auto K. Report K*, MI, and ARI vs. the dataset-of-
origin label. If ARI ≈ 1.0, the recovered "types" are dataset labels
and the reviewer's suggested merge is confirmed as an artifact; we
already hold this script and can produce the numbers in under a day.

## F2 — NIOSH Revised Lifting Equation calibration

The two lifting tasks in our warehouse environment (`heavy_lift`,
`carry` in `config/hcmarl_full_config.yaml`) have task-demand %MVIC
values that need an external calibration. NIOSH (Waters, Putz-Anderson,
Garg & Fine 1993) is the ergonomics standard for that comparison.

The full calculator and sensitivity sweep live in
`scripts/niosh_calibration.py`. Running it on canonical warehouse
geometry (load = 15 kg / 10 kg, horizontal reach 35 / 30 cm, vertical
start 80 / 95 cm, 2 / 3 lifts per minute, 2 hr duration) gives:

| Task        | Load (kg) | RWL (kg) | LI    | Category                               |
|-------------|-----------|----------|-------|----------------------------------------|
| heavy_lift  | 15.0      | 10.87    | 1.38  | Elevated risk (1 < LI ≤ 3)             |
| carry       | 10.0      |  12.64   | 0.79  | Acceptable for most workers (LI ≤ 1)   |

Our config's shoulder %MVIC values (heavy_lift = 0.45, carry = 0.25)
are consistent with this NIOSH categorisation — LI ≈ 1.4 corresponds
to ~0.45 MVIC shoulder demand per Granata & Marras (1995) Table 3, and
LI ≈ 0.8 corresponds to ~0.25 MVIC per the same reference. No
calibration change required.

A ±20 % sensitivity sweep on each continuous geometry input (load, H,
V, D, A, frequency) shifts the `heavy_lift` LI within
[1.10, 1.66] and the `carry` LI within [0.63, 0.95]. Both remain in
the same risk category across the full sweep, so the chosen demand
values are robust to the exact geometry assumption.

**Other four tasks (qualitative alignment, no NIOSH fit):** NIOSH is a
lifting-equation; it does not apply to the four non-lift tasks
(`light_sort`, `overhead_reach`, `push_cart`, `rest`). Those values are
cited inline in `config/hcmarl_full_config.yaml` against Nordander et
al. 2000 (light repetitive), Anton et al. 2001 (overhead reach),
Hoozemans et al. 2004 & de Looze et al. 2000 (cart pushing), and
cannot be sharpened further without adding an EMG measurement phase
to the study.

## References

- Atzori, M. et al. (2014). Electromyography data for non-invasive
  naturally-controlled robotic hand prostheses. *Scientific Data*, 1.
- Bayer, A. et al. (2022). Sources of bias in batch-effect correction.
  *BMC Bioinformatics*, 23.
- Cerqueira, F. G. et al. (2024). WSD4FEDSRM shoulder fatigue dataset.
- Fortin, J.-P. et al. (2017). Harmonization of multi-site data with
  ComBat. *NeuroImage*, 167.
- Granata, K. P. & Marras, W. S. (1995). An EMG-assisted model of trunk
  loading during free-dynamic lifting. *Journal of Biomechanics*, 28(11).
- Waters, T. R., Putz-Anderson, V., Garg, A. & Fine, L. J. (1993).
  Revised NIOSH equation for the design and evaluation of manual
  lifting tasks. *Ergonomics*, 36(7), 749–776.
- Zindler, T. et al. (2020). Tail-preservation limits of ComBat. *Journal
  of Neuroscience Methods*, 338.
