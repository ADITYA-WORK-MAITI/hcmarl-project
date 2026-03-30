# HC-MARL ICML Workshop Timeline Tracker
**Target: Submit by April 20, 2026**
**Last updated: 2026-03-30**

---

## TIMELINE (fixed)

### Phase A: Code Completion + MMICRL Validation (Mar 26–29)
| Day | Date | Task | Status |
|-----|------|------|--------|
| 1 | Mar 26 (Thu) | Fix MAPPO-Lag update(), IPPO update(), wire MMICRL into train.py | DONE |
| 2 | Mar 27 (Fri) | Fix SafePO wrapper. Fix mmicrl.py entropy bug. Dry-run all 6 methods. Per-worker theta_max in env. | DONE |
| 3 | Mar 28 (Sat) | Download PAMAP2. MMICRL + PAMAP2 real data validation. Longer dry runs all methods. | DONE (Mar 27) |
| 3+ | Mar 28 (Sat) | Path G: WSD4FEDSRM real-data calibration pipeline (additional scope). | DONE |
| 4 | Mar 29 (Sun) | End-to-end dry run all methods 50K steps. Freeze codebase. Set up Colab/Kaggle. | PARTIAL — quality audit done, 50K dry run + Colab still TODO |

### Additional work (Mar 29–30, not in original timeline)
| Date | Task | Status |
|------|------|--------|
| Mar 29 | Quality audit of Path G + 6 limitations assessment | DONE |
| Mar 30 | Big Fry 5: Reward unification (3 conflicting → 1 canonical, matches math doc Eq 32) | DONE |
| Mar 30 | Big Fry 6: MAPPO per-agent buffer fix + ECBF alpha correction (0.5→0.05) | DONE |
| Mar 30 | Big Fry 1: MMICRL rewrite — per-step (s,a) features + ConstraintNetwork + trajectory-level Bayesian assignment | DONE |
| Mar 30 | 11 bug fixes across training infra (wrong obs, flat MAPPO-Lag buffer, tautological checkpoint, etc.) | DONE |
| Mar 30 | L9: ECBF intervention tracking + Safety Autonomy Index metric + ecbf_mode ablation | DONE |
| Mar 30 | L12: CFDE validation — 3 bugs fixed (log-scale explosion, momentum=0, posterior mode collapse) + 5 math property tests | DONE |
| Mar 30 | L1: Grip r=15→30 (matches math doc Table 1) | DONE |
| Mar 30 | L4: Dynamic vs isometric F regime reconciliation + cross-validation + 3 tests | DONE |
| Mar 30 | L5: Inter-muscle correlated sampling (rho=0.5 conditioned on shoulder F) + 2 tests | DONE |
| Mar 30 | Pytest tests for real_data_calibration.py (53 tests) | DONE |

### Phase B: Training Experiments (Mar 30 – Apr 8)
| Day | Date | Task | Status |
|-----|------|------|--------|
| 5 | Mar 30 (Mon) | Set up Colab Pro + Kaggle Pro. Launch HC-MARL (5 seeds x 5M steps). | TODO |
| 6 | Mar 31 (Tue) | Finish HC-MARL seeds. Launch MAPPO baseline (5 seeds). | TODO |
| 7 | Apr 1 (Wed) | Launch IPPO (5 seeds) + MAPPO-Lag (5 seeds). | TODO |
| 8 | Apr 2 (Thu) | Launch ablations: no_ecbf, no_nswf, no_mmicrl (15 runs). | TODO |
| 9 | Apr 3 (Fri) | Launch ablations: no_reperfusion, no_divergent (10 runs). | TODO |
| 10 | Apr 4 (Sat) | Launch scaling: N=3,4,6,8,12 (25 runs). | TODO |
| 11 | Apr 5 (Sun) | Safety-Gym validation (optional, if OmniSafe installs). | TODO |
| 12 | Apr 6 (Mon) | Buffer day for reruns / failed seeds. Download all results. | TODO |
| 13 | Apr 7 (Tue) | Run evaluate.py on all checkpoints. Generate raw CSV tables. | TODO |
| 14 | Apr 8 (Wed) | Build plot_results.py. Generate all paper figures from real data. | TODO |

### Phase C: Paper Writing (Apr 9–18)
| Day | Date | Task | Status |
|-----|------|------|--------|
| 15 | Apr 9 (Thu) | Paper skeleton + Method section from math doc. | TODO |
| 16 | Apr 10 (Fri) | Introduction + Related Work. | TODO |
| 17 | Apr 11 (Sat) | Experiments section: setup, baselines, metrics. Insert figures. | TODO |
| 18 | Apr 12 (Sun) | Results + Discussion. Interpret every figure. | TODO |
| 19 | Apr 13 (Mon) | Conclusion + Limitations + Broader Impact. Finalize abstract. | TODO |
| 20 | Apr 14 (Tue) | First complete draft. Self-review pass. | TODO |
| 21 | Apr 15 (Wed) | Send to Dr. Amrit Pal Singh for review. | TODO |
| 22 | Apr 16 (Thu) | Revise based on advisor feedback. Polish figures. | TODO |

### Phase D: Final Polish + Submit (Apr 17–20)
| Day | Date | Task | Status |
|-----|------|------|--------|
| 23 | Apr 17 (Fri) | Final proofread. ICML workshop template check. BibTeX. | TODO |
| 24 | Apr 18 (Sat) | Buffer. | TODO |
| 25 | Apr 19 (Sun) | Buffer. Final read-through. | TODO |
| 26 | Apr 20 (Mon) | **SUBMIT.** | TODO |

---

## WHAT IS DONE

| Component | File(s) | Evidence |
|-----------|---------|----------|
| 3CC-r fatigue ODE | `hcmarl/three_cc_r.py` | Verified vs Eqs 2-6, Table 1. RK45 + Euler. Grip r=30 (L1 fix). |
| ECBF safety filter | `hcmarl/ecbf_filter.py` (429 lines) | Dual-barrier CBF-QP, CVXPY/OSQP. Zero deviations from Eqs 12-23. Alpha corrected to 0.05/0.05/0.1. |
| ECBF intervention tracking | `hcmarl/envs/pettingzoo_wrapper.py`, `warehouse_env.py` | Per-step intervention count + clip magnitude. SAI metric. ecbf_mode on/off. |
| NSWF allocator | `hcmarl/nswf_allocator.py` | Correct Eqs 31-33, Def 6.1. Dead code removed. |
| Reward functions | `hcmarl/envs/reward_functions.py` | Unified canonical: ln(max(U-D, eps)) - lambda*violations. D = kappa*MF^2/(1-MF). Max(MF) aggregation. |
| End-to-end pipeline | `hcmarl/pipeline.py` | All 7 steps of Section 7.3. |
| Warehouse env (single) | `hcmarl/warehouse_env.py` | Gymnasium single-agent env. Unified reward. ECBF tracking. |
| Warehouse env (multi) | `hcmarl/envs/pettingzoo_wrapper.py` | N-agent parallel env, 3CC-r + ECBF + per-worker theta_max. |
| MAPPO agent | `hcmarl/agents/mappo.py` | Per-agent buffer, per-agent GAE, shared actor PPO update. |
| MAPPO-Lagrangian | `hcmarl/agents/mappo_lag.py` | Per-agent LagrangianRolloutBuffer + cost GAE + normalized cost advantages. |
| IPPO | `hcmarl/agents/ippo.py` | Per-agent buffers + store_step() + independent PPO updates. |
| HCMARLAgent wrapper | `hcmarl/agents/hcmarl_agent.py` | Thin wrapper around MAPPO + ECBF/NSWF params. |
| Neural networks | `hcmarl/agents/networks.py` (72 lines) | Actor, Critic, CostCritic. Orthogonal init. |
| MMICRL core + CFDE | `hcmarl/mmicrl.py` | Real CFDE (MADE normalizing flow), per-step (s,a) training, trajectory-level Bayesian E-step, ConstraintNetwork per type, MI from soft posteriors. Log-scale clamped, momentum=0.1, delayed E-step. |
| Data loaders | `hcmarl/data/loaders.py` (377 lines) | RoboMimic, D4RL, PAMAP2 loaders. Real code. |
| Real-data calibration | `hcmarl/real_data_calibration.py` | Path G: WSD4FEDSRM 34 subjects, 1D F calibration, correlated FR sampling, dynamic/isometric reconciliation. 53 pytest tests. |
| Training script | `scripts/train.py` | All agent types, correct obs storage, threshold checkpointing, lambda EMA, proportional type assignment, SAI metric. |
| Evaluation script | `scripts/evaluate.py` (161 lines) | Load checkpoint, run N episodes, compute metrics. |
| Batch launchers | `scripts/run_baselines.py`, `run_ablations.py`, `run_scaling.py`, `run_safety_gym.py` | All use subprocess to call train.py. |
| OmniSafe wrapper | `hcmarl/baselines/omnisafe_wrapper.py` | PPO-Lag, CPO. Graceful fallback. Has train() and load(). |
| SafePO/MACPO wrapper | `hcmarl/baselines/safepo_wrapper.py` | Falls back to MAPPOLag. Has update()/buffer/save/load. |
| 10 baseline methods | `hcmarl/baselines.py` | Fixed-schedule, greedy-safe, etc. All functional. |
| Logger | `hcmarl/logger.py` | CSV + optional W&B. |
| 25 YAML configs | `config/*.yaml` | All methods, ablations, scaling, safety-gym. |
| 7 Colab notebooks | `notebooks/train_*.ipynb` | Training notebooks for each experiment group. |
| PAMAP2 real data | `data/pamap2/` (38856 samples, 9 subjects) | Downloaded from HuggingFace mirror. |
| WSD4FEDSRM real data | `data/wsd4fedsrm/WSD4FEDSRM/` (392.7 MB, 34 subjects) | Zenodo 8415066. EMG+IMU+PPG+Borg+MVIC. |
| MMICRL PAMAP2 validation | `scripts/validate_mmicrl_pamap2.py` | 3 types, MI=0.773, distinct thresholds. ALL CHECKS PASSED. |
| MMICRL WSD4FEDSRM validation | `scripts/validate_mmicrl_real_data.py` | 14+ validations: calibration, demos, types, correlation, Borg RPE, dynamic/isometric, correlated sampling. |
| CFDE validation | `scripts/validate_cfde.py` | 5 math property tests: invertibility, log-det, masking, density normalization, type recovery. |
| Extended dry run | `scripts/extended_dry_run.py` | 6 methods x 20 eps x 60 steps. Zero NaN/Inf. ALL PASSED. |
| Math modelling doc | `MATHEMATICAL MODELLING.pdf` (15 pages) | Complete. Not to be touched. |
| Architecture diagram | `figures/hcmarl_architecture_diagram.png` | Generated. |
| Demo notebook | `notebooks/demo_evaluation.ipynb` | Produces fig1-fig6 from pipeline simulation. |
| Tests | `tests/` (238 tests) | All passing. |

## WHAT IS LEFT

| # | Task | Effort | Blocked by | Status |
|---|------|--------|------------|--------|
| 1 | ~~Fix mmicrl.py line 391 entropy bug~~ | 30 min | — | DONE (Mar 27) |
| 2 | ~~Fix SafePO wrapper~~ | 1 hr | — | DONE (Mar 27) |
| 3 | ~~Dry-run all 6 methods~~ | 2 hrs | — | DONE (Mar 27) |
| 3b | ~~Per-worker theta_max in env~~ | 1 hr | — | DONE (Mar 27) |
| 3c | ~~Full MMICRL→training pipeline test~~ | 1 hr | — | DONE (Mar 27) |
| 4 | ~~Download PAMAP2 dataset~~ | 30 min | — | DONE (Mar 27) |
| 5 | ~~Validate MMICRL on PAMAP2 end-to-end~~ | 2 hrs | — | DONE (Mar 27) |
| 5b | ~~Path G: WSD4FEDSRM real-data pipeline~~ | 4 hrs | — | DONE (Mar 28) |
| 5c | ~~Quality audit + bug fixes (Big Fry 1/5/6, 11 bugs, L1/L4/L5/L9/L12)~~ | 8 hrs | — | DONE (Mar 29-30) |
| 5d | ~~Pytest tests for real_data_calibration.py~~ (53 tests) | 1 hr | — | DONE (Mar 30) |
| 6 | **Set up Colab Pro + Kaggle Pro** — upload repo, verify GPU training | 2 hrs | #5d | TODO |
| 7 | **Run ALL training experiments** — ~95 runs across 7 groups | 10 days | #6 | TODO |
| 8 | **Build scripts/plot_results.py** — learning curves, bar charts, ablation plots | 1 day | #7 | TODO |
| 9 | **Write the paper** — 4-6 pages ICML workshop format | 7-10 days | #8 | TODO |
| 10 | **Advisor review + revise** | 2-3 days | #9 | TODO |

### Stats
- **Tests:** 238 passed, 0 failed
- **Training runs completed:** 0 / ~95
- **Figures from real data:** 0
- **Paper pages written:** 0 / 4-6
- **Checkpoints saved:** 0
