# HC-MARL Project Log

> Format: Each entry carries a date, a time (IST, UTC+5:30), a plain-language summary, and a complete list of terminal commands run in that block — in the exact order they were executed.

---

## 2026-03-04 | ~17:00 IST — Environment Setup & First Commit

The project already existed on disk at `C:\Users\admin\Desktop\hcmarl_project` with 17 source files spread across three folders (`hcmarl/`, `tests/`, `config/`) and four root files (`.gitignore`, `README.md`, `requirements.txt`, `setup.py`). The first task was to verify the Python version, build an isolated virtual environment, install all project dependencies, and confirm that the full test suite ran clean.

Python 3.13.3 was confirmed — well above the required 3.9 minimum. A virtual environment was created inside the project folder under the name `venv`. The project was then installed in editable mode using `pip install -e ".[dev]"`, which pulled in numpy 2.4.2, scipy 1.17.1, cvxpy 1.8.1, osqp 1.1.1, pyyaml 6.0.3, pytest 9.0.2, and pytest-cov 7.0.0. All 120 pytest tests passed with zero failures in 19.37 seconds. Eight non-fatal solver warnings were noted from `ecbf_filter.py:309` — these are CVXPY convergence hints and do not affect correctness.

With the code verified, Git was initialised. During staging it was noticed that `.claude/settings.local.json` (a local Claude Code IDE settings file) was about to be committed. The `.gitignore` was updated to exclude the `.claude/` directory entirely, that file was unstaged, and a clean commit of all 17 project files was made.

**Commands executed (in order):**
```
python --version
python -m venv venv
source venv/Scripts/activate
pip install -e ".[dev]"
pytest tests/ -v
git init
git add .
git status
# → noticed .claude/settings.local.json was staged — excluded it
# → added .claude/ to .gitignore IDE section
git rm --cached .claude/settings.local.json
git add .gitignore
git status
git commit -m "Phase 1: core modules - 3CC-r fatigue model, ECBF safety filter, NSWF allocator, pipeline (17 files, 120 tests passing)"
```

**Commit:** `8759442` — 2026-03-04 17:15:55 IST
**Files committed:** 17 | **Insertions:** 3363 | **Tests:** 120 passed, 0 failed

---

## 2026-03-04 | ~17:30 IST — Notebook Integration & Second Commit

The notebook `demo_evaluation.ipynb` was already present inside the `notebooks/` folder. The task was to install visualisation dependencies into the existing virtual environment, register it as a named Jupyter kernel, execute the notebook end-to-end from the command line to verify every cell ran without error, confirm the expected six figure files were produced, and commit the clean (unexecuted) notebook source to Git.

matplotlib 3.10.8, jupyter 1.1.1, and ipykernel 7.2.0 were installed into the venv alongside all their dependencies. The venv was then registered as a Jupyter kernel under the name `hcmarl` with display name `HC-MARL`. The notebook was executed non-interactively using `jupyter nbconvert`, producing a 705,184-byte executed output file. All cells completed without error. A ProactorEventLoop warning from pyzmq appeared on Windows — this is a known harmless Windows-only asyncio compatibility notice and does not affect execution. Six PNG figures were confirmed in `notebooks/`: `fig1_3ccr_fatigue_dynamics.png`, `fig2_fatigue_resistance_ranking.png`, `fig3_ecbf_safety_filter.png`, `fig4_nswf_allocation.png`, `fig5_full_pipeline_simulation.png`, and `fig6_allocation_timeline.png`. Only the clean source notebook was committed; the executed copy and PNGs are covered by `.gitignore`.

**Commands executed (in order):**
```
source venv/Scripts/activate
pip install matplotlib jupyter ipykernel
python -m ipykernel install --user --name=hcmarl --display-name="HC-MARL"
jupyter nbconvert --to notebook --execute notebooks/demo_evaluation.ipynb \
    --output demo_evaluation_executed.ipynb \
    --ExecutePreprocessor.timeout=120 \
    --ExecutePreprocessor.kernel_name=hcmarl
# → confirmed 6 PNG files generated in notebooks/
git add notebooks/demo_evaluation.ipynb
git commit -m "Add evaluation demo notebook (6 figures, 60-min pipeline simulation)"
```

**Commit:** `0508238` — 2026-03-04 17:41:21 IST
**Files committed:** 1 (`notebooks/demo_evaluation.ipynb`) | **Insertions:** 628

---

## 2026-03-04 | ~17:50 IST — Project Log Initialised

A `logs/` directory was created inside the project root. This file (`project_log.md`) was written to serve as the persistent, timestamped record of all work done on the HC-MARL project. Every future session will append a new dated entry here. Entries will include a plain-language summary and a full chronological list of terminal commands run. The log itself was then staged and committed.

**Commands executed (in order):**
```
mkdir -p logs
# → wrote logs/project_log.md
git add logs/project_log.md
git commit -m "Add project log (logs/project_log.md) — session records from 2026-03-04"
```

---

## 2026-03-15 | ~00:00 IST — Phase 2+3 Integration

Integrated Phase 2 (warehouse environment, MAPPO agent, baselines, training pipeline) and Phase 3 (MM-ICRL inverse constrained RL, online adaptation, simulated safety-gym validation) into the project. Six source modules were copied from `C:\Users\admin\Desktop\hcmarl_phase2_phase3\src\` into `hcmarl/`, two test files into `tests/`, and one config file into `config/`. All `from src.` imports were replaced with `from hcmarl.` across the 3 affected files (training.py, test_phase2.py, test_phase3.py). The `gymnasium` package was installed into the venv. The full test suite ran clean: 155 passed (120 Phase 1 + 17 Phase 2 + 18 Phase 3), 0 failed, 8 warnings (same known CVXPY solver warnings as before). Nine files were committed.

**Commands executed (in order):**
```
cp hcmarl_phase2_phase3/src/{warehouse_env,mappo_agent,baselines,training,mmicrl,safety_gym_validation}.py hcmarl/
cp hcmarl_phase2_phase3/tests/{test_phase2,test_phase3}.py tests/
cp hcmarl_phase2_phase3/config/training.yaml config/
sed -i 's/from src\./from hcmarl./g' hcmarl/training.py tests/test_phase2.py tests/test_phase3.py
source venv/Scripts/activate && pip install gymnasium
pytest tests/ -v
git add config/training.yaml hcmarl/{baselines,mappo_agent,mmicrl,safety_gym_validation,training,warehouse_env}.py tests/{test_phase2,test_phase3}.py
git commit -m "Phase 2+3: warehouse env, MAPPO agent, baselines, MM-ICRL, safety validation (9 files, 155 tests passing)"
```

**Commit:** `9d65d32` — 2026-03-15 IST
**Files committed:** 9 | **Insertions:** 2791 | **Tests:** 155 passed, 0 failed

---

## 2026-03-15 | ~01:00 IST — Phase 2+3 Complete Integration (envs, agents, baselines)

Integrated the remaining Phase 2 and Phase 3 modules from `C:\Users\admin\Desktop\phase23\`. Created three new subpackages: `hcmarl/envs/` (PettingZoo parallel wrapper, task profiles, reward functions, safety-gym ECBF wrapper), `hcmarl/agents/` (MAPPO, IPPO, MAPPO-Lagrangian, actor-critic networks, full HC-MARL agent), and `hcmarl/baselines/` (OmniSafe and SafePO wrappers). Also added `hcmarl/logger.py` (W&B + CSV logging with 9 metrics), 8 config YAML files, 6 new test files, 4 scripts, and 1 docs file.

Two compatibility issues were resolved. First, the new `hcmarl/baselines/` package directory shadowed the existing `hcmarl/baselines.py` module — solved by copying the old module into the package as `_legacy.py` and re-exporting all its symbols from `__init__.py`. Second, `hcmarl/envs/warehouse_env.py` didn't exist in the source — created a thin wrapper that re-exports `SingleWorkerWarehouseEnv` as `WarehouseEnv`. Third, `pettingzoo_wrapper.py` called `MuscleParams.get_default()` which doesn't exist — fixed to use `get_muscle()` from `hcmarl.three_cc_r`. PyTorch CPU was installed. All 175 tests passed (120 Phase 1 + 35 earlier Phase 2/3 + 20 new).

**Commands executed (in order):**
```
mkdir -p hcmarl/envs hcmarl/agents hcmarl/baselines scripts docs
cp phase23/hcmarl/envs/*.py hcmarl/envs/
cp phase23/hcmarl/agents/*.py hcmarl/agents/
cp phase23/hcmarl/baselines/*.py hcmarl/baselines/
cp phase23/hcmarl/logger.py hcmarl/
cp phase23/config/*.yaml config/
cp phase23/tests/*.py tests/
cp phase23/scripts/* scripts/
cp phase23/docs/* docs/
# created hcmarl/envs/warehouse_env.py (thin wrapper)
# copied hcmarl/baselines.py → hcmarl/baselines/_legacy.py
# updated hcmarl/baselines/__init__.py to re-export legacy symbols
# fixed pettingzoo_wrapper.py: MuscleParams.get_default() → get_muscle()
pip install torch --index-url https://download.pytorch.org/whl/cpu
pytest tests/ -v
git add <36 files>
git commit
```

**Commit:** `5f2f078` — 2026-03-15 IST
**Files committed:** 36 | **Insertions:** 1627 | **Tests:** 175 passed, 0 failed

---

## 2026-03-16 | ~00:00 IST — Phase 2+3 Full Verification (3 tasks)

Ran the three remaining Phase 2+3 verification tasks at full specification, no shortcuts. Task 1: the 10K-episode stress test (`scripts/stress_test_env.py`) completed all 10,000 episodes in 1253.4 seconds with peak memory stable at 21.6 MB throughout — no crashes, no memory leak. Task 2: created `notebooks/env_smoke_test.ipynb` programmatically using nbformat with 5 code cells. The notebook runs 1 episode (60 steps, 4 workers, random actions) on the PettingZoo wrapper, verifies the 3CC-r conservation law MR+MA+MF=1 at every step for all workers and muscles (max error 1.19e-07, floating-point level), and produces two figures: `smoke_fig1_shoulder_dynamics.png` (MR/MA/MF trajectories for shoulder) and `smoke_fig2_fatigue_heatmap.png` (fatigue heatmap across all 6 muscles). The notebook was executed via `jupyter nbconvert` with the `hcmarl` kernel. Task 3: `scripts/verify_all_methods.py` ran all 10 methods (HC-MARL, MAPPO, IPPO, MAPPO-Lag, PPO-Lag, CPO, MACPO, FOCOPS, Random, FixedSchedule) for 100 steps each — all passed with no crashes. Updated `docs/milestone_1_report.md` with full verification results.

**Commands executed (in order):**
```
source venv/Scripts/activate && python scripts/stress_test_env.py          # Task 1 (background)
source venv/Scripts/activate && python scripts/verify_all_methods.py       # Task 3 (background)
python scripts/create_env_smoke_test.py                                    # Task 2: create notebook
jupyter nbconvert --to notebook --execute notebooks/env_smoke_test.ipynb \
    --output env_smoke_test_executed.ipynb \
    --ExecutePreprocessor.timeout=120 --ExecutePreprocessor.kernel_name=hcmarl
```

**Files changed:** 3 (`docs/milestone_1_report.md`, `notebooks/env_smoke_test.ipynb`, `scripts/create_env_smoke_test.py`)
**Tests:** 175 passed, 0 failed (unchanged)

---

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->
