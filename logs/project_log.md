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

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->
