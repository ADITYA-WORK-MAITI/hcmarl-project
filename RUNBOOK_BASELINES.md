# RUNBOOK_BASELINES — HC-MARL baselines clean-slate re-run (2026-04-21)

> **If you are Claude Code running inside the rented E2E VM terminal, this entire file is your briefing. Read every line top to bottom — do not skim, do not skip, do not summarise. The §0.1 boot prompt you were pasted tells you exactly when to act and when to stop. Follow it.**

---

## 0. What this runbook covers — and what it does NOT

> **!!! HCMARL IS FORBIDDEN IN THIS SESSION !!!**
>
> HCMARL headline training is **already done**. Its 10 seeds are archived on the laptop at `logs/vm_archive_2026_04_21/repo_state/logs/hcmarl/`. Re-running HCMARL here would cost ~Rs 900 of L4 time for **zero** new data, and would overwrite ablation evidence.
>
> `scripts/run_baselines.py` **DEFAULT BEHAVIOUR** reads all four methods from `config/experiment_matrix.yaml` — including `hcmarl`. On 2026-04-21 01:31 IST, exactly this footgun fired: the launcher was invoked without `--methods` and started training `hcmarl seed 0` first. Caught within 16 minutes and killed; no contamination, but Rs ~50 wasted and user trust shredded.
>
> **STEP 8 in this runbook ALWAYS passes `--methods mappo ippo mappo_lag`. NEVER omit that flag. NEVER add `hcmarl` to the list. If the dry-run banner in STEP 6 CHECK 6 prints "4 methods" or mentions "hcmarl" anywhere, STOP and escalate.**

**In scope (exclusively):**
- Three baselines: `mappo`, `ippo`, `mappo_lag`
- Ten seeds each: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9   → **30 total runs**
- Config files: `config/mappo_config.yaml`, `config/ippo_config.yaml`, `config/mappo_lag_config.yaml`
- Launcher: `scripts/run_baselines.py` with the new `--fresh-logs` flag AND `--methods mappo ippo mappo_lag` filter
- Logs target: `logs/{mappo,ippo,mappo_lag}/seed_{0..9}/training_log.csv` (30 CSVs)

**Out of scope (do NOT touch):**
- HCMARL headline runs — already done, already archived. Do not re-run, do not re-train, do not delete. Running HCMARL again is an automatic STOP condition.
- 500K probe, 1M watch, plateau checks — that was RUNBOOK.md. Not this session.
- Ablation grid (`scripts/run_ablations.py`). Not this session.
- Any edit to MMICRL, Path G, NSWF allocator, ECBF filter code. Not this session.

If in doubt, stop and ask. You are here to re-run three baselines cleanly. Nothing more.

---

## 0.1 Boot prompt (paste into VM-side Claude Code immediately after `claude` launches)

```
You are the VM-side Claude for the HC-MARL BASELINES-ONLY clean-slate re-run on
an E2E L4 GPU node (2026-04-21). This paste IS your full starting instruction —
no further instruction is coming for roughly the next 15-20 hours (30 runs
serial at ~1000 SPS, 2M steps per run, L4 on-demand). Execute continuously.

Your complete briefing is at /root/hcmarl_project/RUNBOOK_BASELINES.md on this
VM. Read every line of that file top to bottom. Do not skim, do not skip, do
not summarise.

!!! HCMARL IS FORBIDDEN !!!
HCMARL headline training is DONE and archived. Re-running it is a hard STOP
condition. `scripts/run_baselines.py` default behaviour includes hcmarl —
STEP 8 ALWAYS passes `--methods mappo ippo mappo_lag` to exclude it. Never
omit that flag. Never add hcmarl to it. If the dry-run banner at STEP 6
CHECK 6 shows "4 methods" or mentions "hcmarl", STOP IMMEDIATELY.

CRITICAL NON-NEGOTIABLE POINTS (if any of these is not true, STOP):
  1. You cloned the MOST RECENT push from
     github.com/ADITYA-WORK-MAITI/hcmarl-project (branch master). Confirm
     `git log -1 --format='%h %ci %s'` shows a commit dated on or after
     2026-04-21 whose message contains both "baseline" and "10 seeds".
  2. ECBF is OFF in ALL THREE baseline configs. Grep prove it:
     `grep -E 'enabled:' config/{mappo,ippo,mappo_lag}_config.yaml`
     — every line must show `enabled: false`.
  3. Every baseline config has a populated environment: section with
     muscle_groups, theta_max, AND tasks. Without this, safety_cost()
     silently returns 0 and the baseline trains in a constraint-free world.
     Grep prove it:
     `grep -l 'theta_max:' config/{mappo,ippo,mappo_lag}_config.yaml`
     — all three paths must print.
  4. logs/{mappo,ippo,mappo_lag}/ and checkpoints/{mappo,ippo,mappo_lag}/
     are EMPTY on the VM. Either never existed (fresh clone) or will be
     wiped by `--fresh-logs` before the launcher spawns its first
     subprocess. NEVER append new data to a contaminated CSV.
  5. `config/experiment_matrix.yaml` has `headline.seeds: [0, 1, 2, 3, 4,
     5, 6, 7, 8, 9]` (ten seeds). Grep prove it:
     `grep -A1 '^headline:' config/experiment_matrix.yaml | grep seeds`
     — output must contain ten integers.
  6. Dry-run banner verification (THIS IS THE HCMARL FOOTGUN GUARD):
     `python scripts/run_baselines.py --methods mappo ippo mappo_lag \
          --dry-run 2>&1 | head -5`
     FIRST LINE must be exactly:
     `Headline grid: 3 methods x 10 seeds = 30 runs`
     SECOND LINE must be exactly:
     `Methods: ['mappo', 'ippo', 'mappo_lag']`
     If either line deviates (says 4 methods, says 15 runs, lists hcmarl),
     STOP.

If ANY of the 6 checks above fails, STOP, post the failing check, wait.

After reading, execute STEPs 1-12 of RUNBOOK_BASELINES.md continuously, in
order, without asking for confirmation between steps:

  STEP 1   hardware sanity (nvidia-smi, lscpu, free, df)
  STEP 2   python3.12 venv at /root/hcmarl_project/venv
  STEP 3   pip install torch from cu124 (fall back to cu121)
  STEP 4   pip install -r requirements.txt
  STEP 5   pytest -q   (pass criterion: 0 failed; 3+ skips acceptable if
           they are env-conditional like no-CUDA or no-matplotlib)
  STEP 6   the SIX non-negotiable pre-flight checks above (all 6 must pass)
  STEP 7   create tmux session `baselines`
  STEP 8   launch scripts/run_baselines.py with --fresh-logs AND
           `--methods mappo ippo mappo_lag` (30 runs). The launch line is
           fixed — do not reword, do not drop flags, do not add --resume.
  STEP 9   automated status report every 20 minutes while the tmux is alive
           (see §5 format). Do NOT wait to be asked.
  STEP 10  when all 30 runs finish, post the final exit-summary table
           (§6 format).
  STEP 11  verify every one of logs/{mappo,ippo,mappo_lag}/seed_{0..9}/
           training_log.csv exists and is non-empty (§7 audit script).
  STEP 12  STOP. Do NOT destroy the node. Do NOT touch logs/hcmarl/. Wait
           for my next paste (§0.2).

Minor-blocker policy (§8): if a trivial issue blocks progress (missing pip
package that requirements.txt should have pulled in, typo in an import, a
logs/{method}/ directory that didn't get wiped, a stale .pyc), FIX IT IN
PLACE with minimal edits and REPORT what you changed. You may not change
hyperparameters, seed lists, total_steps, eval intervals, kill-switch
thresholds, ECBF state, environment parameters, anything under hcmarl/,
or the methods filter. If the fix exceeds those bounds, STOP and escalate.

Begin now. After reading RUNBOOK_BASELINES.md, reply with exactly:
`Read. Six pre-flight checks begin at STEP 6. HCMARL is forbidden.`
and start.
```

---

## 0.2 Close-out prompt (paste after all 30 CSVs are verified)

```
Baselines complete. Your work is done for this session.

The user will now scp the CSVs down to the laptop and destroy the E2E node
manually. Those are [USER] steps.

Do NOT destroy the node. Do NOT delete anything under logs/. Do NOT make
any git commits. Do NOT start any more training.

Final acknowledgement: reply with exactly
  `Baselines complete. Standing by. Local will close the session.`
and then wait.
```

---

## 1. Two-agent division of labour

| Agent | Owns | Forbidden |
|---|---|---|
| **LOCAL** (laptop this-morning session) | scope decisions, git state, RUNBOOK edits, archive of the prior HCMARL run, plateau interpretation from HCMARL data | direct SSH execution on VM |
| **VM** (Claude Code on the L4) | bootstrap, 6 pre-flight checks, tmux launch, monitoring, status reports, minor in-place fixes within §8 bounds, final CSV audit | scope decisions, destroying the node, editing tracked `.py` / config files beyond §8, `git commit` / `git push`, anything involving HCMARL data, running ablations, dropping the `--methods` filter |
| **USER** (Aditya) | E2E dashboard, SSH, `git clone` before claude launches, paste between sessions, scp pull, destroy node after audit passes | running training outside tmux, destroying before scp, editing files on VM directly (ask LOCAL) |

---

## 2. Project state at the start of this session

- Repo tip on origin/master has today's baseline-hardening commit. That commit adds:
  - **Fail-fast env-section guard** in `scripts/train.py`: the training script raises `ValueError` at startup if `config.environment.{muscle_groups, theta_max, tasks}` is missing or empty. This is the physical prevention against the 2026-04-20 contaminated-baseline bug recurring. You will never again be able to train a baseline in a constraint-free environment by accident.
  - **`--fresh-logs` flag** on `scripts/run_baselines.py`: when passed, `rm -rf logs/<method>/ checkpoints/<method>/` for every method in the grid BEFORE the first subprocess launches. This is the physical prevention against the 2026-04-20 "appended clean rows onto contaminated rows" bug. HCMARLLogger appends if the existing CSV header matches, so without this wipe, a re-run contaminates the file.
  - **`kp: 1.0` added** to the ecbf blocks of all three baseline configs. Irrelevant at runtime (ECBF stays off) but satisfies the `test_all_configs_have_kp` audit test, keeping the suite green.
  - **All three baseline configs already carry a populated `environment:` section** with muscle_groups, theta_max, tasks — committed before today but never re-verified on VM until STEP 6 below.
  - **ECBF disabled in all three baseline configs** via `ecbf.enabled: false`.

- Baseline logs/ dirs on the laptop are already empty. Baseline checkpoints/ dirs carry stale `.pt` files from the 2026-04-20 contaminated run — the VM's fresh clone will not have these, so no cleanup needed there for the VM.

- Test suite is **0 failed** on laptop (510 passed, 1 skipped). VM pytest on 2026-04-21 01:29 IST reported 509 passed / 3 skipped / 0 failed — the skip-count drifts by environment (CUDA visibility, matplotlib backend). Pass criterion at STEP 5 is **`0 failed`**, not an exact pass count.

- Budget reality with 10 seeds (THIS IS A KNOWN OVERSHOOT of the Rs 1,600 on-hand credit — user is aware and will top up before running):
  - 3 methods x 10 seeds x 2,000,000 steps x ~0.002s/step ≈ 30-45 hr serial
  - Rs ~57.82/hr (Rs 49 on-demand + 18% GST) x 40 hr ≈ **Rs 2,313 serial**
  - With `--max-parallel=3` (one concurrent seed per method, L4 has 23 GiB VRAM and per-run mem ≈ 300 MiB, so 3-way is safe):
    - wall-clock ≈ 10-15 hr, spend ≈ **Rs 580-870**
    - **This is the recommended mode for 10 seeds.** STEP 8 uses it.
  - Per-run `--budget-inr 1500` kill-switch is a belt-and-braces guard on individual runs (a single seed will never legitimately consume 1500 of wall-clock credit). It is NOT a total-budget cap.
  - **If the total E2E dashboard spend passes Rs 2,200 stop and escalate.**

---

## 3. Pre-flight on the laptop (already done by LOCAL — reported here so VM can verify)

LOCAL confirmed before pushing:
1. `config/{mappo,ippo,mappo_lag}_config.yaml` — `ecbf.enabled: false`, `environment.{muscle_groups,theta_max,tasks}` populated, `total_steps: 2000000`.
2. `scripts/train.py` — new fail-fast guard in `main()` after config load.
3. `scripts/run_baselines.py` — `--fresh-logs` flag wired, `shutil.rmtree` executes for every method before the subprocess loop.
4. `pytest -q` — 510 passed, 1 skipped, 0 failed.
5. `git push origin master` — VM's clone will pick this tip up.

---

## 4. Execution plan (VM owns STEPs 1-12)

### STEP 0 — [USER] boot the VM + clone the LATEST repo

These are done manually before Claude Code is launched. Follow RUNBOOK.md §1-7 for the SSH/clone mechanics, BUT the clone line must target the latest master:

```bash
cd /root
git clone https://github.com/ADITYA-WORK-MAITI/hcmarl-project.git hcmarl_project
cd hcmarl_project
git log -1 --format='%h %ci %s'
ls RUNBOOK_BASELINES.md && wc -l RUNBOOK_BASELINES.md
```

The `git log -1` output **must show a commit dated 2026-04-21 or later** mentioning baselines. If it doesn't, the clone is stale — re-run the clone.

Then launch Claude Code:
```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version
claude
```

Paste the **§0.1 boot prompt** verbatim.

### STEP 1 — [VM] Hardware sanity

```bash
nvidia-smi | head -25
lscpu | grep -E "Model name|Thread|Socket|CPU MHz|CPU max MHz"
free -h
df -h /root
```

Pass criteria: `NVIDIA L4`, VRAM free >= 23 GiB, RAM free >= 100 GiB, disk free >= 200 GiB. Report to user regardless.

### STEP 2 — [VM] Create the venv

```bash
cd /root/hcmarl_project
python3.12 -m venv venv
source venv/bin/activate
python --version   # expect 3.12.x
pip install -q -U pip wheel
```

### STEP 3 — [VM] Install torch from the CUDA wheel index FIRST

```bash
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"
```

Fallback to cu121 if cu124 reports `cuda available: False`. If both fail, STOP.

### STEP 4 — [VM] Install the rest

```bash
pip install -q -r requirements.txt
pip list | grep -iE "^(torch|gymnasium|numpy|cvxpy|osqp|scipy|pyyaml|matplotlib)"
```

Verify torch version contains `+cu124` (or `+cu121`). If pip swapped it for the CPU wheel, re-run STEP 3 with `--force-reinstall`.

### STEP 5 — [VM] Pytest sanity

```bash
pytest -q 2>&1 | tail -5
```

Pass criterion: **`0 failed`**. Skips are acceptable (typically 1-3 depending on CUDA / matplotlib / display-backend availability on the VM). One harmless CVXPY warning from `ecbf_filter.py:325` about "Solution may be inaccurate" is expected on `test_stress_2x_still_holds` and does not count as failure.

If any test fails, STOP. Paste the last 30 lines to the user.

### STEP 6 — [VM] The SIX non-negotiable pre-flight checks

This is the quality gate. Every one of these must pass. Run them as a single block and post the output to the user.

```bash
echo "=== CHECK 1: git tip ==="
git log -1 --format='%h %ci %s'
echo
echo "=== CHECK 2: ECBF off in all three baseline configs ==="
grep -H 'enabled:' config/mappo_config.yaml config/ippo_config.yaml config/mappo_lag_config.yaml
echo
echo "=== CHECK 3: environment section populated in all three ==="
for f in config/mappo_config.yaml config/ippo_config.yaml config/mappo_lag_config.yaml; do
  echo "--- $f ---"
  python -c "
import yaml, sys
c = yaml.safe_load(open('$f'))
env = c.get('environment', {}) or {}
for k in ('muscle_groups','theta_max','tasks'):
    n = len(env.get(k) or {})
    print(f'  {k}: {n} entries')
    assert n > 0, f'{k} is empty or missing in $f'
"
done
echo
echo "=== CHECK 4: logs and checkpoints clean for all three ==="
for m in mappo ippo mappo_lag; do
  for d in logs checkpoints; do
    p="$d/$m"
    if [ -d "$p" ] && [ -n "$(ls -A "$p" 2>/dev/null)" ]; then
      echo "  NON-EMPTY: $p/ — --fresh-logs will wipe"
    else
      echo "  clean: $p/"
    fi
  done
done
echo
echo "=== CHECK 5: experiment_matrix has ten seeds ==="
python - <<'PY'
import yaml
m = yaml.safe_load(open("config/experiment_matrix.yaml"))
seeds = m["headline"]["seeds"]
assert seeds == [0,1,2,3,4,5,6,7,8,9], f"expected 10 seeds [0..9], got {seeds}"
print(f"  headline.seeds = {seeds}  (count={len(seeds)})  OK")
PY
echo
echo "=== CHECK 6: dry-run banner (HCMARL footgun guard) ==="
python scripts/run_baselines.py --methods mappo ippo mappo_lag --dry-run 2>&1 | head -5
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    [sys.executable, "scripts/run_baselines.py",
     "--methods", "mappo", "ippo", "mappo_lag", "--dry-run"],
    text=True).splitlines()
want1 = "Headline grid: 3 methods x 10 seeds = 30 runs"
want2 = "Methods: ['mappo', 'ippo', 'mappo_lag']"
assert out[0] == want1, f"BANNER LINE 1 WRONG:\n  got:  {out[0]}\n  want: {want1}"
assert out[1] == want2, f"BANNER LINE 2 WRONG:\n  got:  {out[1]}\n  want: {want2}"
assert "hcmarl" not in " ".join(out).lower() or "hcmarl_project" in " ".join(out).lower(), \
    "banner mentions hcmarl — scope violation"
print("  dry-run banner matches spec  OK")
PY
echo
echo "All six checks complete."
```

Expected outcome on a fresh VM clone:
- CHECK 1 — commit hash with both "baseline" and "10 seeds" in the subject, dated 2026-04-21 or later.
- CHECK 2 — three lines each `enabled: false`.
- CHECK 3 — every muscle_groups / theta_max / tasks prints `6 entries` or similar, no assertion errors.
- CHECK 4 — six `clean:` lines (fresh clone has no `logs/` or `checkpoints/` subdirs at all). If any is NON-EMPTY, `--fresh-logs` in STEP 8 will handle it, but flag it in the status report.
- CHECK 5 — `headline.seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  (count=10)  OK`.
- CHECK 6 — banner prints `3 methods x 10 seeds = 30 runs` with `['mappo', 'ippo', 'mappo_lag']`, Python assertion prints `OK`.

**If any check fails, STOP. Post the full output. Do not proceed to STEP 7.**

### STEP 7 — [VM] Create tmux session

Idempotent — kills any stale `baselines` session from a prior aborted attempt before creating a fresh one.

```bash
tmux kill-session -t baselines 2>/dev/null || true
pkill -f "scripts/train.py" 2>/dev/null || true
pkill -f "run_baselines.py" 2>/dev/null || true
sleep 1
tmux new -d -s baselines
tmux send-keys -t baselines "cd /root/hcmarl_project && source venv/bin/activate" Enter
sleep 1
tmux list-sessions | grep baselines   # must print the baselines row
```

### STEP 8 — [VM] Launch the grid

This is the single command that runs 30 baseline seeds (3 methods x 10 seeds) with 3-way parallelism.

```bash
tmux send-keys -t baselines "python scripts/run_baselines.py --methods mappo ippo mappo_lag --device cuda --fresh-logs --max-parallel 3 --budget-inr 1500 --cost-per-hour 49.0 2>&1 | tee logs/baselines_run.log" Enter
```

Within 30 seconds of kickoff, `tmux capture-pane -t baselines -p | tail -20` must show the banner:
```
Headline grid: 3 methods x 10 seeds = 30 runs
Methods: ['mappo', 'ippo', 'mappo_lag']
Seeds:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
If the banner says **4 methods, 15 runs, or mentions hcmarl anywhere**: IMMEDIATELY `Ctrl-C` (send-keys `C-c`), `pkill -f run_baselines.py`, and STOP. Do not resume without LOCAL.

**Flags explained — do not improvise:**
- `--methods mappo ippo mappo_lag` — **NON-NEGOTIABLE HCMARL GUARD**. Without this, `run_baselines.py` pulls every method from `experiment_matrix.yaml` — which includes `hcmarl`. Re-running HCMARL here is a hard STOP condition (costs ~Rs 900 and overwrites nothing we need).
- `--device cuda` — L4 GPU required.
- `--fresh-logs` — NON-NEGOTIABLE. Wipes `logs/{mappo,ippo,mappo_lag}/` and `checkpoints/{mappo,ippo,mappo_lag}/` before launching. This is the physical guarantee against appending new good rows onto contaminated rows. Without this flag, STOP.
- `--max-parallel 3` — runs one concurrent seed per method (mappo + ippo + mappo_lag simultaneously). L4 has 23 GiB VRAM; per-run memory ≈ 300 MiB; 3-way is comfortably under the ceiling. The launcher auto-sets OMP/MKL/OPENBLAS thread caps at total_vcpus/3 so there is no CPU oversubscription. Cuts 30-seed wall-clock from ~40 hr (serial) to ~13 hr (3-way parallel). With 10 seeds per method this is the recommended mode; do not drop it without LOCAL approval.
- `--budget-inr 1500` — per-run kill-switch at Rs 1,500 * 0.95 = Rs 1,425 wall-clock spend. Per-run, not total — so the whole grid will never hit it in practice, but it's here as a belt-and-braces safety on individual runs.
- `--cost-per-hour 49.0` — E2E L4 on-demand.
- `tee logs/baselines_run.log` — mirror launcher stdout so the status report can read progress without `tmux capture-pane`.

Do NOT add `--resume` — this is a clean-slate grid.
Do NOT drop `--methods mappo ippo mappo_lag` — dropping it re-introduces the hcmarl footgun.
Do NOT alter the seed list or add `--seeds` — `run_baselines.py` reads the ten seeds from `config/experiment_matrix.yaml` (checked at STEP 6 CHECK 5).

### STEP 9 — [VM] Automated status reports (every 20 minutes)

**You do not wait for the user to ask.** Post the status block below every 20 minutes while the `baselines` tmux is alive. If nothing has changed in 20 minutes, post anyway — "no change" is still a status.

Status report format:

```
### Status report — <UTC HH:MM> (elapsed: <Hh:MMm> since STEP 8 kickoff)

Runs done / in-flight / pending:   <n_done>/<n_inflight>/<n_pending>  (of 30)
Current run:                       <method>_seed_<s>  (<current_global_step> / 2,000,000)
Current SPS (rolling 50 ep):       <sps>
ETA to grid completion (serial):   ~<Hh:MMm>
Wall-clock spend so far:           Rs ~<amount>  (assuming Rs 49/hr)
lazy_agent trips since start:      <count from grep>
budget_tripped events:             <count from grep>
pytest state:                      green (STEP 5) — no re-run needed
Last 5 lines of logs/baselines_run.log:
  <paste>
Notes / minor fixes applied:       <describe anything you fixed under §8, or "none">
```

Commands to compute the fields:

```bash
# n_done (CSV exists AND has >=2 non-header lines)
for m in mappo ippo mappo_lag; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    f=logs/$m/seed_$s/training_log.csv
    [ -f "$f" ] && [ "$(wc -l <"$f")" -ge 2 ] && echo "$m seed_$s"
  done
done | wc -l

# current step from the most recent CSV
ls -t logs/*/seed_*/training_log.csv 2>/dev/null | head -1 | xargs -I{} tail -1 {}
# (global_step is column 3 per logger.py CSV_COLUMNS sorted order — episode, global_step, ...)

# lazy_agent trips
grep -c "lazy-agent kill-switch" logs/baselines_run.log || true

# budget trips
grep -c "budget kill-switch" logs/baselines_run.log || true

# elapsed + spend
# LOCAL computed it in the status-report Python snippet below; VM can reuse.
python - <<'PY'
import os, time
t0 = os.path.getmtime("logs/baselines_run.log") if os.path.exists("logs/baselines_run.log") else time.time()
# Replace t0 with the kickoff epoch stored at STEP 8
PY
```

If you cannot derive a field, write `<unknown>` and explain. Never fabricate.

### STEP 10 — [VM] Final exit summary

When `scripts/run_baselines.py` prints `All 30 jobs complete.` (or fails), post this block:

```
### Grid done — <UTC HH:MM> (total wall-clock: <Hh:MMm>)

| Method    | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | best_reward range |
|-----------|----|----|----|----|----|----|----|----|----|----|-------------------|
| mappo     | D/F| .. | .. | .. | .. | .. | .. | .. | .. | .. | <min>..<max>      |
| ippo      | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | <min>..<max>      |
| mappo_lag | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | <min>..<max>      |
(cells: D=DONE, F=FAIL, L=lazy-trip, B=budget-trip)

Kill-switch events:
  lazy_agent trips: <count>  (list: <method>_seed_<s>, ...)
  budget trips:     <count>  (list: <method>_seed_<s>, ...)

Failures (run_baselines.py exit code != 0 subs):
  <list from the stdout FAILED lines, or "none">

Total spend: Rs ~<amount>   (per-run budget: Rs 1,500; user hard-cap: Rs 2,200)
```

Use this Python block to gather best_reward per seed:
```bash
python - <<'PY'
import json, os
for m in ("mappo","ippo","mappo_lag"):
    for s in range(10):
        p = f"logs/{m}/seed_{s}/summary.json"
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"{m} seed_{s}: best_reward={d.get('best_reward'):.1f} steps={d.get('total_steps')} trip={d.get('budget_tripped')}")
        else:
            print(f"{m} seed_{s}: MISSING summary.json")
PY
```

### STEP 11 — [VM] CSV audit (must pass before standing down)

Every one of the 30 CSVs must exist and have `global_step` columns reaching at least 1,000,000 (half of total_steps = 2M). Anything less indicates an aborted run that must be re-launched.

```bash
python - <<'PY'
import csv, os, sys
missing, short, ok = [], [], []
for m in ("mappo","ippo","mappo_lag"):
    for s in range(10):
        p = f"logs/{m}/seed_{s}/training_log.csv"
        if not os.path.exists(p):
            missing.append(p); continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            short.append((p, 0)); continue
        last_step = int(rows[-1].get("global_step") or 0)
        if last_step < 1_000_000:
            short.append((p, last_step))
        else:
            ok.append((p, last_step))
print(f"OK      ({len(ok)}):", *ok, sep="\n  ")
print(f"SHORT   ({len(short)}):", *short, sep="\n  ")
print(f"MISSING ({len(missing)}):", *missing, sep="\n  ")
if short or missing:
    sys.exit(1)
PY
echo "audit exit=$?"
```

Pass criterion: `audit exit=0`. If not, list the offending runs in a status report and wait for LOCAL to decide whether to re-launch them with specific `--seeds` override. Do not re-launch on your own.

### STEP 12 — [VM] Stand down

When STEP 11 passes, paste a single line:
```
Grid verified. Standing by for §0.2 close-out paste.
```

Then wait. Do not launch anything else.

---

## 5. Status-report cadence

Every **20 minutes** after STEP 8 kickoff. No exceptions. The user will sleep while the grid runs — if you go silent for 40 minutes they have no way to know whether the job is healthy, stalled, or has tripped. The cadence is the observability guarantee.

Additionally, post *immediately* (do not wait for the 20-minute tick) when:
- A run finishes (any of the 30).
- A `lazy-agent kill-switch` line appears in the log.
- A `budget kill-switch` line appears in the log.
- `run_baselines.py` reports a `FAILED (exit code X)` line.
- You encountered a minor blocker and applied a §8 fix.
- You encountered a non-minor blocker and stopped — in which case post the full context and then do not proceed.

---

## 6. Known failure modes and the root causes (context for VM Claude)

These are the specific failure modes from the 2026-04-20 run. Understanding why they happened is why STEP 6 exists.

### 6.1 Constraint-free training (the environment-section bug)

**Symptom:** baselines showed unbounded reward growth with zero violations and zero cost, learned a degenerate "100% heavy_lift" policy, and tripped the lazy-agent kill-switch at ~117K steps because per-agent entropy collapsed to zero on the single task.

**Root cause:** the baseline configs had no `environment:` block. At training time `env_cfg.get("theta_max", {}) or {}` returned `{}`, which is truthy-empty. `safety_cost()` (`hcmarl/envs/reward_functions.py:103`) iterates `fatigue_values` muscle keys and checks `if m in theta_max` — every check returns False, so every step's cost is 0. The policy has no incentive to rest, picks the highest-reward task forever, and entropy collapses.

**Prevention (committed today):** the fail-fast guard in `scripts/train.py` raises `ValueError` at startup if `environment.{muscle_groups, theta_max, tasks}` is missing or empty. The bug can no longer manifest silently — the process dies before any step is taken. Additionally all three baseline configs carry populated `environment:` blocks. STEP 6 CHECK 3 is the VM-side verification.

### 6.2 Contaminated CSV (the append-on-matching-header bug)

**Symptom:** after the environment-section bug was locally patched on 2026-04-20 and a clean run was launched, the resulting `training_log.csv` had the contaminated run's rows at the top and the clean run's rows appended below — because the old CSV still existed and its header matched.

**Root cause:** `HCMARLLogger` (`hcmarl/logger.py:53-68`) detects an existing CSV with a matching column header and opens it in append mode. This is correct for resume-after-crash but catastrophic for "wipe and start fresh" semantics.

**Prevention (committed today):** the `--fresh-logs` flag on `scripts/run_baselines.py` `rm -rf`'s `logs/{method}/` and `checkpoints/{method}/` for every method in the grid before spawning the first subprocess. STEP 6 CHECK 4 is the VM-side verification. STEP 8 uses the flag. Do not launch without it.

### 6.3 Lazy-agent kill-switch tripping

**Symptom:** training halts at ~100-117K steps with a `[lazy-agent kill-switch] ... — halting run.` message.

**Root cause:** 6.1 (constraint-free environment => degenerate single-task policy => entropy collapse). The kill-switch is working correctly — it's detecting a genuinely broken run. It is a diagnostic tool, not the disease.

**Prevention:** fix 6.1. After today's commit, with a populated `environment.theta_max`, the baselines have a legitimate multi-objective trade-off (reward vs. fatigue cost) and should not collapse to a single-task policy.

If the switch trips again under the corrected config, it's a real pathology — escalate. Do not disable the switch.

---

## 7. Files, paths, and layout (quick reference)

On the VM after successful clone + setup:
```
/root/hcmarl_project/
├── config/
│   ├── mappo_config.yaml         # environment: + ecbf.enabled=false + kp=1.0
│   ├── ippo_config.yaml          # same
│   ├── mappo_lag_config.yaml     # same + algorithm.cost_limit etc.
│   └── experiment_matrix.yaml    # seeds: [0..9] (ten); headline.methods (STEP 8 filters to 3 baselines)
├── scripts/
│   ├── train.py                  # fail-fast env-section guard (new today)
│   └── run_baselines.py          # --fresh-logs flag (new today)
├── hcmarl/
│   ├── envs/pettingzoo_wrapper.py
│   ├── envs/reward_functions.py  # safety_cost() lives here
│   └── logger.py                 # CSV append-on-matching-header
├── logs/                         # target: logs/{method}/seed_{s}/training_log.csv
└── checkpoints/                  # target: checkpoints/{method}/seed_{s}/checkpoint_final.pt
```

---

## 8. Minor-blocker authority (what VM Claude MAY fix in place)

User philosophy: "if LOCAL Claude can resolve it, so can VM Claude." But bounded. The whole point of this session is a clean, reproducible baseline grid — changes that alter the experiment itself are forbidden.

### MAY fix (report every one in a status update):
- A missing pip package that `requirements.txt` should have pulled but didn't (e.g. `pip install <pkg>`).
- A typo'd import in a **test** file that blocks pytest collection (not a source file).
- A stale `logs/<method>/` or `checkpoints/<method>/` subdir that `--fresh-logs` missed because it was outside the method list (it shouldn't be — STEP 6 CHECK 4 would have flagged it).
- A `.pyc` staleness issue (`find . -name __pycache__ -exec rm -rf {} +`).
- A CUDA-visibility issue fixed by `export CUDA_VISIBLE_DEVICES=0`.
- A `tmux` session that died — create a new one with the same name, re-launch from where it left off (but ONLY if the prior run's CSVs are intact and the launcher supports resume, which today it doesn't — if tmux dies mid-grid, STOP, do not blindly re-launch because `--fresh-logs` would wipe the partial progress).

### MUST NOT touch:
- Any file under `hcmarl/` (source code).
- `scripts/train.py`, `scripts/run_baselines.py`, `scripts/check_plateau.py`, `scripts/aggregate_learning_curves.py`.
- Any `config/*.yaml` (hyperparameters, seeds, total_steps, ECBF state, environment block, lazy-agent thresholds).
- `tests/*.py` beyond a trivial import-typo fix that blocks collection — and even then escalate afterward.
- `requirements.txt` — if torch pulls the wrong wheel, use `--force-reinstall`; do not pin-bump other libs.
- `.git`, `git add`, `git commit`, `git push`.
- Kill-switches (lazy-agent, budget).

### When in doubt:
STOP. Post the full context. Wait for LOCAL. The 20-minute cadence means the user will see the question within a sleep cycle — the penalty for pausing is low, the penalty for a bad in-place edit is a re-run.

---

## 9. Emergency procedures

| Event | Response |
|---|---|
| SSH drops mid-run | Training is in tmux; it keeps going. Reconnect, `tmux attach -t baselines`, resume status reports. |
| pytest fails STEP 5 | STOP. Paste the last 30 lines. Do not proceed. |
| Any STEP 6 check fails | STOP. Paste CHECK output. Do not proceed. LOCAL will push a fix and tell you to re-clone. |
| CUDA not available after STEP 3 | Retry with cu121. If still False, STOP — driver too old for the L4 image. |
| lazy-agent switch trips on any seed | Post immediately. Do NOT disable the switch. Let the grid complete — a single collapsed seed is data, not a blocker. LOCAL decides whether to re-seed. |
| budget kill-switch trips on any seed | Post immediately. That seed halts with a clean checkpoint; the grid continues on the next seed. LOCAL decides whether to re-launch. |
| `run_baselines.py` hangs with no log output > 20 min | `tmux capture-pane -t baselines -p | tail -200` and post. Do not kill. |
| Any CSV audit short/missing at STEP 11 | Post the output. Do not re-launch without LOCAL approval — re-launching with `--fresh-logs` would wipe the salvageable CSVs. |
| Rate-limit on Claude Code account | User has a standby Anthropic account. Relaunch `claude`, re-paste §0.1. VM Claude re-reads RUNBOOK_BASELINES.md and resumes by inspecting `tmux attach -t baselines`. |
| E2E billing appears out of line | User checks dashboard. STOP all status reports while they investigate. |

---

## 10. End-of-session checklist (user runs this after §0.2 paste)

- [ ] Final grid summary (§6 format) posted by VM
- [ ] CSV audit passed (STEP 11 `audit exit=0`)
- [ ] `scp -r -i ~/.ssh/id_ed25519 root@<IP>:/root/hcmarl_project/logs/{mappo,ippo,mappo_lag} /c/Users/admin/Desktop/hcmarl_project/logs/`
- [ ] `ls /c/Users/admin/Desktop/hcmarl_project/logs/mappo/seed_*/training_log.csv` returns 10 non-empty paths on the laptop; same for ippo, mappo_lag
- [ ] §0.2 close-out paste sent, VM acknowledged
- [ ] E2E node destroyed in dashboard
- [ ] E2E billing shows node charge stopped
- [ ] LOCAL appends session entry to `logs/project_log.md` and commits

Only after all eight boxes tick is this session "done."
