# RUNBOOK_EXP1 — HC-MARL + 3 baselines full-grid headline run (2026-04-24)

> **If you are Claude Code running inside the rented E2E VM terminal, this
> entire file is your briefing. Read every line top to bottom — do not skim,
> do not skip, do not summarise. The §0.1 boot prompt tells you exactly when
> to act and when to stop. Follow it.**

This is **Experiment 1** in the user's numbered experiment plan. It supersedes
`RUNBOOK_BASELINES.md`: that runbook covered baselines-only on L4. This one
covers HCMARL + 3 baselines on **L4 + 50 vCPU**, with parameter-shared IPPO
and results delivered to the top-level `Results 1/` folder.

---

## 0. What this runbook covers — and what it does NOT

**In scope:**
- Four methods: `hcmarl`, `mappo`, `ippo`, `mappo_lag`
- Ten seeds each: 0..9 → **40 total runs**
- Config files: `config/{hcmarl_full,mappo,ippo,mappo_lag}_config.yaml`
- Launcher: `scripts/run_baselines.py --methods hcmarl mappo ippo mappo_lag`
  (same launcher; we simply include hcmarl in the filter this time)
- Hardware: **L4 GPU (48 GB VRAM), 50 vCPUs, Rs 98/hr**
- `--max-parallel 6` (6 seeds concurrent; rationale in §4 STEP 8)
- Logs target: `logs/{hcmarl,mappo,ippo,mappo_lag}/seed_{0..9}/training_log.csv`
- **Final deliverable**: the top-level `Results 1/` folder containing every
  CSV, summary JSON, MMICRL result, checkpoint metadata, and aggregation
  artifact, ready for one `scp -r` pull to the laptop (§11).

**Out of scope:**
- Ablation grid (`scripts/run_ablations.py`) — that is Experiment 2.
- Path G real-data MMICRL evaluation — that is Experiment 3.
- Synthetic K=3 sanity validation — that is Experiment 3.
- Visualization / analysis / interpretation — that is **Experiment 4
  (LOCAL only)**. The VM only *produces* the numerical artefacts; analysis
  is done on the laptop.
- Any edit to `hcmarl/` source files beyond §8 minor fixes.

---

## 0.1 Boot prompt (paste into VM-side Claude Code immediately after `claude` launches)

```
You are the VM-side Claude for the HC-MARL Experiment 1 headline run on an
E2E L4 GPU node (2026-04-25). This paste IS your full starting instruction.

Your complete briefing is at /root/hcmarl_project/RUNBOOK_EXP1.md on this
VM. Read every line of that file top to bottom. Do not skim, do not skip,
do not summarise.

CRITICAL NON-NEGOTIABLE POINTS (if any of these is not true, STOP):
  1. You cloned the MOST RECENT push from
     github.com/ADITYA-WORK-MAITI/hcmarl-project (branch master).
  2. ECBF is OFF in all three baseline configs. Prove it:
     `grep -E 'enabled:' config/{mappo,ippo,mappo_lag}_config.yaml`
     — every line must show `enabled: false`.
  3. ECBF is ON in hcmarl_full_config.yaml. Prove it:
     `grep -E 'enabled:' config/hcmarl_full_config.yaml`
     — must show `enabled: true`.
  4. Every baseline config has a populated environment: section with
     muscle_groups, theta_max, AND tasks. Without this, safety_cost()
     silently returns 0 and the baseline trains in a constraint-free
     world. That happened on 2026-04-20 and ruined the prior run.
  5. logs/{hcmarl,mappo,ippo,mappo_lag}/ and checkpoints/{same}/ are
     EMPTY on the VM. Either never existed (fresh clone) or will be
     wiped by `--fresh-logs` before the launcher spawns its first
     subprocess. NEVER append new data to a contaminated CSV.
  6. `config/experiment_matrix.yaml` has ten seeds [0..9].
  7. IPPO is the parameter-shared variant (single shared actor + single
     shared critic on local obs). Verify:
     `grep -c "parameter-shared\|PS-IPPO\|Yu et al. 2022" hcmarl/agents/ippo.py`
     must print >= 1.
  8. Dry-run banner verification:
     `python scripts/run_baselines.py --methods hcmarl mappo ippo mappo_lag \
          --dry-run 2>&1 | head -5`
     FIRST LINE must be exactly:
     `Headline grid: 4 methods x 10 seeds = 40 runs`
     SECOND LINE must be exactly:
     `Methods: ['hcmarl', 'mappo', 'ippo', 'mappo_lag']`
     If either line deviates, STOP.

If ANY of the 8 checks above fails, STOP, post the failing check, wait.

After reading, execute STEPs 1-13 of RUNBOOK_EXP1.md continuously, in
order, without asking for confirmation between steps:

  STEP 1   hardware sanity (expect L4 or better, >=24 GB VRAM, >=48 vCPUs)
  STEP 2   python3.12 venv
  STEP 3   torch from cu124 (fall back to cu121)
  STEP 4   pip install -r requirements.txt
  STEP 5   pytest -q  (pass: 0 failed)
  STEP 6   the EIGHT pre-flight checks above
  STEP 7   create tmux session `exp1`
  STEP 8   launch: 4 methods x 10 seeds, --max-parallel 6, --fresh-logs
  STEP 9   status report every 20 minutes (format in §5)
  STEP 10  when grid finishes, post the exit summary (§6)
  STEP 11  CSV audit must pass before standing down
  STEP 12  copy everything into `Results 1/` at repo root (§10)
  STEP 13  STOP. Wait for the §0.2 close-out paste.

Minor-blocker policy (§8): fix trivial blockers in place (missing pip pkg,
stale .pyc, tmux recovery) and report. Do NOT modify hyperparameters,
seeds, total_steps, ECBF states, env parameters, or anything under
hcmarl/ source. If the fix exceeds that, STOP and escalate.

Begin now. After reading RUNBOOK_EXP1.md, reply with exactly:
`Read. Eight pre-flight checks begin at STEP 6. Results land in 'Results 1/'.`
and start.
```

---

## 0.2 Close-out prompt (paste after the CSV audit passes and Results 1/ is populated)

```
Experiment 1 complete. Your work is done for this session.

The user will now scp Results 1/ to the laptop and destroy the E2E node
manually. Those are [USER] steps.

Do NOT destroy the node. Do NOT delete anything. Do NOT git commit. Do
NOT start more training.

Reply with exactly
  `EXP1 complete. Standing by. Local will close the session.`
and then wait.
```

---

## 1. Two-agent division of labour

| Agent | Owns | Forbidden |
|---|---|---|
| **LOCAL** (laptop) | scope decisions, git state, RUNBOOK edits, scp pull, node destruction | direct SSH execution on VM |
| **VM** (Claude Code on L4) | bootstrap, 8 pre-flight checks, tmux launch, 20-min status reports, minor §8 fixes, CSV audit, `Results 1/` assembly | scope decisions, editing anything under hcmarl/ source, git commits, destroying the node, running anything after STEP 13 |
| **USER** | E2E dashboard, SSH, git clone before claude launches, paste between sessions, scp pull `Results 1/`, destroy node | running training outside tmux |

---

## 2. Project state at the start of this session

- Repo tip on origin/master carries the EXP1 delta:
  - **IPPO rewritten as parameter-shared** (PS-IPPO, Yu et al. 2022 MAPPO
    benchmark variant). Single shared actor + single shared critic; the
    critic takes **local obs** (not global state), preserving IPPO's
    decentralised-critic identity. CPU SPS: 465. L4 projection: ~1300-1800.
  - **Fail-fast env-section guard** in `scripts/train.py` — process dies
    at startup if `environment.{muscle_groups,theta_max,tasks}` is empty.
  - **`--fresh-logs` flag** on `scripts/run_baselines.py` wipes
    `logs/{method}/ checkpoints/{method}/` before spawning subprocesses.
  - **All 4 configs** (hcmarl_full + 3 baselines) carry populated
    `environment:` blocks with PDF-verified Frey-Law 2012 F, R, r values.
  - **ECBF**: enabled in hcmarl_full_config.yaml; disabled in
    mappo/ippo/mappo_lag configs.
- **Phase A constants are intact** and PDF-verified (see Experiment 0
  output: `Results 0/Result constants_ledger.{txt,csv}` lists every
  constant with its source PDF and page number).
- Test suite: 0 failed on laptop (pytest 2026-04-23).

**Budget reality (L4 on-demand, Rs 98/hr):**
- 4 methods × 10 seeds × 2M steps × ~0.7ms/step on L4 ≈ 5500 run-seconds each
  (L4 is ~1.2-1.5x slower per step than L40S; CPU rollouts still dominate
  so the gap is smaller than raw TFLOPS would suggest)
- Total serial: 40 × 5500s = 61 hours → **Rs ~6,000 if run serial**
- With `--max-parallel 6`: wall-clock ≈ 8-12 hours → **Rs ~780-1,180**
- Per-run kill-switch: `--budget-inr 1500`. Per-run, not total.
- **Hard total stop: Rs 2,000** (user's total EXP1 budget). If E2E dashboard
  shows spend exceeding that, STOP and escalate immediately.

---

## 3. Pre-flight on the laptop (already done by LOCAL — reported here for VM to verify)

LOCAL confirmed before pushing:
1. `hcmarl/agents/ippo.py` — rewritten as parameter-shared; docstring cites
   Yu et al. 2022; `test_ippo_instantiation` updated to assert
   `critic.in_features == obs_dim` (the defining IPPO property).
2. `config/{hcmarl_full,mappo,ippo,mappo_lag}_config.yaml` — each carries:
   - `ecbf.enabled: false` for baselines / `true` for HCMARL
   - `environment:` with muscle_groups, theta_max, tasks (all populated)
   - `total_steps: 2000000`
3. `pytest -q` — passing on laptop.
4. Phase A constants (three_cc_r.py, ecbf_filter.py, nswf_allocator.py,
   real_data_calibration.py) — **untouched** since the PDF-verification pass.
5. `git push origin master`.

---

## 4. Execution plan (VM owns STEPs 1-13)

### STEP 0 — [USER] boot the VM + clone the LATEST repo

Manual, before Claude Code launches:

```bash
cd /root
git clone https://github.com/ADITYA-WORK-MAITI/hcmarl-project.git hcmarl_project
cd hcmarl_project
git log -1 --format='%h %ci %s'
ls RUNBOOK_EXP1.md && wc -l RUNBOOK_EXP1.md
```

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
lscpu | grep -E "Model name|Thread|Socket|CPU max MHz|CPU\(s\):"
free -h
df -h /root
```

Pass criteria: **NVIDIA L4** (or better), VRAM free ≥ 24 GiB, ≥ 48 vCPUs, RAM
free ≥ 100 GiB, disk free ≥ 200 GiB. Target machine spec (E2E rented node):
L4 / 50 vCPU / 220 GB RAM / 48 GB VRAM / 12 / 250 GB SSD / R: 60000 W: 30000.
Report to user regardless.

### STEP 2 — [VM] Create the venv

```bash
cd /root/hcmarl_project
python3.12 -m venv venv
source venv/bin/activate
python --version
pip install -q -U pip wheel
```

### STEP 3 — [VM] Install torch from CUDA wheel index

```bash
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"
```

Fallback to cu121 if cu124 reports `cuda: False`. If both fail, STOP.

### STEP 4 — [VM] Install the rest

```bash
pip install -q -r requirements.txt
pip list | grep -iE "^(torch|gymnasium|numpy|cvxpy|osqp|scipy|pyyaml)"
```

Torch version must include `+cu124` or `+cu121`. If pip replaced it with
the CPU wheel, re-run STEP 3 with `--force-reinstall`.

### STEP 5 — [VM] Pytest sanity

```bash
pytest -q 2>&1 | tail -5
```

Pass: **`0 failed`**. 1-3 skips acceptable (env-conditional). One harmless
CVXPY warning from `ecbf_filter.py` is expected.

### STEP 6 — [VM] The EIGHT non-negotiable pre-flight checks

```bash
echo "=== CHECK 1: git tip ==="
git log -1 --format='%h %ci %s'
echo
echo "=== CHECK 2: ECBF OFF in all three baselines ==="
grep -H 'enabled:' config/mappo_config.yaml config/ippo_config.yaml config/mappo_lag_config.yaml
echo
echo "=== CHECK 3: ECBF ON in hcmarl_full_config ==="
grep -E '^\s*enabled:' config/hcmarl_full_config.yaml
echo
echo "=== CHECK 4: environment section populated in all four ==="
for f in config/hcmarl_full_config.yaml config/mappo_config.yaml \
         config/ippo_config.yaml config/mappo_lag_config.yaml; do
  echo "--- $f ---"
  python -c "
import yaml
c = yaml.safe_load(open('$f'))
env = c.get('environment', {}) or {}
for k in ('muscle_groups','theta_max','tasks'):
    n = len(env.get(k) or {})
    print(f'  {k}: {n} entries')
    assert n > 0, f'{k} is empty in $f'
"
done
echo
echo "=== CHECK 5: logs + checkpoints clean for all four ==="
for m in hcmarl mappo ippo mappo_lag; do
  for d in logs checkpoints; do
    p="$d/$m"
    if [ -d "$p" ] && [ -n "$(ls -A "$p" 2>/dev/null)" ]; then
      echo "  NON-EMPTY: $p/  (--fresh-logs will wipe)"
    else
      echo "  clean: $p/"
    fi
  done
done
echo
echo "=== CHECK 6: experiment_matrix has ten seeds ==="
python - <<'PY'
import yaml
m = yaml.safe_load(open("config/experiment_matrix.yaml"))
seeds = m["headline"]["seeds"]
assert seeds == [0,1,2,3,4,5,6,7,8,9], f"expected 10 seeds, got {seeds}"
print(f"  headline.seeds = {seeds}  (count={len(seeds)})  OK")
PY
echo
echo "=== CHECK 7: IPPO is parameter-shared variant ==="
n=$(grep -c "parameter-shared\|PS-IPPO\|Yu et al. 2022" hcmarl/agents/ippo.py)
[ "$n" -ge 1 ] && echo "  OK ($n signatures found)" || (echo "  FAIL"; exit 1)
python -c "
from hcmarl.agents.ippo import IPPO
ip = IPPO(obs_dim=19, n_actions=6, n_agents=6)
assert hasattr(ip, 'actor') and hasattr(ip, 'critic'), 'IPPO not PS-refactored'
first = next(ip.critic.net.children())
assert first.in_features == 19, f'critic takes non-local obs ({first.in_features})'
print(f'  IPPO critic input dim = {first.in_features} (local obs, not global state) OK')
"
echo
echo "=== CHECK 8: dry-run banner ==="
python scripts/run_baselines.py --methods hcmarl mappo ippo mappo_lag --dry-run 2>&1 | head -5
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    [sys.executable, "scripts/run_baselines.py",
     "--methods","hcmarl","mappo","ippo","mappo_lag","--dry-run"],
    text=True).splitlines()
want1 = "Headline grid: 4 methods x 10 seeds = 40 runs"
want2 = "Methods: ['hcmarl', 'mappo', 'ippo', 'mappo_lag']"
assert out[0] == want1, f"LINE 1 wrong:\n  got:  {out[0]}\n  want: {want1}"
assert out[1] == want2, f"LINE 2 wrong:\n  got:  {out[1]}\n  want: {want2}"
print("  dry-run banner OK")
PY
echo
echo "All eight checks complete."
```

Expected outcome on fresh clone:
- CHECK 1 — recent commit hash.
- CHECK 2 — three `enabled: false` lines (mappo, ippo, mappo_lag).
- CHECK 3 — one `enabled: true` line (hcmarl).
- CHECK 4 — `6 entries` for muscle_groups and tasks, `6 entries` for theta_max in each of the 4 configs.
- CHECK 5 — eight `clean:` lines.
- CHECK 6 — ten-seed list confirmed.
- CHECK 7 — IPPO parameter-shared signature present, critic takes local obs (19-dim).
- CHECK 8 — `4 methods x 10 seeds = 40 runs` / `['hcmarl','mappo','ippo','mappo_lag']`.

**If any check fails, STOP. Post the full output. Do not proceed.**

### STEP 7 — [VM] Symlink `logs/` into `Results 1/`, then create tmux session

The training code writes to `logs/<method>/seed_<N>/` and
`checkpoints/<method>/seed_<N>/` by hardcoded path. Rather than writing
there and copying at the end, we symlink BOTH into `Results 1/`. Every
training artifact (CSVs, summary JSONs, model `.pt` weights,
`run_state.pt` resume files) lands inside the deliverable from the first
step. If the grid is interrupted mid-run, everything produced so far is
already inside `Results 1/` — no salvage step, no data loss.

The user has been explicit: reviewers / reproducibility checks need the
`.pt` weight files too. They belong in `Results 1/`.

```bash
# Preconditions from CHECK 5: logs/ + checkpoints/ are clean or absent.
# Defensive cleanup in case they exist as real dirs from a prior attempt:
rm -rf logs checkpoints 2>/dev/null || true

# Create the deliverable tree, then point logs/ AND checkpoints/ at it.
mkdir -p "Results 1/logs" "Results 1/checkpoints"
ln -sfn "Results 1/logs" logs
ln -sfn "Results 1/checkpoints" checkpoints

ls -la logs checkpoints      # must print two symlink-target lines
test -L logs && test -L checkpoints \
  || { echo "ERROR: logs or checkpoints is not a symlink"; exit 1; }

# Tmux session
tmux kill-session -t exp1 2>/dev/null || true
pkill -f "scripts/train.py" 2>/dev/null || true
pkill -f "run_baselines.py" 2>/dev/null || true
sleep 1
tmux new -d -s exp1
tmux send-keys -t exp1 "cd /root/hcmarl_project && source venv/bin/activate" Enter
sleep 1
tmux list-sessions | grep exp1
```

After this step, `ls -la logs checkpoints` MUST print two symlink lines
(`logs -> Results 1/logs`, `checkpoints -> Results 1/checkpoints`). If
either prints as a regular directory, the symlink didn't land — STOP
and retry the `ln -sfn` line. Training with a non-symlinked path will
still technically work but the deliverable will be missing that data.

### STEP 8 — [VM] Launch the grid

```bash
tmux send-keys -t exp1 "python scripts/run_baselines.py \
  --methods hcmarl mappo ippo mappo_lag \
  --device cuda \
  --fresh-logs \
  --max-parallel 6 \
  --budget-inr 1500 \
  --cost-per-hour 98.0 \
  2>&1 | tee 'Results 1/_exp1_run.log'" Enter
```

Within 30 seconds, `tmux capture-pane -t exp1 -p | tail -20` must show:
```
Headline grid: 4 methods x 10 seeds = 40 runs
Methods: ['hcmarl', 'mappo', 'ippo', 'mappo_lag']
Seeds:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

If the banner mentions **3 methods, 30 runs, or missing any of the 4 methods**,
immediately `Ctrl-C` and STOP.

**Flags explained — do not improvise:**
- `--methods hcmarl mappo ippo mappo_lag` — all four in one grid. Filter is
  explicit so if experiment_matrix.yaml adds a method later it won't
  silently join.
- `--device cuda` — L4.
- `--fresh-logs` — **NON-NEGOTIABLE**. Wipes `logs/{method}/` +
  `checkpoints/{method}/` before launching. Physical prevention against the
  2026-04-20 "append-on-matching-header" bug. Without this flag, STOP.
- `--max-parallel 6` — **6 concurrent seeds**. Rationale:
  - Prior calibration point: 3-way on L4 with 25 vCPUs = 8.3 vCPUs per
    process, and it ran cleanly. That is the ground truth.
  - Scaling honestly to 50 vCPUs while holding vCPUs-per-process constant:
    50 / 8.3 = 6.0 processes. Not 7, not 8.
  - 7-way on 50 vCPUs would give 7.1 vCPUs per process — TIGHTER than the
    validated baseline. That is oversubscription: BLAS threads thrash,
    per-seed SPS drops, wall-clock gets LONGER not shorter, and slow seeds
    risk tripping the `--budget-inr 1500` kill-switch spuriously.
  - BLAS thread cap: launcher auto-sets `OMP_NUM_THREADS=total_vcpus/max_parallel`
    = 50/6 ≈ 8, matching the L4/25-vCPU per-process thread count exactly.
  - L4 48 GB VRAM / ~500 MB per process = 96-way headroom, so GPU memory
    is not the constraint on this machine.
  - Expected wall-clock at 6-way on L4: ~8-12 hr (vs ~13 hr for the prior
    baselines-only 3-way run; parallelism doubled, but 4 methods × 10 seeds
    is ~4.4x more runs than baselines-only × 3 seeds).
- `--budget-inr 1500` — per-run kill-switch; does not cap total.
- `--cost-per-hour 98.0` — L4 on-demand at E2E. **Do not omit; kill-switch
  math depends on it.**
- `tee Results 1/_exp1_run.log` — launcher stdout mirror.

Do NOT add `--resume` — clean-slate grid.
Do NOT drop `--methods` — default would re-scan experiment_matrix.yaml.
Do NOT alter seed list — 10 seeds locked in the matrix.

### STEP 9 — [VM] Automated status reports (every 20 minutes)

**You do not wait to be asked.** Post this block every 20 min:

```
### Status report — <UTC HH:MM> (elapsed: <Hh:MMm> since STEP 8 kickoff)

Runs done / in-flight / pending:   <done>/<inflight>/<pending>  (of 40)
Current SPS (rolling 50 ep):       <sps>
ETA to grid completion:            ~<Hh:MMm>
Wall-clock spend so far:           Rs ~<amount>  (@ Rs 98/hr)
lazy_agent trips since start:      <count>
budget_tripped events:             <count>
pytest state:                      green (STEP 5) — no re-run

Last 5 lines of Results 1/_exp1_run.log:
  <paste>

Notes / minor fixes applied:       <describe or "none">
```

Field-gathering commands:
```bash
# n_done
for m in hcmarl mappo ippo mappo_lag; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    f=logs/$m/seed_$s/training_log.csv
    [ -f "$f" ] && [ "$(wc -l <"$f")" -ge 2 ] && echo "$m seed_$s"
  done
done | wc -l

# lazy/budget trips
grep -c "lazy-agent kill-switch" Results 1/_exp1_run.log 2>/dev/null || echo 0
grep -c "budget kill-switch"      Results 1/_exp1_run.log 2>/dev/null || echo 0
```

Additionally, post *immediately* (do not wait for the 20-min tick) when:
- A run finishes or fails.
- A `lazy-agent kill-switch` or `budget kill-switch` fires.
- `run_baselines.py` prints `FAILED (exit code X)`.
- You apply a §8 minor fix.

### STEP 10 — [VM] Final exit summary

When the launcher prints `All 40 jobs complete.`, post:

```
### EXP1 grid done — <UTC HH:MM> (total wall-clock: <Hh:MMm>)

| Method    | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | best_reward range |
|-----------|----|----|----|----|----|----|----|----|----|----|-------------------|
| hcmarl    |    |    |    |    |    |    |    |    |    |    |                   |
| mappo     |    |    |    |    |    |    |    |    |    |    |                   |
| ippo      |    |    |    |    |    |    |    |    |    |    |                   |
| mappo_lag |    |    |    |    |    |    |    |    |    |    |                   |
(cells: D=DONE, F=FAIL, L=lazy-trip, B=budget-trip)

Kill-switch events:
  lazy_agent trips: <count>  (list)
  budget trips:     <count>  (list)

Failures: <list or "none">

Total spend: Rs ~<amount>   (per-run budget: Rs 1,500; user hard-cap: Rs 2,000)
```

Best-reward gather:
```bash
python - <<'PY'
import json, os
for m in ("hcmarl","mappo","ippo","mappo_lag"):
    for s in range(10):
        p = f"logs/{m}/seed_{s}/summary.json"
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"{m} seed_{s}: best_reward={d.get('best_reward'):.1f} "
                  f"steps={d.get('total_steps')} trip={d.get('budget_tripped')}")
        else:
            print(f"{m} seed_{s}: MISSING summary.json")
PY
```

### STEP 11 — [VM] CSV audit (must pass before Results 1/ assembly)

```bash
python - <<'PY'
import csv, os, sys
missing, short, ok = [], [], []
for m in ("hcmarl","mappo","ippo","mappo_lag"):
    for s in range(10):
        # Training logs live under Results 1/logs/... thanks to the STEP 7 symlink.
        p = f"Results 1/logs/{m}/seed_{s}/training_log.csv"
        if not os.path.exists(p):
            missing.append(p); continue
        rows = list(csv.DictReader(open(p)))
        if not rows:
            short.append((p, 0)); continue
        last_step = int(rows[-1].get("global_step") or 0)
        (short if last_step < 1_000_000 else ok).append((p, last_step))
print(f"OK      ({len(ok)}):", *ok, sep="\n  ")
print(f"SHORT   ({len(short)}):", *short, sep="\n  ")
print(f"MISSING ({len(missing)}):", *missing, sep="\n  ")
if short or missing:
    sys.exit(1)
PY
echo "audit exit=$?"
```

Pass: `audit exit=0`. If any run is SHORT or MISSING, STOP, list them, wait
for LOCAL to decide whether to re-launch specific seeds.

### STEP 12 — [VM] Add provenance + aggregation to `Results 1/`

Training outputs are ALREADY inside `Results 1/logs/` (via the STEP 7
symlink). STEP 12 only adds the metadata that isn't written by the
training loop itself: the frozen configs, git/hardware provenance, a
pre-computed aggregation CSV, and a contents index.

```bash
# The experiment matrix + the four configs used (frozen copy for provenance).
# Copy into the deliverable so a future reviewer sees exactly what was run.
mkdir -p "Results 1/_configs_snapshot"
cp config/experiment_matrix.yaml "Results 1/_configs_snapshot/"
cp config/hcmarl_full_config.yaml config/mappo_config.yaml \
   config/ippo_config.yaml config/mappo_lag_config.yaml \
   "Results 1/_configs_snapshot/"

# Git commit hash + environment info (reproducibility)
{
  echo "# Experiment 1 — provenance snapshot"
  echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo
  echo "## Git"
  git log -1 --format='hash:    %H%nsubject: %s%nauthor:  %an <%ae>%ndate:    %ci'
  echo
  echo "## Python + Torch"
  python --version
  python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
  echo
  echo "## Hardware"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  lscpu | grep -E '^CPU\(s\):|Model name:' | head -2
  free -h | head -2
} > "Results 1/_provenance.txt"

# An aggregation summary (per-method best_reward + final cost + safety).
# Note: summary.json is now at Results 1/logs/<m>/seed_<s>/summary.json
# via the symlink; we just walk the tree directly.
python - <<'PY'
import csv, json
from pathlib import Path
OUT = Path("Results 1/_aggregation_summary.csv")
fields = ["method","seed","total_steps","best_reward","final_cost_ema",
          "final_safety_rate","budget_tripped","lazy_tripped"]
rows = []
for m in ("hcmarl","mappo","ippo","mappo_lag"):
    for s in range(10):
        p = Path(f"Results 1/logs/{m}/seed_{s}")
        summary = p / "summary.json"
        row = {"method": m, "seed": s}
        if summary.exists():
            d = json.loads(summary.read_text())
            for k in ("total_steps","best_reward","final_cost_ema",
                      "final_safety_rate","budget_tripped","lazy_tripped"):
                row[k] = d.get(k)
        rows.append(row)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"Aggregation CSV written: {OUT}")
PY

# Final index of what's in Results 1/
python - <<'PY'
from pathlib import Path
OUT = Path("Results 1/_INDEX.txt")
lines = ["# Results 1/ — contents index", ""]
for p in sorted(Path("Results 1").rglob("*")):
    if p.is_file():
        lines.append(f"  {p.relative_to('Results 1')}  ({p.stat().st_size} bytes)")
OUT.write_text("\n".join(lines))
print(f"Index written: {OUT}")
PY

echo "=== Results 1/ tree ==="
ls -la "Results 1/"
du -sh "Results 1/"
```

**Pass criterion for STEP 12:** `Results 1/_aggregation_summary.csv` exists
with 40 non-empty rows (one per method × seed), and
`Results 1/{hcmarl,mappo,ippo,mappo_lag}/seed_{0..9}/training_log.csv` all
exist. If any row is blank or any CSV is missing, STOP.

### STEP 13 — [VM] Stand down

```
EXP1 verified. Results 1/ assembled. Standing by for §0.2 close-out paste.
```

Then wait. Do not launch anything else.

---

## 5. Status-report cadence

Every **20 minutes** after STEP 8 kickoff. No exceptions. The grid runs for
2-4 hours; the user may step away. Every 20 minutes is the observability
guarantee.

Post *immediately* when:
- A run finishes (any of the 40).
- A `lazy-agent kill-switch` or `budget kill-switch` fires.
- `run_baselines.py` reports FAILED exit.
- You applied a §8 minor fix.
- You encountered a non-minor blocker and stopped.

---

## 6. Known failure modes (context for VM Claude)

### 6.1 Constraint-free baseline training (the env-section bug)
Cause: baseline configs missing `environment:` block → `safety_cost()` always
returns 0 → policy learns degenerate single-task policy → entropy collapses
→ lazy-agent kill-switch trips at ~100K steps.
Prevention: fail-fast guard in `scripts/train.py` + populated env blocks +
STEP 6 CHECK 4.

### 6.2 Contaminated CSV (append-on-matching-header bug)
Cause: `HCMARLLogger` appends to existing CSV if header matches.
Prevention: `--fresh-logs` flag + STEP 6 CHECK 5. STEP 8 ALWAYS uses
`--fresh-logs`.

### 6.3 IPPO 50-60 SPS (the slow-baseline bug, fixed in EXP1)
Cause: per-agent separate networks → n_agents CUDA launches per step.
Fix: parameter-shared IPPO (this runbook). Verified 465 SPS on CPU, expect
~1300-1800 SPS on L4.

### 6.4 HCMARL MMICRL MI-collapse (NOT a failure)
On real WSD4FEDSRM data, MMICRL returns K with MI≈0 (continuum, no modes).
The MI-collapse guard at `hcmarl/utils.py:189` falls back to per-worker
config-default ceilings. This is the designed behaviour, not a bug.
HCMARL then runs with uniform config ceilings = same as baselines. This
is expected and documented.

---

## 7. Files, paths, layout (quick reference)

```
/root/hcmarl_project/
├── RUNBOOK_EXP1.md                  ← THIS FILE
├── config/
│   ├── hcmarl_full_config.yaml      # ECBF enabled
│   ├── mappo_config.yaml            # ECBF off
│   ├── ippo_config.yaml             # ECBF off (PS-IPPO)
│   ├── mappo_lag_config.yaml        # ECBF off (cost_limit set)
│   └── experiment_matrix.yaml       # seeds [0..9]
├── hcmarl/agents/
│   ├── ippo.py                      # PARAMETER-SHARED (Yu et al. 2022 variant)
│   ├── mappo.py                     # shared actor + CTDE critic
│   └── mappo_lag.py                 # Lagrangian-constrained MAPPO
├── scripts/
│   ├── train.py                     # fail-fast env guard
│   └── run_baselines.py             # --fresh-logs + --max-parallel
├── logs/                            # SYMLINK -> Results 1/logs/ (set up in STEP 7)
├── checkpoints/                     # SYMLINK -> Results 1/checkpoints/ (STEP 7)
└── Results 1/                       # ← FINAL DELIVERABLE (live from STEP 7 onward)
    ├── logs/                        # training CSVs/JSONs land here via symlink
    │   ├── hcmarl/seed_{0..9}/{training_log.csv, summary.json, mmicrl_results.json}
    │   ├── mappo/seed_{0..9}/{training_log.csv, summary.json}
    │   ├── ippo/seed_{0..9}/{training_log.csv, summary.json}
    │   └── mappo_lag/seed_{0..9}/{training_log.csv, summary.json}
    ├── checkpoints/                 # .pt weights + run_state.pt land here via symlink
    │   ├── hcmarl/seed_{0..9}/{checkpoint_*.pt, checkpoint_best.pt, checkpoint_final.pt, run_state.pt}
    │   ├── mappo/seed_{0..9}/{same}
    │   ├── ippo/seed_{0..9}/{same}
    │   └── mappo_lag/seed_{0..9}/{same}
    ├── _configs_snapshot/           # frozen 4 configs + matrix (written STEP 12)
    ├── _provenance.txt              # git hash, torch, hardware (written STEP 12)
    ├── _aggregation_summary.csv     # one row per (method, seed) (written STEP 12)
    ├── _exp1_run.log                # launcher stdout (teed live from STEP 8)
    └── _INDEX.txt                   # file listing (written STEP 12)
```

**Why symlinks and not direct writes:** `scripts/train.py` and
`hcmarl/logger.py` have `logs/` and `checkpoints/` as hardcoded output
roots. Refactoring those paths is a bigger change than EXP1 wants. The
symlinks give us the same outcome (everything lands in `Results 1/`)
with zero source edits, zero code risk, and a single-folder deliverable.

**Why both logs/ AND checkpoints/ go inside Results 1/:** reviewers and
future reproducibility checks need the trained `.pt` weights. Leaving
them out would hand over analysis-only data and lose the models
themselves. Deliverable = everything needed to audit, reproduce, or
re-evaluate. Size impact: ~50 MB × 4 methods × 10 seeds = 2 GB. scp
over a decent connection pulls that in 1-3 minutes.

---

## 8. Minor-blocker authority (what VM Claude MAY fix in place)

**MAY fix (report every one):**
- Missing pip package that requirements.txt should have pulled.
- Typo'd import in a **test** file that blocks collection (not source).
- Stale `.pyc` (`find . -name __pycache__ -exec rm -rf {} +`).
- CUDA visibility (`export CUDA_VISIBLE_DEVICES=0`).
- Tmux session recovery (new session, same name, ONLY if all CSVs are
  intact — if tmux died mid-grid, STOP because `--fresh-logs` would wipe
  partials).

**MUST NOT touch:**
- Any file under `hcmarl/` (source).
- `scripts/train.py`, `scripts/run_baselines.py`,
  `scripts/aggregate_learning_curves.py`.
- Any `config/*.yaml`.
- `tests/*.py` beyond trivial import-typo fixes.
- `requirements.txt` — use `--force-reinstall` if needed.
- `.git`, `git add/commit/push`.
- Kill-switches.
- Phase A constants (three_cc_r, ecbf_filter, nswf_allocator,
  real_data_calibration) — these are PDF-verified and SACRED.

---

## 9. Emergency procedures

| Event | Response |
|---|---|
| SSH drops mid-run | tmux keeps running. `tmux attach -t exp1`, resume reports. |
| pytest fails STEP 5 | STOP. Post last 30 lines. |
| Any STEP 6 check fails | STOP. Post full output. Wait for LOCAL. |
| CUDA not available after STEP 3 | Retry cu121. If still False, STOP. |
| lazy-agent trip on any seed | Post immediately. Do NOT disable. Let grid finish. |
| budget trip on any seed | Post immediately. That seed halts cleanly; grid continues. |
| Hang with no log output > 20 min | `tmux capture-pane -t exp1 -p | tail -200`. Don't kill. |
| STEP 11 audit short/missing | Post the output. Wait for LOCAL. |
| E2E billing out of line | STOP status reports, user checks dashboard. |
| Rate limit on Claude | User has standby account. Relaunch claude, re-paste §0.1. |

---

## 10. Results-format summary (for the record)

Every file under `Results 1/` is:
- **Human-readable** — CSVs open in a spreadsheet; JSONs pretty-print;
  summary.json and training_log.csv already exist per seed with labelled
  columns.
- **Python-analyzable** — all numeric data in CSV / JSON, no squeezed
  aggregations at the VM stage (that happens in EXP4 locally).
- **Claude-interpretable** — plain text + structured data throughout.
- **Visualizable** — training_log.csv per seed has one row per eval episode
  with (episode, global_step, reward, cost, safety_rate, peak_MF, ...)
  ready for pandas/matplotlib/seaborn.

No visualization / analysis / interpretation on VM. That is Experiment 4,
done locally on the laptop.

---

## 11. End-of-session pull commands (USER runs locally after §0.2)

On the laptop, after VM replies `EXP1 complete. Standing by.`:

Replace `<VM_IP>` with the IP from the E2E dashboard. `<KEY>` is your
SSH private key (default `~/.ssh/id_ed25519`).

All three pulls land in `/c/Users/admin/Downloads/`. After the three pulls
succeed you move each folder wherever you want on the laptop — the user
will manage placement. The VM side only guarantees that `Results 1/`,
`logs/`, and `checkpoints/` on the VM each carry the complete training
artefacts (because `logs/` and `checkpoints/` on the VM are symlinks that
point INTO `Results 1/`, so pulling any of the three gives the same bytes).

### 11.a Primary pull — one folder gets EVERYTHING

Because STEP 7 symlinked both `logs/` and `checkpoints/` inside
`Results 1/`, a single pull of `Results 1/` fetches:
- every training_log.csv, summary.json, mmicrl_results.json
- every checkpoint_*.pt, checkpoint_best.pt, checkpoint_final.pt, run_state.pt
- the frozen configs snapshot, provenance, aggregation, launcher log, index

`Results 1/logs/` and `Results 1/checkpoints/` are real directories on
the VM (they are the symlink *targets*, not the symlinks themselves).
Rsync and scp copy them as real directories.

```bash
# PRIMARY — single pull gets everything (~2-3 GB)
rsync -avzP -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/"Results 1" \
    /c/Users/admin/Downloads/
```

Or with scp:
```bash
scp -r -i ~/.ssh/id_ed25519 \
    root@<public-ip>:/root/hcmarl_project/"Results 1" \
    /c/Users/admin/Downloads/
```

Verify:
```bash
du -sh "/c/Users/admin/Downloads/Results 1/"
ls "/c/Users/admin/Downloads/Results 1/logs/"
ls "/c/Users/admin/Downloads/Results 1/checkpoints/"
cat "/c/Users/admin/Downloads/Results 1/_INDEX.txt" | head -30
```

Expected: `Results 1/` is ~2-3 GB (2 GB checkpoints + ~30 MB CSVs/JSONs
+ metadata). `logs/` has 40 seed subdirectories, `checkpoints/` has 40
seed subdirectories.

### 11.b Belt-and-braces — three separate pulls for maximum redundancy

Per the user's explicit request ("I don't wanna lose nothing"), run
these **after** 11.a succeeds. They fetch the same data redundantly
(the VM-side `logs/` and `checkpoints/` are symlinks that resolve into
the same files already inside `Results 1/`, so these are second copies
of the same bytes, not additional bytes).

The `-L` flag here dereferences the top-level symlinks on the VM so the
pull copies real files, not dangling symlink markers.

```bash
# BELT-AND-BRACES #1 — independent mirror of logs/ (CSVs + JSONs + mmicrl results, ~50 MB)
rsync -avzP -L -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/logs/ \
    /c/Users/admin/Downloads/logs_exp1_mirror/

# BELT-AND-BRACES #2 — independent mirror of checkpoints/ (.pt weights, ~2 GB)
rsync -avzP -L -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/checkpoints/ \
    /c/Users/admin/Downloads/checkpoints_exp1_mirror/
```

After all three pulls succeed, `/c/Users/admin/Downloads/` contains:
- `Results 1/` (the canonical deliverable — CSVs + JSONs + .pt + metadata)
- `logs_exp1_mirror/` (second copy of CSVs + JSONs only)
- `checkpoints_exp1_mirror/` (second copy of .pt weights only)

Any single corruption doesn't lose the data. User moves them locally as
needed after the pulls complete.

### 11.c After pulls complete

1. `du -sh /c/Users/admin/Downloads/"Results 1"` — sanity-check the size (expect 2-3 GB).
2. Spot-check one CSV: `head -3 "/c/Users/admin/Downloads/Results 1/logs/hcmarl/seed_0/training_log.csv"`
3. Spot-check one checkpoint: `ls -la "/c/Users/admin/Downloads/Results 1/checkpoints/hcmarl/seed_0/"`
4. Send §0.2 close-out paste to VM; VM acknowledges.
5. Destroy the E2E node from the dashboard.
6. Confirm billing has stopped.

---

## 12. End-of-session checklist (user runs after §0.2 paste)

- [ ] VM final summary (§10 format) posted
- [ ] STEP 11 audit exit=0
- [ ] STEP 12 `Results 1/_INDEX.txt` exists with 40+ seed entries
- [ ] §11.a primary `rsync` of `Results 1/` → `/c/Users/admin/Downloads/`
      completed; folder has logs/ + checkpoints/ + metadata
- [ ] §11.b mirror #1 of `logs/` → `/c/Users/admin/Downloads/logs_exp1_mirror/`
      completed
- [ ] §11.b mirror #2 of `checkpoints/` → `/c/Users/admin/Downloads/checkpoints_exp1_mirror/`
      completed
- [ ] `ls /c/Users/admin/Downloads/"Results 1"/logs/{hcmarl,mappo,ippo,mappo_lag}/seed_*/training_log.csv`
      returns 40 non-empty paths
- [ ] §0.2 close-out sent; VM acknowledged
- [ ] E2E node destroyed in dashboard
- [ ] Billing shows node charge stopped

Only after all ten boxes tick is this session "done."

---

## 13. What EXP1 deliberately does NOT do (scope discipline)

| Item | Why deferred | To |
|---|---|---|
| Visualisation / plotting | VM time is expensive; matplotlib on headless VM is a footgun | EXP4 (laptop) |
| Statistical aggregation (IQM, bootstrap CI) | Needs all 40 CSVs present first | EXP4 (laptop) |
| Ablation grid | Separate run, 5 rungs × 5 seeds, different launcher | EXP2 (future VM session) |
| Path G real-data eval | CPU-side, 3-minute job, no need to burn L40S on it | EXP3 (laptop or cheap VM) |
| Synthetic K=3 MMICRL validation | CPU-side sanity check | EXP3 (laptop) |
| Paper writing | LOCAL-only, not a VM job | POST-EXP4 |

Keep EXP1 narrow: produce 40 clean training CSVs and the `Results 1/`
deliverable. Everything else comes later.
