# RUNBOOK_EXP2 — HC-MARL remove-one ablation grid (2026-04-24)

> **If you are Claude Code running inside the rented E2E VM terminal, this
> entire file is your briefing. Read every line top to bottom — do not skim,
> do not skip, do not summarise. The §0.1 boot prompt tells you exactly when
> to act and when to stop. Follow it.**

This is **Experiment 2** in the user's numbered experiment plan. EXP1
(`RUNBOOK_EXP1.md`) was the headline run (4 methods × 10 seeds). EXP2 is
the **REMOVE-ONE ABLATION** attribution study: 4 rungs × 5 seeds = 20 runs
on L4 + 50 vCPU.

**IMPORTANT:** EXP2 does **NOT** re-run full HC-MARL. The 5th reference
point (full HC-MARL baseline) comes from EXP1's first 5 seeds
(`logs/hcmarl/seed_{0..4}/`). On THIS VM (the EXP2 ablation node) those
EXP1 logs do NOT need to exist — the LOCAL post-pull aggregator merges
`Results 1/` and `Results 2/` after both runs finish. Do not re-run HCMARL.
Do not look for EXP1 artefacts on this VM; they live on a separate machine
and on the user's laptop.

**Concurrent-execution mode (2026-04-25, dual-VM run):**
EXP1 and EXP2 may be running simultaneously on two separate L4 nodes (two
E2E accounts). Each VM has its own `logs/`, `checkpoints/`, and `Results N/`
deliverable. The two VMs do NOT share state, do NOT share IPs, and do NOT
coordinate. EXP2's pre-flight does NOT check for `Results 1/` because that
folder lives on the EXP1 VM (or the laptop), never on the EXP2 VM. Proceed
without looking for EXP1 outputs.

---

## 0. What this runbook covers — and what it does NOT

**In scope:**
- Four REMOVE-ONE ablations, each differing from full HC-MARL by exactly
  one component:
  | Rung            | Ablated component                                   |
  |-----------------|-----------------------------------------------------|
  | `no_ecbf`       | ECBF safety filter off                              |
  | `no_nswf`       | NSWF Hungarian allocator off                        |
  | `no_divergent`  | Disagreement utility flat (D_i = kappa)             |
  | `no_reperfusion`| 3CC-r reperfusion multiplier r = 1 (vs 15/30)       |
- Five seeds each: 0, 1, 2, 3, 4  → **20 total runs**
- Config files: `config/ablation_no_{ecbf,nswf,divergent,reperfusion}.yaml`
- Launcher: `scripts/run_ablations.py` (reads `config/experiment_matrix.yaml`)
- Hardware: **L4 GPU (48 GB VRAM), 50 vCPUs, Rs 98/hr**
- `--max-parallel 6` (6 seeds concurrent; same rationale as EXP1 §4 STEP 8)
- Logs target: `logs/ablation_<rung>/seed_{0..4}/training_log.csv`
- Checkpoints target: `checkpoints/ablation_<rung>/seed_{0..4}/*.pt`
- Final deliverable: `Results 2/` (symlink-populated from STEP 7 onward).

**Out of scope:**
- HCMARL retrain / EXP1 redo — HCMARL full runs are EXP1 only.
  Comparison reference = first 5 seeds of EXP1 (`logs/hcmarl/seed_{0..4}/`).
- `no_mmicrl` ablation — redundant (MMICRL is null-op on real data via
  the MI-collapse guard; removing it yields identical behaviour to leaving
  it on). Documented honestly in the paper, not measured here.
- Synthetic K=3 experiment (EXP3, laptop/CPU only).
- IQM + bootstrap CI computation (LOCAL, post-pull; aggregator is CPU-only
  and takes ~30 s).
- Visualisation / analysis / interpretation (EXP4, LOCAL only).
- Any edit to `hcmarl/` source or Phase A constants (3CC-r / ECBF / NSWF /
  MMICRL / POPULATION_FR).

---

## 0.1 Boot prompt (paste into VM-side Claude Code immediately after `claude` launches)

```
You are the VM-side Claude for the HC-MARL Experiment 2 REMOVE-ONE ablation
run on an E2E L4 GPU node (2026-04-25). This paste IS your full starting
instruction.

CONCURRENT-EXECUTION NOTICE: EXP1 is running RIGHT NOW on a different L4
node (a separate E2E account, separate IP, separate VM). That EXP1 VM
produces `Results 1/`. You will NOT see `Results 1/` on THIS VM and you
should NOT look for it. Do NOT block on its absence. Do NOT scp from the
other VM. The HCMARL reference merges with your ablation outputs LOCALLY
on the user's laptop after both VMs finish (LOCAL EXP4 step). On THIS VM,
your sole job is producing 20 clean ablation CSVs in `Results 2/`.

Your complete briefing is at /root/hcmarl_project/RUNBOOK_EXP2.md on this
VM. Read every line of that file top to bottom. Do not skim, do not skip,
do not summarise.

CRITICAL NON-NEGOTIABLE POINTS (if any of these is not true, STOP):
  1. You cloned the MOST RECENT push from
     github.com/ADITYA-WORK-MAITI/hcmarl-project (branch master) and the
     tip includes the EXP2 matrix changes (ablation.rungs has 4 entries
     named no_ecbf/no_nswf/no_divergent/no_reperfusion, NOT the old
     build-up ladder).
  2. Matrix verification:
     `python -c "import yaml; m=yaml.safe_load(open('config/experiment_matrix.yaml'))['ablation']; \
       print('rungs:', [r['name'] for r in m['rungs']]); print('seeds:', m['seeds'])"`
     Expected:
       rungs: ['no_ecbf', 'no_nswf', 'no_divergent', 'no_reperfusion']
       seeds: [0, 1, 2, 3, 4]
  3. All 4 ablation configs declare total_steps=2000000 (aligned with EXP1
     HCMARL reference). Prove it:
     `grep -H 'total_steps:' config/ablation_no_{ecbf,nswf,divergent,reperfusion}.yaml`
     — every line must show `total_steps: 2000000`.
  4. Each of the 4 configs flips EXACTLY ONE axis vs full HC-MARL:
     `pytest tests/test_batch_d.py -q 2>&1 | tail -3`
     — must show `0 failed`. The parameterized test
     `test_remove_one_config_flips_exactly_one_axis` asserts the single-
     axis-flip invariant.
  5. logs/ablation_* and checkpoints/ablation_* are EMPTY on the VM
     (either fresh clone or will be wiped by --fresh-logs before
     subprocess launch). NEVER append to a contaminated CSV.
  6. IPPO is still the parameter-shared variant (EXP1 delta); not used
     in EXP2 but must not have drifted:
     `grep -c "parameter-shared\|PS-IPPO\|Yu et al. 2022" hcmarl/agents/ippo.py`
     must print >= 1.
  7. Dry-run banner verification:
     `python scripts/run_ablations.py --dry-run 2>&1 | head -5`
     FIRST LINE must contain:
     `Ablation grid: 4 rungs x 5 seeds = 20 runs`  (or equivalent)
     and list rung names ['no_ecbf','no_nswf','no_divergent','no_reperfusion'].

If ANY of the 7 checks above fails, STOP, post the failing check, wait.

After reading, execute STEPs 1-13 of RUNBOOK_EXP2.md continuously, in
order, without asking for confirmation between steps:

  STEP 1   hardware sanity (expect L4 or better, >=24 GB VRAM, >=48 vCPUs)
  STEP 2   python3.12 venv
  STEP 3   torch from cu124 (fall back to cu121)
  STEP 4   pip install -r requirements.txt
  STEP 5   pytest -q  (pass: 0 failed)
  STEP 6   the SEVEN pre-flight checks above
  STEP 7   symlink logs/ and checkpoints/ into Results 2/, create tmux `exp2`
  STEP 8   launch: scripts/run_ablations.py --max-parallel 6 --fresh-logs
  STEP 9   status report every 20 minutes (format in §5)
  STEP 10  when grid finishes, post exit summary (§6)
  STEP 11  CSV audit must pass before standing down
  STEP 12  add provenance + aggregation metadata to Results 2/
  STEP 13  STOP. Wait for the §0.2 close-out paste.

Minor-blocker policy (§8): same as EXP1. Do NOT modify hyperparameters,
seeds, total_steps, ECBF states, env parameters, Phase A constants, or
anything under hcmarl/ source.

Begin now. After reading RUNBOOK_EXP2.md, reply with exactly:
`Read. Seven pre-flight checks begin at STEP 6. Results land in 'Results 2/'.`
and start.
```

---

## 0.2 Close-out prompt (paste after CSV audit passes and Results 2/ is populated)

```
Experiment 2 complete. Your work is done for this session.

The user will now scp Results 2/ (and mirrors) to the laptop and destroy
the E2E node manually.

Do NOT destroy the node. Do NOT delete anything. Do NOT git commit.
Do NOT start more training.

Reply with exactly
  `EXP2 complete. Standing by. Local will close the session.`
and then wait.
```

---

## 1. Two-agent division of labour

Identical to EXP1 §1. Summary:

| Agent | Owns | Forbidden |
|---|---|---|
| **LOCAL** (laptop) | scope, git, RUNBOOK edits, scp pull, node destruction | direct VM SSH |
| **VM** (Claude on L4) | bootstrap, 7 pre-flight checks, tmux launch, 20-min reports, minor §8 fixes, Results 2/ assembly | scope changes, hcmarl/ source edits, git commits, destroying node |
| **USER** | E2E dashboard, SSH, clone before claude, paste between sessions, scp pull, destroy | training outside tmux |

---

## 2. Project state at the start of this session

- Repo tip on origin/master carries BOTH EXP1 and EXP2 deltas. EXP1
  unchanged; EXP2 additions are:
  - `config/experiment_matrix.yaml` — ablation section rewritten as 4
    REMOVE-ONE rungs (was 5-rung build-up ladder). `curve_anchors_steps`
    dropped 3M (unreachable at 2M total_steps).
  - `config/ablation_no_{ecbf,nswf,divergent,reperfusion}.yaml` —
    `total_steps: 2000000` (was 3M; aligned with EXP1 HCMARL reference).
  - `tests/test_batch_d.py` — TestD2 rewritten for remove-one semantics;
    new `test_remove_one_config_flips_exactly_one_axis` parameterised
    test pins each rung flips exactly one axis vs full HC-MARL.
  - `RUNBOOK_EXP2.md` (this file).
- **Phase A constants intact** — PDF-verified values in `three_cc_r.py`,
  `ecbf_filter.py`, `nswf_allocator.py`, `mmicrl.py`, `real_data_calibration.py`,
  and `muscle_groups` blocks in all configs. STEP 6 CHECK 4 `pytest`
  proves this.
- **EXP1 reference is OFF-VM, not a pre-flight gate.** EXP1 may be running
  concurrently on a separate L4 node, or may have already completed and
  been pulled to the laptop. Either way, `Results 1/` does NOT exist on
  THIS VM and is NOT expected to. Do NOT block on its absence. Do NOT
  attempt to scp it from the other VM. The HCMARL reference merges with
  these ablation results LOCALLY post-pull, in EXP4 (laptop-side
  aggregation). On this VM, just produce 20 clean ablation runs.

**Budget reality (L4 on-demand, Rs 98/hr):**
- 4 rungs × 5 seeds × 2M steps. Per-seed time depends on rung:
  - `no_ecbf`        ~baseline-class speed (no ECBF QP solves) → ~75 min/seed
  - `no_nswf`        ~95% HCMARL speed → ~130 min/seed
  - `no_divergent`   ~95% HCMARL speed → ~130 min/seed
  - `no_reperfusion` ~HCMARL speed (3CC-r ODE same cost) → ~135 min/seed
- 20 runs at `--max-parallel 6`: rung-major launcher runs each rung's 5
  seeds concurrently (5 fits in 6 with one slot idle), so 4 sequential
  rung-waves of ~75/130/130/135 min ≈ **~7-9 hours wall-clock**.
- Expected spend: **Rs ~690-880** (well under user's Rs 2,000 cap).
- Per-run kill-switch: `--budget-inr 1500` (per run, not total).
- Hard total stop: **if E2E dashboard shows spend > Rs 2,000, STOP.**

---

## 3. Pre-flight on the laptop (already done by LOCAL)

LOCAL confirmed before pushing:
1. `config/experiment_matrix.yaml` — ablation section has exactly 4
   remove-one rungs; seeds `[0, 1, 2, 3, 4]`; curve_anchors drops 3M.
2. All 4 ablation configs: `total_steps: 2000000` (was 3M).
3. `tests/test_batch_d.py` — remove-one test passes for all 4 rungs.
4. `pytest -q` — 0 failed on laptop.
5. Phase A constants unchanged since EXP1.

---

## 4. Execution plan (VM owns STEPs 1-13)

### STEP 0 — [USER] boot VM + clone repo

See `C:\Users\admin\Desktop\NOTEPADS\RUN INITIAL.txt` for the canonical
sequence. Summary:

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@<public-ip>

apt-get update -q
apt-get install -y -q curl ca-certificates git tmux htop python3.12-venv python3-pip nodejs npm
apt-get install -y -q nvtop || true

cd /root
if [ ! -d "hcmarl_project" ]; then
  git clone https://github.com/ADITYA-WORK-MAITI/hcmarl-project.git hcmarl_project
fi

cd hcmarl_project
git pull origin master
git log --oneline -5     # top commit must include EXP2 matrix changes

ls RUNBOOK_EXP2.md && git log -1 --format='%h %s'

curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version

tmux new -s run
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Paste the §0.1 boot prompt.

### STEP 1 — [VM] Hardware sanity

```bash
nvidia-smi | head -25
lscpu | grep -E "Model name|Thread|Socket|CPU max MHz|CPU\(s\):"
free -h
df -h /root
```

Pass: **NVIDIA L4** (or better), VRAM ≥ 24 GB, ≥ 48 vCPUs, RAM ≥ 100 GB, disk
≥ 200 GB. Target machine spec (E2E rented node, second account):
L4 / 50 vCPU / 220 GB RAM / 48 GB VRAM / 12 / 250 GB SSD / R: 60000 W: 30000.

### STEP 2 — [VM] Create venv

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

Fallback to cu121 if cu124 reports False. STOP if both fail.

### STEP 4 — [VM] Install rest

```bash
pip install -q -r requirements.txt
pip list | grep -iE "^(torch|gymnasium|numpy|cvxpy|osqp|scipy|pyyaml)"
```

Torch version must include `+cu124` or `+cu121`. Re-run STEP 3 with
`--force-reinstall` if CPU wheel crept in.

### STEP 5 — [VM] Pytest sanity

```bash
pytest -q 2>&1 | tail -5
```

Pass: `0 failed`. Expected 1-3 skips (env-conditional). CVXPY harmless
warning on `test_stress_2x_still_holds` does not count as failure.

### STEP 6 — [VM] The SEVEN non-negotiable pre-flight checks

```bash
echo "=== CHECK 1: git tip ==="
git log -1 --format='%h %ci %s'
echo

echo "=== CHECK 2: matrix has 4 remove-one rungs ==="
python - <<'PY'
import yaml
m = yaml.safe_load(open("config/experiment_matrix.yaml"))["ablation"]
names = [r["name"] for r in m["rungs"]]
assert names == ["no_ecbf","no_nswf","no_divergent","no_reperfusion"], \
    f"rung names wrong: {names}"
assert m["seeds"] == [0,1,2,3,4], f"seeds wrong: {m['seeds']}"
assert all(r["method"] == "hcmarl" for r in m["rungs"]), \
    "all rungs must use method=hcmarl for remove-one semantics"
print(f"  rungs: {names}  OK")
print(f"  seeds: {m['seeds']}  OK")
PY
echo

echo "=== CHECK 3: all 4 configs at total_steps=2000000 ==="
grep -H 'total_steps:' config/ablation_no_ecbf.yaml \
     config/ablation_no_nswf.yaml \
     config/ablation_no_divergent.yaml \
     config/ablation_no_reperfusion.yaml
python - <<'PY'
import yaml
for name in ("no_ecbf","no_nswf","no_divergent","no_reperfusion"):
    c = yaml.safe_load(open(f"config/ablation_{name}.yaml"))
    steps = c["training"]["total_steps"]
    assert steps == 2_000_000, f"{name}: total_steps={steps}"
    print(f"  ablation_{name}.yaml: total_steps=2,000,000  OK")
PY
echo

echo "=== CHECK 4: single-axis-flip invariant (pytest D2) ==="
pytest tests/test_batch_d.py::TestD2AttributionAblationMatrix -q --no-header 2>&1 | tail -3
echo

echo "=== CHECK 5: logs + checkpoints clean for all 4 rungs ==="
for r in no_ecbf no_nswf no_divergent no_reperfusion; do
  for d in logs checkpoints; do
    p="$d/ablation_$r"
    if [ -d "$p" ] && [ -n "$(ls -A "$p" 2>/dev/null)" ]; then
      echo "  NON-EMPTY: $p/  (--fresh-logs will wipe)"
    else
      echo "  clean: $p/"
    fi
  done
done
echo

echo "=== CHECK 6: IPPO (EXP1 delta) still PS variant ==="
n=$(grep -c "parameter-shared\|PS-IPPO\|Yu et al. 2022" hcmarl/agents/ippo.py)
[ "$n" -ge 1 ] && echo "  OK ($n signatures)" || (echo "  FAIL"; exit 1)
echo

echo "=== CHECK 7: dry-run banner ==="
python scripts/run_ablations.py --dry-run 2>&1 | head -10
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    [sys.executable, "scripts/run_ablations.py", "--dry-run"],
    text=True).splitlines()
joined = " ".join(out).lower()
assert "no_ecbf" in joined and "no_nswf" in joined \
       and "no_divergent" in joined and "no_reperfusion" in joined, \
    f"dry-run missing one of the 4 rung names:\n{out}"
assert "20" in joined or "4 rungs" in joined or "5 seeds" in joined, \
    f"dry-run missing count signal:\n{out}"
print("  dry-run banner OK")
PY
echo

echo "All seven checks complete."
```

Expected on fresh clone:
- CHECK 1 — recent commit.
- CHECK 2 — rungs list + seeds list both OK.
- CHECK 3 — four "total_steps=2,000,000" lines all OK.
- CHECK 4 — `0 failed`.
- CHECK 5 — eight `clean:` lines.
- CHECK 6 — `OK (1+ signatures)`.
- CHECK 7 — banner contains all 4 rung names and the 20-run signal.

**If any check fails, STOP.**

### STEP 7 — [VM] Symlink logs/ + checkpoints/ into Results 2/, create tmux

Same symlink trick as EXP1, but for `Results 2/`. Training writes land
directly inside the deliverable.

```bash
# Defensive cleanup (CHECK 5 should have confirmed they're clean; this
# catches the case where they exist as real dirs from a prior attempt):
rm -rf logs checkpoints 2>/dev/null || true

# Create deliverable tree, then symlink top-level logs/ + checkpoints/.
mkdir -p "Results 2/logs" "Results 2/checkpoints"
ln -sfn "Results 2/logs" logs
ln -sfn "Results 2/checkpoints" checkpoints

ls -la logs checkpoints      # must print two symlink lines
test -L logs && test -L checkpoints \
  || { echo "ERROR: logs or checkpoints is not a symlink"; exit 1; }

# Tmux
tmux kill-session -t exp2 2>/dev/null || true
pkill -f "scripts/train.py" 2>/dev/null || true
pkill -f "run_ablations.py" 2>/dev/null || true
sleep 1
tmux new -d -s exp2
tmux send-keys -t exp2 "cd /root/hcmarl_project && source venv/bin/activate" Enter
sleep 1
tmux list-sessions | grep exp2
```

After this, `ls -la logs checkpoints` MUST print two symlink lines
(`logs -> Results 2/logs`, `checkpoints -> Results 2/checkpoints`). If
either is a regular directory, the symlink didn't land — STOP and retry.

### STEP 8 — [VM] Launch the grid

```bash
tmux send-keys -t exp2 "python scripts/run_ablations.py \
  --device cuda \
  --max-parallel 6 \
  --budget-inr 1500 \
  --cost-per-hour 98.0 \
  2>&1 | tee 'Results 2/_exp2_run.log'" Enter
```

> **Note:** `scripts/run_ablations.py` does NOT currently have a
> `--fresh-logs` flag (unlike `run_baselines.py`). The defensive `rm -rf`
> in STEP 7 is the guarantee. If the launcher has been updated to support
> `--fresh-logs` in a later commit, add it here.

Within 30 seconds, `tmux capture-pane -t exp2 -p | tail -20` must show
the dry-run-style banner listing the 4 rung names and total-run count.

**Flags explained — do not improvise:**
- `--device cuda` — L4.
- `--max-parallel 6` — 6 concurrent seeds. Rationale (matches EXP1):
  - Prior calibration: L4 with 25 vCPUs ran 3 seeds cleanly = 8.3 vCPUs
    per process. Scaled honestly to 50 vCPUs: 50/8.3 = 6.0 processes.
  - 7-way on 50 vCPUs would be tighter per-process than the validated
    L4/25-vCPU baseline → BLAS thread thrash, slower per-seed SPS,
    longer wall-clock. Same setting as EXP1 for consistency.
  - L4 48 GB VRAM (2× L4 cards on this E2E SKU) easily handles 20-way
    concurrency; CPU is the true constraint.
  - With 5 seeds per rung and 6-way parallel, all 5 seeds of a rung run
    concurrently (one slot idle) — clean rung-major scheduling.
- `--budget-inr 1500` — per-run kill-switch (not total).
- `--cost-per-hour 98.0` — L4 on-demand at E2E. Do not omit; kill-switch
  arithmetic depends on it.
- `tee 'Results 2/_exp2_run.log'` — launcher stdout mirror inside the
  deliverable (lands via working-dir; NOT via symlink).

Do NOT add `--resume` — clean-slate grid.
Do NOT alter rung list or seeds — both are locked in the matrix.

### STEP 9 — [VM] Status reports every 20 minutes

```
### Status report — <UTC HH:MM> (elapsed: <Hh:MMm> since STEP 8 kickoff)

Runs done / in-flight / pending:  <done>/<inflight>/<pending>  (of 20)
Current SPS (rolling 50 ep):      <sps>
ETA to grid completion:           ~<Hh:MMm>
Wall-clock spend so far:          Rs ~<amount>  (@ Rs 98/hr)
lazy_agent trips since start:     <count>
budget_tripped events:            <count>
pytest state:                     green (STEP 5) — no re-run

Last 5 lines of Results 2/_exp2_run.log:
  <paste>

Notes / minor fixes applied:      <describe or "none">
```

Field commands:
```bash
# n_done
for r in no_ecbf no_nswf no_divergent no_reperfusion; do
  for s in 0 1 2 3 4; do
    f=logs/ablation_$r/seed_$s/training_log.csv
    [ -f "$f" ] && [ "$(wc -l <"$f")" -ge 2 ] && echo "ablation_$r seed_$s"
  done
done | wc -l

# lazy/budget trips
grep -c "lazy-agent kill-switch" "Results 2/_exp2_run.log" 2>/dev/null || echo 0
grep -c "budget kill-switch"      "Results 2/_exp2_run.log" 2>/dev/null || echo 0
```

Post immediately (do not wait for 20-min tick) when:
- Any run finishes or fails.
- `lazy-agent kill-switch` or `budget kill-switch` fires.
- `run_ablations.py` prints FAILED exit.
- Any §8 minor fix applied.
- Non-minor blocker encountered.

### STEP 10 — [VM] Final exit summary

When launcher prints `All 20 jobs complete.` or similar:

```
### EXP2 ablation grid done — <UTC HH:MM> (total wall-clock: <Hh:MMm>)

| Rung            | s0 | s1 | s2 | s3 | s4 | best_reward range |
|-----------------|----|----|----|----|----|-------------------|
| no_ecbf         |    |    |    |    |    |                   |
| no_nswf         |    |    |    |    |    |                   |
| no_divergent    |    |    |    |    |    |                   |
| no_reperfusion  |    |    |    |    |    |                   |
(cells: D=DONE, F=FAIL, L=lazy-trip, B=budget-trip)

Kill-switch events:
  lazy_agent trips: <count>  (list)
  budget trips:     <count>  (list)

Failures: <list or "none">

Total spend: Rs ~<amount>  (per-run budget: Rs 1,500; hard-cap: Rs 1,000)
```

Best-reward gather:
```bash
python - <<'PY'
import json, os
for r in ("no_ecbf","no_nswf","no_divergent","no_reperfusion"):
    for s in range(5):
        p = f"logs/ablation_{r}/seed_{s}/summary.json"
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"ablation_{r} seed_{s}: best_reward={d.get('best_reward'):.1f} "
                  f"steps={d.get('total_steps')} trip={d.get('budget_tripped')}")
        else:
            print(f"ablation_{r} seed_{s}: MISSING summary.json")
PY
```

### STEP 11 — [VM] CSV audit

```bash
python - <<'PY'
import csv, os, sys
missing, short, ok = [], [], []
for r in ("no_ecbf","no_nswf","no_divergent","no_reperfusion"):
    for s in range(5):
        p = f"Results 2/logs/ablation_{r}/seed_{s}/training_log.csv"
        if not os.path.exists(p):
            missing.append(p); continue
        rows = list(csv.DictReader(open(p)))
        if not rows:
            short.append((p, 0)); continue
        last_step = int(rows[-1].get("global_step") or 0)
        # EXP2 runs at 2M steps; require >= 1M as "at least half done"
        (short if last_step < 1_000_000 else ok).append((p, last_step))
print(f"OK      ({len(ok)}):", *ok, sep="\n  ")
print(f"SHORT   ({len(short)}):", *short, sep="\n  ")
print(f"MISSING ({len(missing)}):", *missing, sep="\n  ")
if short or missing:
    sys.exit(1)
PY
echo "audit exit=$?"
```

Pass: `audit exit=0`. Otherwise STOP.

### STEP 12 — [VM] Add provenance + configs to Results 2/

Training artefacts (CSVs + JSONs + .pt + run_state.pt) already live in
`Results 2/logs/` and `Results 2/checkpoints/` via the STEP 7 symlinks.
STEP 12 only adds metadata.

```bash
mkdir -p "Results 2/_configs_snapshot"
cp config/experiment_matrix.yaml "Results 2/_configs_snapshot/"
cp config/ablation_no_ecbf.yaml \
   config/ablation_no_nswf.yaml \
   config/ablation_no_divergent.yaml \
   config/ablation_no_reperfusion.yaml \
   config/hcmarl_full_config.yaml \
   "Results 2/_configs_snapshot/"

# Provenance
{
  echo "# Experiment 2 — provenance snapshot"
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
} > "Results 2/_provenance.txt"

# Per-(rung, seed) aggregation summary (IQM/CI computation happens LOCAL
# via scripts/aggregate_learning_curves.py; this is the raw summary).
python - <<'PY'
import csv, json
from pathlib import Path
OUT = Path("Results 2/_aggregation_summary.csv")
fields = ["rung","seed","total_steps","best_reward","final_cost_ema",
          "final_safety_rate","budget_tripped","lazy_tripped"]
rows = []
for r in ("no_ecbf","no_nswf","no_divergent","no_reperfusion"):
    for s in range(5):
        p = Path(f"Results 2/logs/ablation_{r}/seed_{s}")
        summary = p / "summary.json"
        row = {"rung": r, "seed": s}
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

# Index
python - <<'PY'
from pathlib import Path
OUT = Path("Results 2/_INDEX.txt")
lines = ["# Results 2/ — contents index", ""]
for p in sorted(Path("Results 2").rglob("*")):
    if p.is_file():
        try:
            lines.append(f"  {p.relative_to('Results 2')}  ({p.stat().st_size} bytes)")
        except ValueError:
            pass
OUT.write_text("\n".join(lines))
print(f"Index written: {OUT}")
PY

echo "=== Results 2/ tree ==="
ls -la "Results 2/"
du -sh "Results 2/"
```

### STEP 13 — [VM] Stand down

```
EXP2 verified. Results 2/ assembled. Standing by for §0.2 close-out paste.
```

Wait. Do not launch anything else.

---

## 5. Status-report cadence

Every 20 minutes after STEP 8. Post immediately when any of the event
triggers fires (§9 equivalents in EXP1 §5).

---

## 6. Known failure modes (context for VM Claude)

### 6.1 Build-up ladder leakage
Old matrix had 5-rung build-up rungs (`mappo`, `plus_ecbf`, etc.). If
the VM git-pulled before the EXP2 push landed, `pytest` STEP 5 will fail
on D2 tests because the config test names don't match. Fix: re-pull.

### 6.2 3M anchor legacy error
Old matrix had `curve_anchors_steps: [500K, 1M, 2M, 3M]`. If a stale
copy runs with 2M-step ablations, the aggregator legitimately errors
at 3M. Fix: ensure matrix has `[500K, 1M, 2M]` only.

### 6.3 HCMARL reference contamination
EXP2 does NOT re-train HCMARL. If VM mistakenly invokes
`run_baselines.py --methods hcmarl`, it would overwrite EXP1 HCMARL
logs. Guard: STEP 8 uses `run_ablations.py`, not `run_baselines.py`.
Double-check tmux command string before Enter.

### 6.4 ECBF QP CVXPY warning
`ecbf_filter.py` occasionally emits "Solution may be inaccurate" under
high-stress states. Benign. Does NOT count as a pytest failure.

### 6.5 MMICRL MI=0 on real data (NOT a failure)
MMICRL is expected to return MI ≈ 0 on real WSD4FEDSRM data → safety
ceilings fall back to config defaults. This is the MI-collapse guard
working correctly, not a bug. All 4 EXP2 ablations should exhibit this
identically (MMICRL stays on for all 4 so each is a clean remove-one).

---

## 7. Files, paths, layout

```
/root/hcmarl_project/
├── RUNBOOK_EXP2.md                       ← THIS FILE
├── config/
│   ├── hcmarl_full_config.yaml           # reference; NOT re-run in EXP2
│   ├── ablation_no_ecbf.yaml             # total_steps=2M
│   ├── ablation_no_nswf.yaml             # total_steps=2M
│   ├── ablation_no_divergent.yaml        # total_steps=2M
│   ├── ablation_no_reperfusion.yaml      # total_steps=2M
│   └── experiment_matrix.yaml            # 4 remove-one rungs, 5 seeds
├── scripts/
│   ├── train.py
│   └── run_ablations.py                  # reads experiment_matrix.yaml
├── logs/                                 # SYMLINK -> Results 2/logs/
├── checkpoints/                          # SYMLINK -> Results 2/checkpoints/
└── Results 2/                            # ← deliverable, live from STEP 7
    ├── logs/
    │   └── ablation_{no_ecbf,no_nswf,no_divergent,no_reperfusion}/seed_{0..4}/
    │       ├── training_log.csv
    │       └── summary.json
    ├── checkpoints/
    │   └── ablation_{...}/seed_{0..4}/
    │       ├── checkpoint_<step>.pt
    │       ├── checkpoint_best.pt
    │       ├── checkpoint_final.pt
    │       └── run_state.pt
    ├── _configs_snapshot/
    ├── _provenance.txt
    ├── _aggregation_summary.csv
    ├── _exp2_run.log
    └── _INDEX.txt
```

Expected size: ~1.5 GB (20 runs × ~70 MB each: CSVs ~1 MB, checkpoints
~70 MB).

---

## 8. Minor-blocker authority

Identical to EXP1 §8. Do NOT modify hcmarl/ source, Phase A constants,
hyperparameters, seeds, total_steps, ECBF/NSWF/MMICRL toggles,
`muscle_groups`, `tasks`, `theta_max`, or kill-switches.

MAY fix: missing pip package, stale `.pyc`, CUDA visibility, tmux
recovery (only if CSVs intact).

---

## 9. Emergency procedures

Identical to EXP1 §9.

---

## 10. Results format (for the record)

Every file under `Results 2/` is:
- **Human-readable** — CSVs open in any spreadsheet; JSONs pretty-print.
- **Python-analyzable** — `training_log.csv` has labelled columns per
  eval episode: `(episode, global_step, cumulative_reward, cost,
  safety_rate, peak_MF, per_agent_entropy_mean, per_agent_entropy_min,
  lazy_agent_flag, ...)` — ready for pandas/matplotlib/seaborn.
- **Claude-interpretable** — plain text + structured data.
- **IQM/CI-ready** — every CSV has `global_step` and `cumulative_reward`
  columns. The LOCAL command after pull:
  ```bash
  python scripts/aggregate_learning_curves.py \
      --matrix config/experiment_matrix.yaml \
      --logs-root "Results 2/logs" \
      --out "Results 2/aggregated_results.json"
  ```
  produces IQM + stratified bootstrap 95% CI at each anchor per rung.
  **Runs on the laptop** in ~30 seconds (CPU-only; do NOT run on VM —
  that wastes L4-hours).

No visualization on VM. That is EXP4 (LOCAL).

---

## 11. End-of-session pull commands (USER runs LOCAL after §0.2)

Mirrors EXP1 §11 structure. All three pulls land in `/c/Users/admin/Downloads/`.
Replace `<public-ip>` with the E2E dashboard IP.

```bash
# PRIMARY — single pull gets everything (logs + checkpoints + metadata, ~1.5 GB)
rsync -avzP -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/"Results 2" \
    /c/Users/admin/Downloads/

# BELT-AND-BRACES #1 — independent mirror of logs/ (CSVs + JSONs only, ~20 MB)
rsync -avzP -L -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/logs/ \
    /c/Users/admin/Downloads/logs_exp2_mirror/

# BELT-AND-BRACES #2 — independent mirror of checkpoints/ (.pt weights, ~1.4 GB)
rsync -avzP -L -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/checkpoints/ \
    /c/Users/admin/Downloads/checkpoints_exp2_mirror/
```

Rationale on `-L`:
- Primary pull does NOT need `-L` because `Results 2/logs/` and
  `Results 2/checkpoints/` on the VM are real directories (they are the
  symlink *targets*).
- Mirror pulls DO need `-L` because `/root/hcmarl_project/logs/` and
  `/checkpoints/` are the symlinks themselves; `-L` dereferences.

After all three complete, `/c/Users/admin/Downloads/` contains:
- `Results 2/` — canonical deliverable
- `logs_exp2_mirror/` — second copy of CSVs + JSONs
- `checkpoints_exp2_mirror/` — second copy of .pt weights

Verify:
```bash
du -sh /c/Users/admin/Downloads/"Results 2"/
ls /c/Users/admin/Downloads/"Results 2"/logs/
ls /c/Users/admin/Downloads/"Results 2"/checkpoints/
```

Then LOCAL can optionally compute IQM/CI:
```bash
cd /c/Users/admin/Desktop/hcmarl_project
python scripts/aggregate_learning_curves.py \
    --matrix config/experiment_matrix.yaml \
    --logs-root /c/Users/admin/Downloads/"Results 2"/logs \
    --out /c/Users/admin/Downloads/"Results 2"/aggregated_results.json
```

---

## 12. End-of-session checklist (user runs after §0.2 paste)

- [ ] VM final summary (§10 format) posted
- [ ] STEP 11 audit exit=0
- [ ] STEP 12 `Results 2/_INDEX.txt` exists with 20+ seed entries
- [ ] §11 primary rsync of `Results 2/` → Downloads/ completed
- [ ] §11 mirror #1 of logs/ → Downloads/logs_exp2_mirror/ completed
- [ ] §11 mirror #2 of checkpoints/ → Downloads/checkpoints_exp2_mirror/ completed
- [ ] `ls /c/Users/admin/Downloads/"Results 2"/logs/ablation_*/seed_*/training_log.csv`
      returns 20 non-empty paths
- [ ] §0.2 close-out sent; VM acknowledged
- [ ] E2E node destroyed in dashboard
- [ ] Billing shows node charge stopped

All ten boxes ticked = session done.

---

## 13. What EXP2 deliberately does NOT do (scope discipline)

| Item | Why deferred | To |
|---|---|---|
| Re-train HCMARL full | Already done in EXP1; first 5 seeds are the reference | EXP1 (already done) |
| `no_mmicrl` ablation | MMICRL is null-op on real data; measurement is noise | Honestly disclosed in paper |
| Synthetic K=3 MMICRL validation | CPU-side; not a VM job | EXP3 (laptop) |
| IQM/CI computation on VM | CPU-only; wasted L4-hours | LOCAL post-pull |
| Visualisation / plotting | Headless VM matplotlib is a footgun | EXP4 (laptop) |
| Paper writing | LOCAL-only | POST-EXP4 |

Keep EXP2 narrow: 20 clean ablation CSVs + weights + metadata, single
`Results 2/` deliverable, one scp pull.
