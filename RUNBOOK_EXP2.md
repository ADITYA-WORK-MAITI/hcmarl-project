# RUNBOOK_EXP2 — HC-MARL remove-one ablation grid on L4 (2026-05-03 v4 FINAL)

> **If you are Claude Code running inside the rented E2E VM terminal, this
> entire file is your briefing. Read every line top to bottom — do not skim,
> do not skip, do not summarise. The §0.1 boot prompt tells you exactly when
> to act and when to stop. Follow it.**

This is **Experiment 2** in the user's numbered experiment plan.
EXP2 is the **REMOVE-ONE ABLATION** attribution study:
**5 rungs × 10 seeds = 50 runs** on **L4 + 50 vCPU**.

**APPLES-TO-APPLES METHODOLOGY:** EXP1 (HCMARL + 4 baselines × 10 seeds)
runs on L4 / 50 vCPU. EXP2 (5 ablations × 10 seeds) runs on **identical
L4 / 50 vCPU hardware**. Same GPU architecture, same vCPU count, same
HIGH-determinism settings, same code commit, same hyperparameters. The
remove-one comparison `HCMARL (EXP1) vs no_X (EXP2)` is therefore on
matched hardware — no cross-hardware trajectory drift to defend.

**IMPORTANT:** EXP2 does **NOT** re-run full HC-MARL. The HCMARL reference
point comes from EXP1's hcmarl seeds (logs/hcmarl/seed_{0..9}/) on a
**different VM (separate E2E account #1)**, running **right now**. The
two VMs do NOT share state, do NOT share IPs, do NOT coordinate. Local
post-pull aggregation merges `Results 1/` and `Results 2/` after both
runs finish. **Do not look for EXP1 artefacts on this VM.** Do not block
on their absence. Just produce 50 clean ablation runs.

**Concurrent-execution mode (2026-05-03):** EXP1 (HCMARL + 4 baselines ×
10 seeds = 50 runs) is in progress on a separate L4 VM under E2E account
#1, expected to finish ~6:30 AM IST May 4. EXP2 runs on a separate L4 VM
under E2E account #2. Total combined spend across both VMs is bounded
by the user's separate caps per account.

---

## 0. What this runbook covers — and what it does NOT

**In scope (post-2026-05-02 lineup):**
- **Five REMOVE-ONE ablations**, each differing from full HC-MARL by exactly
  one component:

  | Rung              | Ablated component                                   |
  |-------------------|-----------------------------------------------------|
  | `no_ecbf`         | ECBF safety filter off (`ecbf.enabled=false`)       |
  | `no_nswf`         | NSWF Hungarian allocator off (`nswf.enabled=false`) |
  | `no_divergent`    | Disagreement utility constant (`type=constant`)     |
  | `no_reperfusion`  | 3CC-r reperfusion multiplier `r=1` (vs 15/30)       |
  | `no_mmicrl`       | MMICRL type-inference off (`mmicrl.enabled=false`)  |

- **Ten seeds each:** 0..9 → **50 total runs**.
- Config files: `config/ablation_no_{ecbf,nswf,divergent,reperfusion,mmicrl}.yaml`
- Launcher: `scripts/run_ablations.py` (matrix-driven, `--fresh-logs` enabled).
- **Hardware:** NVIDIA L4 / 50 vCPU / 220 GB RAM / 48 GB GPU memory /
  CUDA 12 / 250 GB SSD / IOPS R 60000 W 30000 / **₹98 per hour** (E2E
  Networks on-demand, account #2). **Identical to EXP1's L4 hardware.**
- `--max-parallel 6` (rationale §4 STEP 8). 6-way parallel on 50 vCPUs
  gives 50/6 ≈ 8.3 vCPUs/proc; auto thread-cap = 8 threads/proc. Matches
  the L4/25-vCPU 3-way calibration point exactly.
- **HIGH determinism active:** `torch.use_deterministic_algorithms(True,
  warn_only=False)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, matmul precision
  `"highest"`, cudnn.deterministic=True. Same B2 patch as EXP1.
- Logs target: `logs/ablation_<rung>/seed_{0..9}/training_log.csv`.
- Checkpoints target: `checkpoints/ablation_<rung>/seed_{0..9}/*.pt`.
- **Final deliverable:** top-level `Results 2/` containing every CSV,
  summary JSON, MMICRL result, checkpoint, configs snapshot, provenance.
  Pulled to laptop via tar-piped-ssh per user gitbash workflow (NOT
  rsync, NOT scp).

**Out of scope (deferred):**
- HCMARL retrain / EXP1 redo — HCMARL full runs are EXP1 only.
  Comparison reference = EXP1's hcmarl seeds on the OTHER (account #1) VM.
- Headline 6-method baseline grid (EXP1's job).
- Synthetic K=3 experiment (EXP3 Part B, separate VM after EXP1+EXP2).
- IQM + bootstrap CI computation — LOCAL post-pull (aggregator is
  CPU-only and takes ~30 s; do NOT run on VM, that wastes L4-hours).
- Visualisation / analysis / interpretation (EXP4, LOCAL only).
- Any edit to `hcmarl/` source or Phase A constants (3CC-r / ECBF /
  NSWF / MMICRL / POPULATION_FR).

---

## 0.1 Boot prompt (paste into VM-side Claude Code immediately after `claude` launches)

```
You are the VM-side Claude for the HC-MARL Experiment 2 REMOVE-ONE
ablation run on an E2E L4 GPU node, account #2 (2026-05-03). This
paste IS your full starting instruction.

CONCURRENT-EXECUTION NOTICE: EXP1 is running RIGHT NOW on a different
L4 node (separate E2E account #1, separate IP, separate VM). That EXP1
VM produces `Results 1/`. You will NOT see `Results 1/` on THIS VM
and you should NOT look for it. Do NOT block on its absence. Do NOT
attempt to scp from the other VM. The HCMARL reference merges with
your ablation outputs LOCALLY on the user's laptop after both VMs
finish (LOCAL EXP4 step). On THIS VM, your sole job is producing 50
clean ablation CSVs in `Results 2/`.

Your complete briefing is at /root/hcmarl_project/RUNBOOK_EXP2.md on
this VM. Read every line top to bottom. Do not skim, do not skip, do
not summarise.

CRITICAL NON-NEGOTIABLE PRE-FLIGHT (if any of these is not true, STOP):
  1. You cloned the MOST RECENT push from
     github.com/ADITYA-WORK-MAITI/hcmarl-project (branch master).
     Recent commits should include the EXP2 RUNBOOK v4 commit (L4
     final) plus 6518bd8, b27f091, 48f5b04, 4ee2ef3, 45baa8b, 235f63b,
     e883024.
  2. The 5 ablation configs all exist on disk:
     `ls config/ablation_no_{ecbf,nswf,divergent,reperfusion,mmicrl}.yaml`
     -- must list 5 files, no errors.
  3. experiment_matrix.yaml has exactly 5 ablation rungs and 10 seeds:
     `python -c "import yaml; m=yaml.safe_load(open('config/experiment_matrix.yaml'))['ablation']; \
       print('rungs:', [r['name'] for r in m['rungs']]); print('seeds:', m['seeds'])"`
     Expected:
       rungs: ['no_ecbf', 'no_nswf', 'no_divergent', 'no_reperfusion', 'no_mmicrl']
       seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  4. All 5 ablation configs declare total_steps=2000000:
     `grep -H 'total_steps:' config/ablation_no_{ecbf,nswf,divergent,reperfusion,mmicrl}.yaml`
     -- every line must show `total_steps: 2000000`.
  5. mi_collapse_threshold=0.01 in the 4 ablations that keep MMICRL on:
     `grep -H 'mi_collapse_threshold:' config/ablation_no_{ecbf,nswf,divergent,reperfusion}.yaml`
     -- every line must show `mi_collapse_threshold: 0.01`.
     (no_mmicrl has no threshold because MMICRL is disabled entirely.)
  6. Each rung flips EXACTLY ONE axis vs full HC-MARL. The single-axis-
     flip invariant is enforced by the parametrized test (now covering
     all 5 rungs including no_mmicrl):
     `pytest tests/test_batch_d.py -q --no-header 2>&1 | tail -3`
     -- must show `0 failed`, expect `24 passed`.
  7. logs/ablation_* and checkpoints/ablation_* are EMPTY on the VM
     (either fresh clone or will be wiped by --fresh-logs).
  8. HIGH determinism is active inside seed_everything (B2 patch):
     `grep -n "use_deterministic_algorithms\|CUBLAS_WORKSPACE_CONFIG\|matmul_precision" hcmarl/utils.py`
     -- must show all three: torch.use_deterministic_algorithms(True),
     CUBLAS_WORKSPACE_CONFIG=":4096:8", set_float32_matmul_precision("highest").
  9. run_ablations.py supports --fresh-logs and reads experiment_matrix.yaml:
     `grep -c "fresh-logs\|experiment_matrix" scripts/run_ablations.py`
     -- must print >= 2.
 10. Dry-run banner verification:
     `python scripts/run_ablations.py --dry-run 2>&1 | head -5`
     FIRST LINE must be exactly:
     `Ablation ladder: 5 rungs x 10 seeds = 50 runs`
     SECOND LINE must list all 5 rungs:
     `Rungs: ['no_ecbf', 'no_nswf', 'no_divergent', 'no_reperfusion', 'no_mmicrl']`
     If either deviates, STOP.

If ANY of the 10 checks above fails, STOP, post the failing check, wait.

After reading, execute STEPs 1-13 of RUNBOOK_EXP2.md continuously, in
order, without asking for confirmation between steps:

  STEP 1   hardware sanity (expect L4, 24+ GB VRAM, 50 vCPUs)
  STEP 2   python3.12 venv
  STEP 3   torch from cu124 (fall back to cu121)
  STEP 4   pip install -r requirements.txt
  STEP 5   pytest -q  (pass: 0 failed; expect 590 passed, 2 skipped)
  STEP 6   the TEN pre-flight checks above
  STEP 7   symlink logs/ + checkpoints/ into Results 2/
  STEP 7.5 EXP0 50K GPU probe (no_ecbf rung) -- catches HIGH determinism
           CUDA-op crashes BEFORE burning the full grid. 10-15 min on L4.
  STEP 8   create tmux session `exp2`; launch the 5-rung 10-seed grid
           with --max-parallel 6
  STEP 9   status report every 20 minutes (format in §5)
  STEP 10  when grid finishes, post the exit summary (§6)
  STEP 11  CSV audit must pass before standing down
  STEP 12  add provenance + aggregation to Results 2/ (CPU-only, do NOT
           compute IQM/CI on VM -- aggregation_summary.csv is a raw
           dump, full IQM/CI runs LOCAL post-pull)
  STEP 13  STOP. Wait for the §0.2 close-out paste.

Minor-blocker policy (§8): fix trivial blockers in place (missing pip
pkg, stale .pyc, tmux recovery) and report. Do NOT modify hyperparameters,
seeds, total_steps, ECBF states, env parameters, OR anything under
hcmarl/ source. If the fix exceeds that, STOP and escalate.

DETERMINISM EMERGENCY POLICY: If STEP 7.5 fails with
`RuntimeError: ... does not have a deterministic implementation`, you
have authority to:
  (a) Identify the failing op from the traceback and report it.
  (b) Patch hcmarl/utils.py to flip warn_only=False -> warn_only=True
      (this turns the hard error into a printed warning; the run still
      executes, but bit-identity across two re-runs of the same seed is
      no longer guaranteed -- only within-run determinism is preserved).
  (c) Document the patch in your status report and continue.
This is the ONLY source-edit you are pre-authorised to make.

GIT PUSH POLICY: This VM has NO GitHub credentials (same as EXP1 VM).
If you make any commits during the run (e.g., RUNBOOK amendment, config
fix), `git push` will fail with `fatal: could not read Username`. That
is EXPECTED. Save patches via `git format-patch` into Results 2/ instead.
Do NOT attempt to set up GitHub credentials or push.

Begin now. After reading RUNBOOK_EXP2.md, reply with exactly:
`Read. Ten pre-flight checks at STEP 6. Results land in 'Results 2/'.`
and start.
```

---

## 0.2 Close-out prompt (paste after CSV audit passes and Results 2/ is populated)

```
Experiment 2 complete. Your work is done for this session.

The user will now tar-pipe-ssh Results 2/ to the laptop and destroy
the E2E node manually. Those are [USER] steps.

Do NOT destroy the node. Do NOT delete anything. Do NOT git push.
Do NOT make tar files (the user pulls via ssh-piped tar from local
gitbash). Do NOT start more training.

Reply with exactly
  `EXP2 complete. Standing by. Local will close the session.`
and then wait.
```

---

## 1. Three-agent division of labour

| Agent | Owns | Forbidden |
|---|---|---|
| **LOCAL** (laptop) | scope decisions, git state, RUNBOOK edits, tar+ssh pull, node destruction | direct SSH execution on VM |
| **VM** (Claude Code on L4 account #2) | bootstrap, 10 pre-flight checks, EXP0 probe, tmux launch, 20-min status reports, minor §8 fixes, CSV audit, `Results 2/` assembly | scope decisions, editing anything under hcmarl/ source (except the determinism warn_only flip per §0.1 emergency policy), git commits beyond format-patch, destroying the node, running anything after STEP 13 |
| **USER** | E2E account #2 dashboard, SSH into account #2 VM, git clone before claude launches, paste between sessions, tar+ssh pull `Results 2/`, destroy node | running training outside tmux |

---

## 2. Project state at the start of this session

- **Five ablation configs** committed to GitHub at HEAD:
  `config/ablation_no_{ecbf,nswf,divergent,reperfusion,mmicrl}.yaml`,
  all with `total_steps: 2_000_000`, all using `method: hcmarl`
  (remove-one semantics).
- **experiment_matrix.yaml** ablation section has 5 rungs × 10 seeds = 50 runs.
- **HIGH determinism active** (B2 patch 2026-05-02) in
  `hcmarl/utils.py::seed_everything`. Same setup that ran EXP3 Part 1
  successfully (ARI=1.0 on synthetic K=3) and EXP1 (in progress).
- **`run_ablations.py`** is matrix-driven and has `--fresh-logs`.
- **Phase A constants intact** — PDF-verified values in `three_cc_r.py`,
  `ecbf_filter.py`, `nswf_allocator.py`, `mmicrl.py`,
  `real_data_calibration.py`. SACRED — not touched by any commit since
  the verification pass.
- **Test suite green:** 590 passed, 2 skipped, 0 failed (no_mmicrl test
  case added 2026-05-03 to close coverage gap).
- **EXP1 status (separate L4 VM, account #1, concurrent):** running,
  ETA ~6:30 AM IST May 4. Does NOT block EXP2. Does NOT share files
  with EXP2. Local laptop merges results post-pull.

**Budget reality (L4 on-demand, ₹98/hr):**

Per-rung wall-time projection per seed (calibrated from EXP1's actual
L4 measurement of 215-228 SPS for HCMARL on the same hardware):

| Rung           | Per-seed cost vs HCMARL          | Per-seed wall-time | Per-rung SPS estimate |
|----------------|----------------------------------|--------------------|------------------------|
| `no_ecbf`      | ~50% (no QP per step)            | ~1.3 h             | ~400 SPS               |
| `no_nswf`      | ~95% (Hungarian is cheap)        | ~2.5 h             | ~225 SPS               |
| `no_divergent` | ~95% (utility-fn change only)    | ~2.5 h             | ~225 SPS               |
| `no_reperfusion`| ~100% (constant change in ODE)  | ~2.6 h             | ~215 SPS               |
| `no_mmicrl`    | ~95% (skips ~52s MMICRL pretrain)| ~2.4 h             | ~235 SPS               |

- **Total worker-hours:** 10×1.3 + 10×2.5 + 10×2.5 + 10×2.6 + 10×2.4 = ~113 worker-hours.
- **6-way parallel:** 113 / 6 ≈ **19 hours wall-clock**.
- **Realistic with overhead + thermal throttling:** 18-22 hours.
- **Expected spend:** ~₹1,800-2,200 + pre-grid overhead ~₹200-300.
  Total **~₹2,000-2,500**.

**Hard total stop:** ₹2,500 (gives ~₹0-500 buffer over the projected
upper bound; same prudent-margin pattern as EXP1's revised cap). If the
E2E dashboard or in-VM `--budget-margin` math shows total spend
approaching ₹2,500, STOP and escalate.

**SPS realism note:** Earlier runbook drafts referenced "500+ SPS"
targets. **That is not realistic on this L4 hardware for HCMARL-class
methods.** EXP1 measured 215-228 SPS for HCMARL (full stack with ECBF
QP per step). `no_ecbf` will be faster (~350-450 SPS), the rest will
match HCMARL pace (~200-250 SPS). Do NOT promise 500+. The grid-time
math above uses the measured numbers, not aspirational ones.

---

## 3. Pre-flight on the laptop (already done by LOCAL — reported here for VM to verify)

LOCAL confirmed before pushing:
1. **All 5 ablation configs present** with populated `environment:` block
   (muscle_groups + theta_max + tasks all PDF-verified Frey-Law 2012).
2. **ECBF / NSWF / MMICRL flags correct per ablation:**
   - `no_ecbf`: `ecbf.enabled=false`, others on.
   - `no_nswf`: `nswf.enabled=false`, others on.
   - `no_divergent`: `disagreement.type=constant`, others normal.
   - `no_reperfusion`: `muscle_groups.*.r=1`, others 15/30.
   - `no_mmicrl`: `mmicrl.enabled=false`, others on.
3. **`config/experiment_matrix.yaml`** ablation section: 5 rungs × 10 seeds.
4. **HIGH determinism active** in `seed_everything`.
5. **`scripts/run_ablations.py`** has `--fresh-logs`.
6. **`tests/test_batch_d.py`** single-axis-flip invariant passes for all 5 rungs.
   Test count: 24 passed (was 23; no_mmicrl test case added 2026-05-03).
7. **`pytest -q`** -- 590 passed, 2 skipped, 0 failed (laptop, 2026-05-03).
8. **Phase A constants** -- untouched since the PDF-verification pass.
9. **`mi_collapse_threshold = 0.01`** verified in 4 of 5 configs (no_mmicrl
   doesn't have it because MMICRL is disabled).
10. **`config/pathg_profiles.json`** tracked, current version (Apr 26),
    34 profiles matching WSD4FEDSRM.
11. **`git push origin master`** complete; tip is the EXP2 RUNBOOK v4 commit.

---

## 4. Execution plan (VM owns STEPs 1-13)

### STEP 0 — [USER] boot the VM + clone the LATEST repo

Manual, before Claude Code launches (per `exp 1 gitbash.txt` workflow).
**Use the SSH key for E2E account #2** — verify on the E2E dashboard.

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@<account2-public-ip>

# Bootstrap
apt-get update -q
apt-get install -y -q curl ca-certificates git tmux htop python3.12-venv python3-pip nodejs npm
apt-get install -y -q nvtop || true

cd /root
if [ ! -d "hcmarl_project" ]; then
  git clone https://github.com/ADITYA-WORK-MAITI/hcmarl-project.git hcmarl_project
fi
cd hcmarl_project
git pull origin master
git log --oneline -5

# Confirm RUNBOOK_EXP2.md (NOT EXP1) and recent commit
ls RUNBOOK_EXP2.md && git log -1 --format='%h %s'

# Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version

# Launch VM Claude inside tmux
tmux new -s run
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Paste the **§0.1 boot prompt** verbatim into the VM Claude.

### STEP 1 — [VM] Hardware sanity

```bash
nvidia-smi | head -25
lscpu | grep -E "Model name|Thread|Socket|CPU max MHz|CPU\(s\):"
free -h
df -h /root
```

Pass criteria:
- **GPU**: NVIDIA L4 (matches EXP1 hardware). VRAM total: report what
  `nvidia-smi` prints (E2E listing says 48 GB; if the actual card shows
  24 GB, that is still acceptable — pre-flight memory budget §2 fits in
  24 GB). VRAM free at startup: ≥ 20 GiB.
- **vCPUs**: ≥ 48 (target is 50)
- **RAM free**: ≥ 100 GiB (we'll only use ~20 GiB peak)
- **Disk free**: ≥ 200 GiB

If GPU is missing, NOT L4 (e.g., L40S substitution), or VRAM < 16 GiB,
STOP and escalate. The methodology requires identical hardware to EXP1.

### STEP 2 — [VM] Create the venv

```bash
cd /root/hcmarl_project
python3.12 -m venv venv
source venv/bin/activate
python --version           # 3.12.x
pip install -q -U pip wheel
```

### STEP 3 — [VM] Install torch from cu124 wheel index

```bash
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"
```

Fallback to cu121 if cu124 reports `cuda: False`:

```bash
pip uninstall -y torch
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

If both fail, STOP.

### STEP 4 — [VM] Install rest of requirements

```bash
pip install -q -r requirements.txt
pip list | grep -iE "^(torch|gymnasium|numpy|cvxpy|osqp|scipy|pyyaml|pypdf|pytest)"
```

Torch version must include `+cu124` or `+cu121`. If pip replaced it with
the CPU wheel, re-run STEP 3 with `--force-reinstall`.

### STEP 5 — [VM] Pytest sanity

```bash
pytest -q 2>&1 | tail -5
```

Expected: **590 passed, 2 skipped, 0 failed** (the 2 skips are
env-conditional). 4 additional skips (588 + 4 skipped) are also
acceptable if WSD4FEDSRM raw-data tests skip on a fresh VM clone (those
tests need data/wsd4fedsrm/ which isn't deployed). Any FAILURE: STOP.

### STEP 6 — [VM] The TEN non-negotiable pre-flight checks

```bash
echo "=== CHECK 1: git tip ==="
git log -1 --format='%h %ci %s'
echo

echo "=== CHECK 2: 5 ablation configs exist ==="
ls config/ablation_no_{ecbf,nswf,divergent,reperfusion,mmicrl}.yaml
echo

echo "=== CHECK 3: matrix has 5 rungs, 10 seeds ==="
python - <<'PY'
import yaml
m = yaml.safe_load(open("config/experiment_matrix.yaml"))["ablation"]
names = [r["name"] for r in m["rungs"]]
expected = ["no_ecbf","no_nswf","no_divergent","no_reperfusion","no_mmicrl"]
assert set(names) == set(expected), f"rung names wrong: {names}"
assert m["seeds"] == [0,1,2,3,4,5,6,7,8,9], f"seeds wrong: {m['seeds']}"
assert all(r["method"] == "hcmarl" for r in m["rungs"]), \
    "all rungs must use method=hcmarl for remove-one semantics"
print(f"  rungs: {sorted(names)}  OK")
print(f"  seeds: {m['seeds']}  OK")
PY
echo

echo "=== CHECK 4: all 5 configs at total_steps=2000000 ==="
grep -H 'total_steps:' config/ablation_no_{ecbf,nswf,divergent,reperfusion,mmicrl}.yaml
python - <<'PY'
import yaml
for name in ("no_ecbf","no_nswf","no_divergent","no_reperfusion","no_mmicrl"):
    c = yaml.safe_load(open(f"config/ablation_{name}.yaml"))
    steps = c["training"]["total_steps"]
    assert steps == 2_000_000, f"{name}: total_steps={steps}"
    print(f"  ablation_{name}.yaml: total_steps=2,000,000  OK")
PY
echo

echo "=== CHECK 5: mi_collapse_threshold=0.01 in 4 MMICRL-on rungs ==="
grep -H 'mi_collapse_threshold:' config/ablation_no_{ecbf,nswf,divergent,reperfusion}.yaml
python - <<'PY'
import yaml
for name in ("no_ecbf","no_nswf","no_divergent","no_reperfusion"):
    c = yaml.safe_load(open(f"config/ablation_{name}.yaml"))
    thresh = c.get("mmicrl",{}).get("mi_collapse_threshold")
    assert thresh == 0.01, f"{name}: mi_collapse_threshold={thresh}, expected 0.01"
    print(f"  ablation_{name}.yaml: mi_collapse_threshold=0.01  OK")
# no_mmicrl has no threshold; expected, not an error
print("  ablation_no_mmicrl.yaml: threshold N/A (MMICRL disabled)  OK")
PY
echo

echo "=== CHECK 6: single-axis-flip invariant (pytest D2) ==="
pytest tests/test_batch_d.py -q --no-header 2>&1 | tail -3
echo

echo "=== CHECK 7: logs + checkpoints clean for all 5 rungs ==="
for r in no_ecbf no_nswf no_divergent no_reperfusion no_mmicrl; do
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

echo "=== CHECK 8: HIGH determinism active in seed_everything ==="
grep -n "use_deterministic_algorithms\|CUBLAS_WORKSPACE_CONFIG\|matmul_precision" hcmarl/utils.py | head -10
python - <<'PY'
from hcmarl.utils import seed_everything
import inspect
src = inspect.getsource(seed_everything)
assert "use_deterministic_algorithms(True" in src, "MISSING: use_deterministic_algorithms(True"
assert "CUBLAS_WORKSPACE_CONFIG" in src, "MISSING: CUBLAS_WORKSPACE_CONFIG"
assert 'matmul_precision("highest")' in src, "MISSING: matmul_precision(highest)"
print("  HIGH determinism: all 3 mechanisms present in seed_everything OK")
PY
echo

echo "=== CHECK 9: run_ablations.py has --fresh-logs and is matrix-driven ==="
n=$(grep -c "fresh-logs\|experiment_matrix" scripts/run_ablations.py)
[ "$n" -ge 2 ] && echo "  OK ($n matches)" || (echo "  FAIL ($n matches)"; exit 1)
echo

echo "=== CHECK 10: dry-run banner ==="
python scripts/run_ablations.py --dry-run 2>&1 | head -10
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    [sys.executable, "scripts/run_ablations.py", "--dry-run"],
    text=True).splitlines()
joined = " ".join(out).lower()
for r in ("no_ecbf","no_nswf","no_divergent","no_reperfusion","no_mmicrl"):
    assert r in joined, f"dry-run missing rung name {r}:\n{out}"
assert "5 rungs" in joined and "10 seeds" in joined and "50 runs" in joined, \
    f"dry-run missing count signal:\n{out}"
print("  dry-run banner OK (all 5 rungs listed; 5/10/50 signal present)")
PY
echo

echo "All ten checks complete."
```

Expected outcome on fresh clone:
- CHECK 1 — recent commit hash includes the EXP2 RUNBOOK v4 commit.
- CHECK 2 — 5 yaml file paths printed, no errors.
- CHECK 3 — rung set matches expected; seeds=[0..9].
- CHECK 4 — five `total_steps: 2000000` lines all OK.
- CHECK 5 — four `mi_collapse_threshold: 0.01` lines + no_mmicrl N/A.
- CHECK 6 — `0 failed`, `24 passed` from D2 tests.
- CHECK 7 — ten `clean:` lines.
- CHECK 8 — HIGH determinism mechanisms present.
- CHECK 9 — `--fresh-logs` and `experiment_matrix` references present.
- CHECK 10 — banner contains all 5 rung names AND 5-rung/10-seed/50-run signal.

**If any check fails, STOP. Post the full output. Do not proceed.**

### STEP 7 — [VM] Symlink `logs/` and `checkpoints/` into `Results 2/`

```bash
# Defensive cleanup in case logs/ or checkpoints/ exist as real dirs:
rm -rf logs checkpoints 2>/dev/null || true

# Create the deliverable tree, then point logs/ AND checkpoints/ at it.
mkdir -p "Results 2/logs" "Results 2/checkpoints"
ln -sfn "$(pwd)/Results 2/logs" logs
ln -sfn "$(pwd)/Results 2/checkpoints" checkpoints

ls -la logs checkpoints
test -L logs && test -L checkpoints \
  || { echo "ERROR: logs or checkpoints is not a symlink"; exit 1; }
```

After this step, `ls -la logs checkpoints` MUST print two symlink lines.
If either is a real directory, STOP and retry the `ln -sfn` line.

### STEP 7.5 — [VM] EXP0 50K GPU probe (CRITICAL — pre-resolve determinism)

This is the moment HIGH determinism either works or doesn't on this
specific L4. `torch.use_deterministic_algorithms(True, warn_only=False)`
raises immediately on the first non-deterministic CUDA op. If the full
grid launches and a method hits a non-deterministic op 70 minutes in,
that's wasted GPU money. Do the smallest possible probe FIRST.

Use `no_ecbf` as the probe — it's the fastest rung (no QP per step) and
exercises HCMARL+MMICRL+NSWF+divergent paths. If `no_ecbf`'s 50K probe
runs clean on L4, the other 4 rungs (which only differ in single
components) will too.

**Important:** `train.py` does NOT have a `--max-steps` CLI flag.
`total_steps` is read from the config only. To probe at 50K instead of
2M, write a temporary probe config that copies `ablation_no_ecbf.yaml`
but overrides `training.total_steps: 50000`:

```bash
mkdir -p "Results 2/_exp0_gpu_probe"

# Write a temporary probe config with total_steps=50000 (everything else
# inherits from ablation_no_ecbf.yaml so the determinism probe exercises
# the full HCMARL+MMICRL+NSWF+divergent stack minus ECBF).
python - <<'PY'
import yaml, os
src = "config/ablation_no_ecbf.yaml"
dst = "Results 2/_exp0_gpu_probe/probe_config.yaml"
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["training"]["total_steps"] = 50_000
cfg["training"]["checkpoint_interval"] = 25_000   # save 1-2 ckpts during probe
cfg["training"]["eval_interval"] = 25_000
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Wrote probe config: {dst}")
print(f"  total_steps:      {cfg['training']['total_steps']:,}")
print(f"  checkpoint_interval: {cfg['training']['checkpoint_interval']:,}")
PY

# Now run the probe with the override config
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONUNBUFFERED=1 timeout 900 python -u scripts/train.py \
    --config "Results 2/_exp0_gpu_probe/probe_config.yaml" \
    --method hcmarl \
    --seed 0 \
    --device cuda \
    --run-name _exp0_gpu_probe_no_ecbf \
    2>&1 | tee "Results 2/_exp0_gpu_probe/no_ecbf_50k.log"
echo "no_ecbf probe exit=$?"
```

**Pass criterion:** exit code 0; log tail shows training progress (no
`RuntimeError: ... does not have a deterministic implementation`);
"Training complete: NN episodes, ~50,000 steps" line appears; SPS
reported in log ≥ 200 (conservative; expect 350-450 for `no_ecbf` on L4).

**If a deterministic-op error fires:** apply the §0.1 emergency policy
(flip warn_only False→True), document, re-probe.

**If probe SPS < 150:** STOP and escalate. That suggests CPU-bound
rollout or thread-cap misconfiguration; the L4 budget math assumes
≥200 SPS for HCMARL-family methods.

After probe passes, clean up to avoid contaminating the headline grid:

```bash
rm -rf "Results 2/logs/_exp0_gpu_probe_"* \
       "Results 2/checkpoints/_exp0_gpu_probe_"*
# Keep "Results 2/_exp0_gpu_probe/probe_config.yaml" and the log file
# as provenance (they're small, ~few KB total).
```

### STEP 8 — [VM] Create tmux session + launch the grid

```bash
# Tmux session (separate from the one Claude itself is in)
tmux kill-session -t exp2 2>/dev/null || true
pkill -f "scripts/train.py" 2>/dev/null || true
pkill -f "run_ablations.py" 2>/dev/null || true
sleep 1
tmux new -d -s exp2
tmux send-keys -t exp2 "cd /root/hcmarl_project && source venv/bin/activate" Enter
sleep 1
tmux list-sessions | grep exp2

# Launch the grid (single-line command for tmux send-keys safety;
# tmux multi-line backslash continuation is risky on some shells, so
# we use one continuous line here matching EXP1's launch pattern).
tmux send-keys -t exp2 "python scripts/run_ablations.py --device cuda --fresh-logs --max-parallel 6 --budget-inr 1500 --cost-per-hour 98.0 --budget-margin 0.95 2>&1 | tee 'Results 2/_exp2_run.log'" Enter
```

Within 30 seconds, `tmux capture-pane -t exp2 -p | tail -20` must show:
```
Ablation ladder: 5 rungs x 10 seeds = 50 runs
Rungs: ['no_ecbf', 'no_nswf', 'no_divergent', 'no_reperfusion', 'no_mmicrl']
Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Thread cap per child: OMP/MKL/OPENBLAS_NUM_THREADS=8 (max_parallel=6)
```
(verified against `run_ablations.py:142-208` banner output).

If the banner mentions **4 rungs, 5 seeds, 20 runs**, or is missing
`no_mmicrl`, immediately Ctrl-C and STOP.

**Flags explained — do not improvise:**
- `--device cuda` — L4.
- `--fresh-logs` — **NON-NEGOTIABLE**. Wipes `logs/ablation_<r>/` +
  `checkpoints/ablation_<r>/` for each rung before launching. Physical
  prevention against append-on-matching-header CSV contamination.
  Without this flag, STOP.
- `--max-parallel 6` — 6 concurrent seeds. Rationale:
  - L4 calibration: 25 vCPUs / 3 procs = 8.3 vCPUs/proc validated.
  - L4 50 vCPUs / 6 procs = 8.3 vCPUs/proc — matches calibration exactly.
  - 7-way (50/7 = 7.1) would tighten per-proc thread count below the
    validated baseline → BLAS thread thrash risk.
  - Auto thread-cap: `50 / 6 = 8` BLAS threads per process (matches L4
    calibration's 8.3 vCPU/proc thread allocation).
  - 48 GB VRAM (or 24 GB shown) / ~3 GB peak per process = ~8-16-way
    GPU memory headroom. Not GPU-memory-constrained; CPU thread
    oversubscription is the risk.
- `--budget-inr 1500` — per-run kill-switch (~15.3 hours per run @
  ₹98/hr; way over expected ~2.5h).
- `--cost-per-hour 98.0` — L4 on-demand at E2E account #2.
- `--budget-margin 0.95` — kill at 95% of the per-run budget.
- `tee Results 2/_exp2_run.log` — launcher stdout mirror inside the
  deliverable.

Do NOT add `--resume` — clean-slate grid.
Do NOT alter rung list or seeds — both are locked in `experiment_matrix.yaml`.

### STEP 9 — [VM] Automated status reports (every 20 minutes)

```
### Status report — <UTC HH:MM> (elapsed: <Hh:MMm> since STEP 8 kickoff)

Runs done / in-flight / pending:   <done>/<inflight>/<pending>  (of 50)
Per-rung SPS (rolling 50 ep):
  no_ecbf:        <sps>   (expected ~400 on L4)
  no_nswf:        <sps>   (expected ~225 on L4)
  no_divergent:   <sps>   (expected ~225 on L4)
  no_reperfusion: <sps>   (expected ~215 on L4)
  no_mmicrl:      <sps>   (expected ~235 on L4)
ETA to grid completion:            ~<Hh:MMm>
Wall-clock spend so far:           Rs ~<amount>  (@ Rs 98/hr)
lazy_agent trips since start:      <count>
budget_tripped events:             <count>
GPU mem in use (nvidia-smi):       <MiB> / 23-48 GiB total
CPU load (uptime):                 <load>
pytest state:                      green (STEP 5)

Last 5 lines of Results 2/_exp2_run.log:
  <paste>

Notes / minor fixes applied:       <describe or "none">
```

Field-gathering commands:
```bash
# n_done -- count seed dirs with >= 2 lines in training_log.csv
for r in no_ecbf no_nswf no_divergent no_reperfusion no_mmicrl; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    f=logs/ablation_$r/seed_$s/training_log.csv
    [ -f "$f" ] && [ "$(wc -l <"$f")" -ge 2 ] && echo "ablation_$r seed_$s"
  done
done | wc -l

# Per-rung SPS (rolling, last 50 episodes if available)
python - <<'PY'
import csv, glob, os
for r in ("no_ecbf","no_nswf","no_divergent","no_reperfusion","no_mmicrl"):
    sps_vals = []
    for s in range(10):
        f = f"logs/ablation_{r}/seed_{s}/training_log.csv"
        if not os.path.exists(f): continue
        with open(f) as h:
            rows = list(csv.DictReader(h))
        if len(rows) < 5: continue
        last = rows[-50:] if len(rows) > 50 else rows
        try:
            steps = int(last[-1]["global_step"]) - int(last[0]["global_step"])
            t = float(last[-1].get("wall_time", 0)) - float(last[0].get("wall_time", 0))
            if t > 0: sps_vals.append(steps / t)
        except (KeyError, ValueError):
            pass
    if sps_vals:
        print(f"  ablation_{r:18s} mean SPS = {sum(sps_vals)/len(sps_vals):.0f} (n={len(sps_vals)} seeds)")
    else:
        print(f"  ablation_{r:18s} no SPS data yet")
PY

# lazy/budget trips
grep -c "lazy-agent kill-switch" "Results 2/_exp2_run.log" 2>/dev/null || echo 0
grep -c "budget kill-switch"     "Results 2/_exp2_run.log" 2>/dev/null || echo 0

# GPU memory + utilization
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Additionally, post *immediately* (do not wait for 20-min tick) when:
- A run finishes or fails.
- A `lazy-agent kill-switch` or `budget kill-switch` fires.
- `run_ablations.py` prints `FAILED (exit code X)`.
- You apply a §8 minor fix.
- nvidia-smi shows GPU memory > 40 GB (potential OOM warning).
- Total elapsed exceeds 24 hours (was projected 18-22h).
- Total spend approaches ₹2,200 (90% of ₹2,500 hard cap).

### STEP 10 — [VM] Final exit summary

When the launcher prints `All 50 jobs complete.`, post:

```
### EXP2 ablation grid done — <UTC HH:MM> (total wall-clock: <Hh:MMm>)

| Rung           | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | best_reward range |
|----------------|----|----|----|----|----|----|----|----|----|----|-------------------|
| no_ecbf        |    |    |    |    |    |    |    |    |    |    |                   |
| no_nswf        |    |    |    |    |    |    |    |    |    |    |                   |
| no_divergent   |    |    |    |    |    |    |    |    |    |    |                   |
| no_reperfusion |    |    |    |    |    |    |    |    |    |    |                   |
| no_mmicrl      |    |    |    |    |    |    |    |    |    |    |                   |
(cells: D=DONE, F=FAIL, L=lazy-trip, B=budget-trip)

Kill-switch events:
  lazy_agent trips: <count>  (list)
  budget trips:     <count>  (list)

Failures: <list or "none">

Per-rung SPS final:
  no_ecbf:        <sps>
  no_nswf:        <sps>
  no_divergent:   <sps>
  no_reperfusion: <sps>
  no_mmicrl:      <sps>

Total spend: Rs ~<amount>   (per-run budget: Rs 1,500; user hard-cap: Rs 2,500)
```

Best-reward gather:
```bash
python - <<'PY'
import json, os
for r in ("no_ecbf","no_nswf","no_divergent","no_reperfusion","no_mmicrl"):
    for s in range(10):
        p = f"logs/ablation_{r}/seed_{s}/summary.json"
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"ablation_{r:18s} seed_{s}: best_reward={d.get('best_reward'):.1f} "
                  f"steps={d.get('total_steps')} trip={d.get('budget_tripped')}")
        else:
            print(f"ablation_{r:18s} seed_{s}: MISSING summary.json")
PY
```

### STEP 11 — [VM] CSV audit

```bash
python - <<'PY'
import csv, os, sys
missing, short, ok = [], [], []
for r in ("no_ecbf","no_nswf","no_divergent","no_reperfusion","no_mmicrl"):
    for s in range(10):
        p = f"Results 2/logs/ablation_{r}/seed_{s}/training_log.csv"
        if not os.path.exists(p):
            missing.append(p); continue
        rows = list(csv.DictReader(open(p)))
        if not rows:
            short.append((p, 0)); continue
        last_step = int(rows[-1].get("global_step") or 0)
        # Require >= 1.9M (95% of 2M)
        (short if last_step < 1_900_000 else ok).append((p, last_step))
print(f"OK      ({len(ok)})")
print(f"SHORT   ({len(short)}):")
for p, s in short:
    print(f"  {p}  last_step={s:,}")
print(f"MISSING ({len(missing)}):")
for p in missing:
    print(f"  {p}")
if short or missing:
    sys.exit(1)
PY
echo "audit exit=$?"
```

Pass: `audit exit=0`. If any run is SHORT or MISSING, STOP, list them,
wait for LOCAL to decide whether to re-launch specific seeds.

### STEP 12 — [VM] Add provenance + aggregation to `Results 2/`

```bash
# Frozen configs snapshot
mkdir -p "Results 2/_configs_snapshot"
cp config/experiment_matrix.yaml "Results 2/_configs_snapshot/"
cp config/ablation_no_ecbf.yaml \
   config/ablation_no_nswf.yaml \
   config/ablation_no_divergent.yaml \
   config/ablation_no_reperfusion.yaml \
   config/ablation_no_mmicrl.yaml \
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
  echo
  echo "## Determinism state"
  python -c "
from hcmarl.utils import seed_everything
import inspect, os
src = inspect.getsource(seed_everything)
print('use_deterministic_algorithms(True):', 'use_deterministic_algorithms(True' in src)
print('CUBLAS_WORKSPACE_CONFIG:           ', 'CUBLAS_WORKSPACE_CONFIG' in src)
print('matmul_precision(highest):         ', 'matmul_precision(\"highest\")' in src)
print('Currently in env:', os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<not set in shell>'))
"
} > "Results 2/_provenance.txt"

# Per-(rung, seed) aggregation summary.
# (IQM/CI computation runs LOCAL; this is the raw summary.)
# Field list matches what train.py:1033-1044 actually writes to summary.json
# PLUS a few extras computed from training_log.csv (cost_ema, safety_rate
# at last step; sps_mean over the run).
python - <<'PY'
import csv, json
from pathlib import Path
OUT = Path("Results 2/_aggregation_summary.csv")
# These fields exist in summary.json directly:
SUMMARY_FIELDS = ["total_episodes", "total_steps", "best_reward",
                  "wall_time_seconds", "wall_time_hours",
                  "budget_tripped", "spent_inr"]
# These are computed from training_log.csv last row + run-mean SPS:
EXTRA_FIELDS = ["last_cost_ema", "last_safety_rate", "sps_mean"]
fields = ["rung", "seed"] + SUMMARY_FIELDS + EXTRA_FIELDS
rows = []
for r in ("no_ecbf","no_nswf","no_divergent","no_reperfusion","no_mmicrl"):
    for s in range(10):
        p = Path(f"Results 2/logs/ablation_{r}/seed_{s}")
        row = {"rung": r, "seed": s}
        # 1. Pull from summary.json (fields confirmed in train.py:1033-1044)
        summary = p / "summary.json"
        if summary.exists():
            d = json.loads(summary.read_text())
            for k in SUMMARY_FIELDS:
                row[k] = d.get(k)
        # 2. Compute extras from training_log.csv (last row + global SPS)
        csv_path = p / "training_log.csv"
        if csv_path.exists():
            try:
                csv_rows = list(csv.DictReader(open(csv_path)))
                if csv_rows:
                    last = csv_rows[-1]
                    row["last_cost_ema"] = last.get("cost_ema")
                    row["last_safety_rate"] = last.get("safety_rate")
                    if len(csv_rows) >= 2:
                        first = csv_rows[0]
                        try:
                            d_step = int(last["global_step"]) - int(first["global_step"])
                            d_time = float(last["wall_time"]) - float(first["wall_time"])
                            row["sps_mean"] = round(d_step / d_time, 1) if d_time > 0 else None
                        except (KeyError, ValueError, ZeroDivisionError):
                            row["sps_mean"] = None
            except Exception as e:
                print(f"  WARN: failed to parse {csv_path}: {e}")
        rows.append(row)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"Aggregation CSV written: {OUT}  ({len(rows)} rows expected = 50)")
PY

# Format-patch any unpushed commits made on the VM during this run
# (e.g., RUNBOOK amendment, config fix). The VM has no GitHub credentials,
# so push will fail; patches in Results 2/ ride along with the tarball.
cd /root/hcmarl_project
n_unpushed=$(git rev-list --count origin/master..HEAD 2>/dev/null || echo 0)
if [ "$n_unpushed" -gt 0 ]; then
  echo "Saving $n_unpushed unpushed commits as patches..."
  git format-patch -${n_unpushed} -o "Results 2/"
  ls "Results 2/"*.patch
else
  echo "No unpushed commits."
fi

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

echo "=== Results 2/ tree (depth 2) ==="
ls -la "Results 2/"
du -sh "Results 2/"
```

**Pass criterion for STEP 12:** `_aggregation_summary.csv` has 50
non-empty rows; `Results 2/logs/ablation_<r>/seed_{0..9}/training_log.csv`
all exist. If any row is blank or any CSV is missing, STOP.

### STEP 13 — [VM] Stand down

```
EXP2 verified. Results 2/ assembled. 50/50 runs OK. Standing by for §0.2 close-out paste.
```

Wait. Do not launch anything else.

---

## 5. Status-report cadence

Every **20 minutes** after STEP 8 kickoff. The grid runs ~18-22 hours;
the user may step away. Every 20 minutes is the observability guarantee.

Post *immediately* when: a run finishes; any kill-switch fires;
`run_ablations.py` reports FAILED; you applied a §8 minor fix; nvidia-smi
shows GPU memory > 40 GB; you encountered a non-minor blocker and
stopped; total elapsed > 24h; total spend approaches ₹2,200 (90% of cap).

---

## 6. Known failure modes (context for VM Claude)

### 6.1 Build-up ladder leakage (legacy)
Old experiment_matrix.yaml had 5-rung build-up rungs (`mappo`,
`plus_ecbf`, etc.). If the VM git-pulled before the EXP2 push landed,
`pytest` STEP 5 will fail on D2 tests because the config test names
don't match. Fix: re-pull from `origin/master`.

### 6.2 3M anchor legacy error
Old matrix had `curve_anchors_steps: [500K, 1M, 2M, 3M]`. If a stale
copy runs with 2M-step ablations, the aggregator legitimately errors
at 3M. Current matrix has `[500K, 1M, 2M]` only. Verify in CHECK 3.

### 6.3 HCMARL reference contamination
EXP2 does NOT re-train HCMARL. If the VM mistakenly invokes
`run_baselines.py --methods hcmarl`, it would launch full-HCMARL runs
into `logs/hcmarl/` instead of ablation rungs. Guard: STEP 8 uses
`run_ablations.py`, not `run_baselines.py`. Double-check tmux command
string before Enter.

### 6.4 ECBF QP CVXPY warning
`ecbf_filter.py:309` occasionally emits "Solution may be inaccurate"
under high-stress states. Benign. Does NOT count as a pytest failure.
Note that this only fires for rungs that keep ECBF on
(no_nswf, no_divergent, no_reperfusion, no_mmicrl). `no_ecbf` skips
the CVXPY solver entirely.

### 6.5 MMICRL MI=0 on real data (NOT a failure for 4 of 5 rungs)
On real WSD4FEDSRM data, MMICRL reports MI ≈ 0 → safety ceilings fall
back to per-muscle config floors. This is the documented MI-collapse
guard working correctly. For `no_ecbf`, `no_nswf`, `no_divergent`,
`no_reperfusion`, MMICRL stays enabled but collapses identically — so
each is a clean remove-one. The `no_mmicrl` rung disables MMICRL
entirely so the comparison is "MMICRL ran but collapsed" vs "MMICRL
never ran".

### 6.6 Determinism-op crash mid-grid (PRE-RESOLVE via STEP 7.5)
If `torch.use_deterministic_algorithms(True, warn_only=False)` raises
on a CUDA op the audit missed, STEP 7.5's 50K probe catches it before
the full launch. If it slips past the probe and fires during the grid,
the §0.1 emergency policy authorises VM Claude to flip
`warn_only=False → warn_only=True` in `hcmarl/utils.py`. This is the
ONLY pre-authorised source edit.

### 6.7 GitHub push blocked by missing credentials (EXPECTED, NOT A FAILURE)
The VM has no SSH outbound key, no PAT, no `gh` CLI logged in, no
`.netrc`. Any `git push` will fail with `fatal: could not read Username
for 'https://github.com'`. This is expected. Do NOT attempt to set up
credentials. Save commits via `git format-patch` into `Results 2/` per
STEP 12. Patches ride along with the tarball when the user pulls.

### 6.8 Resume on mid-run crash is NOT automated
`scripts/run_ablations.py:177-184` invokes `train.py` without `--resume`.
If a seed crashes at, say, 1.5M of 2M steps, `run_ablations.py` will
mark it FAILED but NOT auto-relaunch. Manual recovery is possible:
```
python scripts/train.py --config config/ablation_<rung>.yaml --method hcmarl \
  --seed <s> --device cuda --run-name ablation_<rung> \
  --resume checkpoints/ablation_<rung>/seed_<s>/checkpoint_<step>.pt
```
The run will resume from `global_step` and append new rows to the CSV
(logger validates header). Watchdog STEP 9 flags FAILED runs immediately.

### 6.9 Concurrent-VM mistake
If you find yourself looking for `Results 1/` — STOP. That folder lives
on a DIFFERENT VM (account #1). Do NOT scp from there. Do NOT block on
its absence. EXP2 is fully independent of EXP1 on the VM side; merging
happens LOCAL post-pull.

### 6.10 L4 thermal throttling
L4 sustained clock at high-load can throttle 5-10% after hour 1.
Monitor via `nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics
--format=csv,noheader`. If sustained drops > 15%, reduce `--max-parallel
6→4` (will extend wall-clock by ~20% but reduce thermal load).

---

## 7. Files, paths, layout (quick reference)

```
/root/hcmarl_project/
├── RUNBOOK_EXP2.md                          ← THIS FILE
├── config/
│   ├── hcmarl_full_config.yaml              # reference; NOT re-run in EXP2
│   ├── ablation_no_ecbf.yaml                # ECBF off, others on
│   ├── ablation_no_nswf.yaml                # NSWF off, others on
│   ├── ablation_no_divergent.yaml           # disagreement constant, others on
│   ├── ablation_no_reperfusion.yaml         # r=1, others on
│   ├── ablation_no_mmicrl.yaml              # MMICRL off, others on (NEW 2026-05-02)
│   ├── pathg_profiles.json                  # 34 calibrated profiles for MMICRL pretrain
│   └── experiment_matrix.yaml               # 5 rungs, seeds [0..9]
├── hcmarl/                                  # SACRED — do not edit
│   ├── utils.py                             # seed_everything has B2 HIGH-determinism patch
│   ├── three_cc_r.py
│   ├── ecbf_filter.py
│   ├── nswf_allocator.py
│   ├── mmicrl.py
│   └── real_data_calibration.py
├── scripts/
│   ├── train.py
│   └── run_ablations.py                     # --fresh-logs + --max-parallel + matrix-driven
├── logs/                                    # SYMLINK -> Results 2/logs/ (set up in STEP 7)
├── checkpoints/                             # SYMLINK -> Results 2/checkpoints/ (STEP 7)
└── Results 2/                               # ← FINAL DELIVERABLE
    ├── logs/ablation_{5 rungs}/seed_{0..9}/{training_log.csv, summary.json, mmicrl_results.json}
    ├── checkpoints/ablation_{5 rungs}/seed_{0..9}/{checkpoint_*.pt, run_state.pt}
    ├── _exp0_gpu_probe/                     # STEP 7.5 probe outputs (small)
    ├── _configs_snapshot/                   # frozen 5 ablation configs + matrix + hcmarl ref
    ├── _provenance.txt                      # git, torch, hardware, determinism state
    ├── _aggregation_summary.csv             # one row per (rung, seed); IQM/CI computed LOCAL
    ├── _exp2_run.log                        # launcher stdout (teed live)
    ├── 0001-*.patch ... NNNN-*.patch        # any unpushed commits (STEP 12)
    └── _INDEX.txt                           # file listing
```

**Why symlinks:** `scripts/train.py` and `hcmarl/logger.py` hardcode
`logs/` and `checkpoints/` as output roots. The symlinks redirect those
writes into `Results 2/` with zero source edits and zero data-loss risk
on grid interruption.

**Expected size:** ~2-3 GB (50 runs × ~50 MB each: CSVs ~1 MB, checkpoints ~50 MB).

---

## 8. Minor-blocker authority

**MAY fix in place + report:**
- Missing pip package (`pip install <pkg>`).
- Stale `.pyc` (`find . -name __pycache__ -exec rm -rf {} +`).
- CUDA visibility (`export CUDA_VISIBLE_DEVICES=0`).
- Tmux session recovery (new session, same name, ONLY if all CSVs are intact).
- **Determinism warn_only flip** per §6.6 / §0.1 emergency policy
  (the ONLY pre-authorised source edit).
- `git format-patch` to save commits when push fails (expected behaviour).

**MUST NOT touch:**
- Any file under `hcmarl/` (source) EXCEPT the warn_only flip.
- `scripts/train.py`, `scripts/run_ablations.py`,
  `scripts/aggregate_learning_curves.py`.
- Any `config/*.yaml`.
- `tests/`.
- `git push` (will fail anyway; do not retry or attempt credential setup).
- Kill-switches.
- Phase A constants (three_cc_r, ecbf_filter, nswf_allocator,
  mmicrl, real_data_calibration) — PDF-verified and SACRED.

---

## 9. Emergency procedures

| Situation | What VM does |
|---|---|
| pytest fails STEP 5 | STOP. Post last 30 lines. |
| Any STEP 6 check fails | STOP. Post full output. Wait for LOCAL. |
| CUDA not available after STEP 3 | Retry cu121. If still False, STOP. |
| STEP 1 reports something other than L4 | STOP. Escalate (methodology requires identical hardware to EXP1). |
| STEP 7.5 probe deterministic-op crash | Identify op; flip warn_only; re-probe. Document in next status. |
| STEP 7.5 probe SPS < 150 | STOP. Escalate. |
| lazy-agent trip on any seed | Post immediately. Do NOT disable. Let grid finish. |
| budget trip on any seed | Post immediately. That seed halts cleanly; grid continues. |
| Hang with no log output > 20 min | `tmux capture-pane -t exp2 -p \| tail -200`. Don't kill. |
| nvidia-smi memory > 40 GB | Post immediately as warning. Continue unless OOM kills a process. |
| OOM kill on a single seed | Post immediately, do NOT restart. Wait for LOCAL decision. |
| L4 sustained clock drop > 15% | Post immediately. Consider --max-parallel 6→4. |
| STEP 11 audit short/missing | Post the output. Wait for LOCAL. |
| E2E billing out of line | STOP status reports, user checks dashboard. |
| Rate limit on Claude | User has standby account. Relaunch claude, re-paste §0.1. |
| Total elapsed > 24h | Post immediately; check budget vs hard cap (₹2,500). |
| Total spend > ₹2,200 | Post immediately; await user decision. |

---

## 10. Results-format summary (for the record)

Every file under `Results 2/` is:
- **Human-readable** — CSVs open in a spreadsheet; JSONs pretty-print.
- **Python-analyzable** — `training_log.csv` has labelled columns per
  eval episode: `(episode, global_step, cumulative_reward, cumulative_cost,
  safety_rate, peak_fatigue, per_agent_entropy_mean, per_agent_entropy_min,
  lazy_agent_flag, ...)` — ready for pandas/matplotlib/seaborn.
  Schema verified compatible with `scripts/aggregate_learning_curves.py`
  (requires `global_step` and `cumulative_reward` — both present).
- **Claude-interpretable** — plain text + structured data throughout.
- **IQM/CI-ready** — every CSV has `global_step` and `cumulative_reward`
  columns. The LOCAL command after pull:
  ```bash
  python scripts/aggregate_learning_curves.py \
      --matrix config/experiment_matrix.yaml \
      --logs-root "Results 2/logs" \
      --out "Results 2/aggregated_results.json"
  ```
  produces IQM + stratified bootstrap 95% CI at each anchor (500K, 1M,
  2M) per rung. **Runs on the laptop** in ~30 seconds (CPU-only; do
  NOT run on VM — that wastes L4-hours).

No visualization / analysis / interpretation on VM. That is Experiment
4, done locally on the laptop.

---

## 11. End-of-session pull commands (USER runs locally after §0.2)

User workflow per `exp 1 gitbash.txt`: **tar-piped-ssh** (NOT rsync, NOT scp).

After VM replies `EXP2 complete. Standing by.`:

Replace `<account2-ip>` with the E2E account #2 IP. Use the SSH key
configured for account #2 (likely `~/.ssh/id_ed25519`, but verify on
E2E dashboard).

```bash
# ============================================================
# 1. NEW — Full VM repo snapshot (separate folder; existing local
#    stays untouched). Includes .git history with any commits
#    made on this VM (saved as patches in Results 2/ too, redundant).
# ============================================================
mkdir -p /c/Users/admin/Desktop/hcmarl_VM_SNAPSHOT_exp2
ssh -i ~/.ssh/id_ed25519 root@<account2-ip> "cd /root && tar -czf - hcmarl_project" \
  | tar -xzf - -C /c/Users/admin/Desktop/hcmarl_VM_SNAPSHOT_exp2 --strip-components=1

# ============================================================
# 2. PRIMARY — Results 2 to Downloads (your headline ablation data)
# ============================================================
ssh -i ~/.ssh/id_ed25519 root@<account2-ip> "cd /root/hcmarl_project && tar -czf - 'Results 2'" \
  | tar -xzf - -C /c/Users/admin/Downloads/

# ============================================================
# 3. BELT-AND-BRACES #1 — logs mirror (~30 MB, fast safety copy)
# ============================================================
mkdir -p /c/Users/admin/Downloads/logs_exp2_mirror
ssh -i ~/.ssh/id_ed25519 root@<account2-ip> "cd /root/hcmarl_project && tar -czhf - logs" \
  | tar -xzf - -C /c/Users/admin/Downloads/logs_exp2_mirror/

# ============================================================
# 4. BELT-AND-BRACES #2 — checkpoints mirror (~2 GB, slow but worth it)
# ============================================================
mkdir -p /c/Users/admin/Downloads/checkpoints_exp2_mirror
ssh -i ~/.ssh/id_ed25519 root@<account2-ip> "cd /root/hcmarl_project && tar -czhf - checkpoints" \
  | tar -xzf - -C /c/Users/admin/Downloads/checkpoints_exp2_mirror/
```

The `-h` flag in transfers #3 and #4 dereferences the top-level symlinks
on the VM so the mirrors copy real files (not dangling symlink markers).

After all transfers:
```bash
du -sh /c/Users/admin/Downloads/"Results 2"/
ls /c/Users/admin/Downloads/"Results 2"/logs/      # 5 ablation_* dirs
ls /c/Users/admin/Downloads/"Results 2"/checkpoints/  # 5 ablation_* dirs
cat /c/Users/admin/Downloads/"Results 2"/_INDEX.txt | head -30
```

Expected: `Results 2/` ~2-3 GB, 5 rung subdirs in both logs/ and
checkpoints/, each with 10 seed subdirs, each with `training_log.csv`,
`summary.json`, and (for 4 of 5 rungs) `mmicrl_results.json`.

After verification:
1. Send §0.2 close-out paste to VM; VM acknowledges.
2. Destroy the E2E account #2 node from the dashboard.
3. Confirm billing has stopped on account #2.

Then locally compute IQM/CI:
```bash
cd /c/Users/admin/Desktop/hcmarl_project
python scripts/aggregate_learning_curves.py \
    --matrix config/experiment_matrix.yaml \
    --logs-root /c/Users/admin/Downloads/"Results 2"/logs \
    --out /c/Users/admin/Downloads/"Results 2"/aggregated_results.json
```

---

## 12. End-of-session checklist (user runs after §0.2 paste)

- [ ] VM final summary (§10 format) posted with 50-row rung × seed table.
- [ ] STEP 11 audit `exit=0` confirmed.
- [ ] STEP 12 `Results 2/_INDEX.txt` exists with 50+ seed entries.
- [ ] §11 transfer #1 (full VM repo snapshot) completed; verify `git log` has any unpushed commits.
- [ ] §11 transfer #2 (Results 2 to Downloads) completed; folder ~2-3 GB.
- [ ] §11 transfer #3 (logs mirror) completed.
- [ ] §11 transfer #4 (checkpoints mirror) completed.
- [ ] `ls /c/Users/admin/Downloads/"Results 2"/logs/ablation_*/seed_*/training_log.csv`
      returns 50 non-empty paths.
- [ ] §0.2 close-out sent; VM acknowledged.
- [ ] E2E account #2 node destroyed in dashboard.
- [ ] Billing shows account #2 charge stopped.

Only after all ten boxes tick is this session "done."

---

## 13. What EXP2 deliberately does NOT do (scope discipline)

| Item | Why deferred | To |
|---|---|---|
| Re-train HCMARL full | Already done in EXP1; first 10 seeds on identical L4 hardware are the reference | EXP1 (running concurrently on account #1) |
| Headline 6-method baseline grid | Out of EXP2 scope | EXP1 (running) |
| Synthetic K=3 MMICRL validation | CPU-side, ARI=1.0 done | EXP3 Part 1 (already complete) |
| Synthetic K=3 HCMARL vs no_MMICRL | Different experiment | EXP3 Part B (next VM session, after EXP1+EXP2) |
| Sensitivity analysis (Path G ±20%/±50%) | Uses checkpoints from EXP1+EXP2; runs LOCALLY post-EXP1 | LOCAL post-EXP4 |
| Two-axis (worker, seed) bootstrap | Aggregation step, needs all CSVs | EXP4 (laptop) |
| IQM/CI computation on VM | CPU-only; wasted L4-hours | LOCAL post-pull |
| Visualisation / plotting | Headless VM matplotlib is a footgun | EXP4 (laptop) |
| Paper writing | LOCAL-only | POST-EXP4 |

Keep EXP2 narrow: produce 50 clean ablation training CSVs and the
`Results 2/` deliverable. Everything else comes later.

---

## 14. Pre-critic / pre-resolve summary (TL;DR for VM Claude)

The two prior EXP2 attempts (and the EXP1 attempts whose lessons inform
this) failed on:

1. **Configuration drift** (env block missing, wrong rung count, wrong
   seed count, wrong total_steps, no_mmicrl missing, ECBF wrong,
   determinism missing). RESOLVED by STEP 6 CHECKS 1-10.

2. **Subtle agent bugs** (HAPPO M_running collapse, MACPO div-by-zero,
   MACPO buffer dispatch). EXP2 does NOT use HAPPO or MACPO — only
   HCMARL with one component disabled per rung. So those blockers
   cannot fire here. The only HCMARL-specific risk is the ECBF QP CVXPY
   warning, which is documented benign.

3. **GitHub push failure** (no credentials on VM). EXPECTED. Use
   `git format-patch` instead. Patches saved into `Results 2/` ride
   along with the tarball.

4. **Determinism-op crash** (HIGH determinism on a CUDA op the audit
   missed). PRE-RESOLVED by STEP 7.5 probe — catches in 10-15 min on L4.
   Emergency authority: flip warn_only.

5. **Speed projection failure** (earlier runbook said "500+ SPS";
   reality on L4 was 215-228 SPS for HCMARL). RESOLVED on this revision:
   §2 budget math uses MEASURED EXP1 L4 SPS (200-250 SPS for HCMARL-class,
   ~400 SPS for `no_ecbf`). Aspirational targets removed.

6. **Cross-hardware concern (RESOLVED by hardware choice).** Earlier
   v3 of this runbook targeted L40S which would have introduced
   trajectory drift vs EXP1's L4 baseline. v4 reverts to L4 to maintain
   apples-to-apples methodology with EXP1. The remove-one comparison
   `HCMARL (EXP1, L4)` vs `no_X (EXP2, L4)` is therefore on identical
   hardware, identical determinism, identical code commit. No
   methodology footnote needed.

The remaining risks:

- **L4 thermal throttling** at hour 1+ — monitor in 20-min reports.
  Reduce `--max-parallel` 6→4 if clocks drop > 15%.
- **Per-seed cost variance** — `no_ecbf` is ~50% faster, others are
  ~95% of HCMARL pace. Per-rung SPS in status reports tracks this.
- **MMICRL collapse on 4 of 5 rungs** is EXPECTED (real-data MI≈0
  documented). The `no_mmicrl` rung removes MMICRL entirely so that
  axis is cleanly separated from the env-data behaviour.
- **GPU OOM** — unlikely on 24-48 GB but watch the 40 GB warning line.
- **Concurrent VM (account #1, also L4) interference** — there is none.
  The two VMs share nothing on the runtime side. Local merging happens
  post-pull only.

If those risks materialise, the runbook tells you exactly what to do.
This v4 runbook is informed by EXP1's actual measured L4 behaviour and
matches its hardware exactly. Every step exists because something
earlier in the project broke without it.
