# RUNBOOK_EXP1 — HC-MARL + 5 baselines headline run on L4 (2026-05-02 v3)

> **If you are Claude Code running inside the rented E2E VM terminal, this
> entire file is your briefing. Read every line top to bottom — do not skim,
> do not skip, do not summarise. The §0.1 boot prompt tells you exactly when
> to act and when to stop. Follow it.**

This is **Experiment 1, third attempt** in the user's numbered experiment
plan. Two prior attempts failed:
- Attempt #1 (2026-04-20): seed mismatch, ECBF left on for some baselines,
  baseline configs missing the `environment:` block → `safety_cost()`
  returned 0 → silent constraint-free training.
- Attempt #2 (2026-04-25): PS-IPPO logic was per-agent (50-60 SPS) and
  HIGH determinism was not active, so seed runs diverged.

**This is the FINAL chance.** Pre-critic and pre-resolve are mandatory.

---

## 0. What this runbook covers — and what it does NOT

**In scope (post-2026-05-02 lineup):**
- **Six methods**: `hcmarl`, `mappo`, `mappo_lag`, `macpo`, `happo`,
  `shielded_mappo` — exactly the headline grid in
  `config/experiment_matrix.yaml`. **NO IPPO. NO PS-IPPO.**
- **Ten seeds each**: 0..9 → **60 total runs**.
- **Hardware**: NVIDIA L4 / 50 vCPU / 220 GB RAM / 48 GB GPU memory /
  CUDA 12 / 250 GB SSD / IOPS R 60000 W 30000 / **₹98 per hour** (E2E
  Networks on-demand).
- `--max-parallel 6` (rationale §4 STEP 8). 6-way parallel + 50 vCPUs
  gives 8 BLAS threads per process, matching the validated 25-vCPU /
  3-way calibration point.
- HIGH determinism: `torch.use_deterministic_algorithms(True,
  warn_only=False)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, matmul precision
  `"highest"`, cudnn.deterministic=True. Set inside `seed_everything()`
  in `hcmarl/utils.py`.
- Logs target: `logs/{hcmarl,mappo,mappo_lag,macpo,happo,shielded_mappo}/seed_{0..9}/training_log.csv`.
- **Final deliverable**: top-level `Results 1/` containing every CSV,
  summary JSON, MMICRL result, checkpoint, configs snapshot, provenance.
  Pulled to laptop via `tar -czf - … | tar -xzf -` per user workflow.

**Out of scope (deferred):**
- Ablation grid (`scripts/run_ablations.py`) — Experiment 2.
- Path G real-data MMICRL evaluation — Experiment 3 Part 1 (already done).
- Synthetic K=3 sanity validation — Experiment 3 Part B.
- Visualisation / analysis / interpretation — Experiment 4 (LOCAL ONLY).
- Any edit to `hcmarl/` source files beyond §8 minor fixes.

---

## 0.1 Boot prompt (paste into VM-side Claude Code immediately after `claude` launches)

```
You are the VM-side Claude for the HC-MARL Experiment 1 headline run on
an E2E L4 GPU node (THIRD attempt, 2026-05-02). This paste IS your full
starting instruction.

Your complete briefing is at /root/hcmarl_project/RUNBOOK_EXP1.md on
this VM. Read every line top to bottom. Do not skim, do not skip, do
not summarise.

CRITICAL NON-NEGOTIABLE PRE-FLIGHT (if any of these is not true, STOP):
  1. You cloned the MOST RECENT push from
     github.com/ADITYA-WORK-MAITI/hcmarl-project (branch master).
     Recent commits include: 4ee2ef3 (EXP0 runner update), 45baa8b
     (HAPPO/MACPO blocker fixes), 235f63b (citation discipline),
     e883024 (math doc v14).
  2. ECBF is OFF in all FIVE baseline configs. Prove it:
     `grep -E '^\s*enabled:' config/{mappo,mappo_lag,macpo,happo,shielded_mappo}_config.yaml`
     -- every line must show `enabled: false`.
  3. ECBF is ON in hcmarl_full_config.yaml. Prove it:
     `grep -E '^\s*enabled:' config/hcmarl_full_config.yaml`
     -- must show `enabled: true`.
  4. Every baseline config has a populated environment: section with
     muscle_groups, theta_max, AND tasks. Without this, safety_cost()
     silently returns 0. That happened on 2026-04-20 and ruined the
     first run.
  5. logs/{6 methods}/ and checkpoints/{6 methods}/ are EMPTY on the
     VM. Either never existed (fresh clone) or will be wiped by
     `--fresh-logs` before the launcher spawns its first subprocess.
     NEVER append new data to a contaminated CSV.
  6. `config/experiment_matrix.yaml` headline.methods has exactly 6
     entries (NO ippo) and seeds=[0..9].
  7. The new agent classes exist:
       hcmarl/agents/{happo,macpo,shielded_mappo}.py
     and the train.py METHODS dict has all 6 of:
       hcmarl, mappo, mappo_lag, macpo, happo, shielded_mappo
     (ippo may still be present in METHODS for backward compat -- it
     is NOT in the headline matrix and MUST NOT be launched).
  8. HIGH determinism is active inside seed_everything (the B2 patch
     of 2026-05-02). Verify:
     `grep -n "use_deterministic_algorithms\|CUBLAS_WORKSPACE_CONFIG\|matmul_precision" hcmarl/utils.py`
     -- must show all three: torch.use_deterministic_algorithms(True),
     CUBLAS_WORKSPACE_CONFIG=":4096:8", set_float32_matmul_precision("highest").
  9. Dry-run banner verification:
     `python scripts/run_baselines.py --methods hcmarl mappo mappo_lag macpo happo shielded_mappo --dry-run 2>&1 | head -5`
     FIRST LINE must be exactly:
     `Headline grid: 6 methods x 10 seeds = 60 runs`
     SECOND LINE must be exactly (Python list str repr):
     `Methods: ['hcmarl', 'mappo', 'mappo_lag', 'macpo', 'happo', 'shielded_mappo']`
     If either line deviates, STOP.

If ANY of the 9 checks above fails, STOP, post the failing check, wait.

After reading, execute STEPs 1-13 of RUNBOOK_EXP1.md continuously, in
order, without asking for confirmation between steps:

  STEP 1   hardware sanity (expect L4 or better, >= 24 GB VRAM, >= 48 vCPUs)
  STEP 2   python3.12 venv
  STEP 3   torch from cu124 (fall back to cu121)
  STEP 4   pip install -r requirements.txt
  STEP 5   pytest -q  (pass: 0 failed; expect 589 passed, 2 skipped)
  STEP 6   the NINE pre-flight checks above
  STEP 7   symlink logs/ + checkpoints/ into Results 1/
  STEP 7.5 EXP0 50K GPU probe -- catches HIGH-determinism crashes BEFORE
           burning the full grid. If torch.use_deterministic_algorithms
           hits a non-deterministic CUDA op, this step shows it in 2-5 min
           instead of ~70 min into the first headline seed.
  STEP 8   create tmux session `exp1`; launch the 6-method 10-seed grid
  STEP 9   status report every 20 minutes (format in §5)
  STEP 10  when grid finishes, post the exit summary (§6)
  STEP 11  CSV audit must pass before standing down
  STEP 12  add provenance + aggregation to Results 1/
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

Begin now. After reading RUNBOOK_EXP1.md, reply with exactly:
`Read. Nine pre-flight checks at STEP 6. Results land in 'Results 1/'.`
and start.
```

---

## 0.2 Close-out prompt (paste after the CSV audit passes and Results 1/ is populated)

```
Experiment 1 complete. Your work is done for this session.

The user will now tar+ssh-pull Results 1/ to the laptop and destroy
the E2E node manually. Those are [USER] steps.

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
| **LOCAL** (laptop) | scope decisions, git state, RUNBOOK edits, tar+ssh pull, node destruction | direct SSH execution on VM |
| **VM** (Claude Code on L4) | bootstrap, 9 pre-flight checks, EXP0 probe, tmux launch, 20-min status reports, minor §8 fixes, CSV audit, `Results 1/` assembly | scope decisions, editing anything under hcmarl/ source (except the determinism warn_only flip per §0.1 emergency policy), git commits, destroying the node, running anything after STEP 13 |
| **USER** | E2E dashboard, SSH, git clone before claude launches, paste between sessions, tar+ssh pull `Results 1/`, destroy node | running training outside tmux |

---

## 2. Project state at the start of this session (post-commit `4ee2ef3`)

- **Six methods in headline matrix**: hcmarl, mappo, mappo_lag, macpo,
  happo, shielded_mappo. IPPO removed from headline (the agent class
  remains in `train.py` METHODS dict for backward compat but the
  experiment_matrix.yaml does NOT list it).
- **HAPPO** (`hcmarl/agents/happo.py`, 2026-05-02): heterogeneous-agent
  PPO with PPO-clip variant per Kuba et al. 2022 ICLR Eq. 11.
  Per-agent (heterogeneous) actors + centralised critic, sequential
  per-agent update with random permutation each epoch + M-product
  importance scaling. Blocker fix 2026-05-02: `M_running.clamp(0.5, 2.0)`
  hard floor prevents the M-vector from collapsing to zero.
- **MACPO** (`hcmarl/agents/macpo.py`, 2026-05-02): multi-agent
  constrained policy optimisation per Gu et al. 2023.
  Per-agent actors + centralised reward + cost critics, conjugate-
  gradient natural gradient direction (10 CG iters, damping 0.1),
  single-cost-constraint dual projection (CPO Appendix B closed form),
  KL+cost backtracking line search (10 steps, decay 0.8). Blocker fix
  2026-05-02: zero-gradient guard skips the agent's update cleanly when
  `g.norm() < 1e-8` (avoids 1/0 in the dual step computation).
- **Shielded-MAPPO** (`hcmarl/agents/shielded_mappo.py`, 2026-05-02):
  MAPPO + hand-designed static-threshold task-refusal shield. If any
  required muscle is at >= (theta_max - safety_margin), substitute
  action -> rest. PPO-consistent: log_prob recomputed for the shielded
  action so the buffer's (s, a, log_p) triple stays internally
  consistent. Critical for the ECBF-vs-shielding contribution claim
  (Shielded-MAPPO is the static-threshold non-RL safety baseline that
  isolates whether ECBF's QP-based continuous adjustment buys anything
  over a one-line if-statement).
- **HIGH determinism** (`hcmarl/utils.py::seed_everything`, B2 patch
  2026-05-02): `torch.use_deterministic_algorithms(True,
  warn_only=False)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `matmul_precision="highest"`, `python random.seed(seed)` added.
  EXP3 Part 1 ran under exactly this setup (ARI=1.0 on synthetic K=3).
- **Train.py dispatch**: `is_lagrangian = is_mappo_lag or is_macpo`
  (pre-launch audit fix 2026-05-02). Without this, MACPO would crash
  at the first env step calling `buffer.store(...)` without `cost` /
  `cost_values`.
- **Run scripts**: `run_baselines.py` and `run_ablations.py` both have
  `--fresh-logs` (run_ablations.py got it 2026-05-02). Both are matrix-
  driven from `config/experiment_matrix.yaml`.
- **Test suite**: 589 passed, 2 skipped, 0 failed (laptop, 2026-05-02).
- **EXP0 (local CPU smoke)**: 9 of 9 steps OK in ~8 minutes
  (`Results 0/Result EXPERIMENT_0_SUMMARY.txt`). All 6 method smokes
  pass. All 5 ablation smokes pass. Constants ledger writes cleanly
  (niosh import bug fixed). Test suite green.

**Budget reality (L4 on-demand, ₹98/hr):**
- 6 methods × 10 seeds × 2M steps. SPS varies by method:
  - MAPPO / Shielded-MAPPO: ~1500 SPS expected (T1 batched action select)
  - MAPPO-Lag: ~1200 SPS (extra cost critic)
  - HCMARL: ~800-1000 SPS (full stack with ECBF QP per step)
  - HAPPO: ~600-900 SPS (per-agent forward passes + sequential update)
  - MACPO: ~300-500 SPS (CG iterations + line search per agent)
- Average ~900 SPS across the lineup. 2M steps / 900 sps = ~2200 sec = 37 min/run.
- 60 runs / 6-way parallel = 10 sequential batches × ~37 min = ~6 hours.
- Realistic with overhead + thermal throttling at hour 1+: **8-12 hours**.
- **Expected spend: ₹800-1200**. Budget cap below.

**Per-run kill-switch:** `--budget-inr 1500`. Triggers per individual run
when its wall-clock spend exceeds the converted seconds. Non-binding for
expected SPS.

**Hard total stop:** ₹2,500 user-set ceiling. If E2E dashboard shows
spend exceeding that, STOP and escalate immediately.

---

## 3. Pre-flight on the laptop (already done by LOCAL — reported here for VM to verify)

LOCAL confirmed before pushing commit `4ee2ef3`:
1. **All 6 method configs present** with populated `environment:` block
   (muscle_groups + theta_max + tasks all PDF-verified Frey-Law 2012).
2. **ECBF enabled state**:
   - `hcmarl_full_config.yaml`: `enabled: true` (ECBF on)
   - `mappo_config.yaml`, `mappo_lag_config.yaml`, `macpo_config.yaml`,
     `happo_config.yaml`, `shielded_mappo_config.yaml`: all `enabled: false`
3. **HAPPO blocker fix landed**: M_running clamp `(0.5, 2.0)`.
4. **MACPO blocker fix landed**: zero-gradient guard before CG.
5. **Shielded-MAPPO config** has `shield: { safety_margin: 0.05,
   demand_threshold: 0.0 }`.
6. **MACPO config** has trust-region hyperparams: `delta_kl: 0.01`,
   `cg_iters: 10`, `cg_damping: 0.1`, `line_search_steps: 10`,
   `line_search_decay: 0.8`, `cost_limit: 0.1`.
7. **`config/experiment_matrix.yaml`** headline.methods has exactly 6
   entries (NO ippo): hcmarl, mappo, mappo_lag, happo, macpo,
   shielded_mappo. Seeds [0..9].
8. **HIGH determinism active** in `seed_everything` (B2 patch).
9. **train.py is_lagrangian** dispatch includes MACPO.
10. **`pytest -q`** -- 589 passed, 2 skipped, 0 failed.
11. **EXP0 local smoke** -- 9 of 9 steps OK; `Results 0/` populated.
12. **Phase A constants** -- untouched since the PDF-verification pass
    (Frey-Law 2012, Liu 2002, Xia 2008, Kaneko-Nakamura 1979).
13. **`git push origin master`** -- complete; tip is `4ee2ef3`.

---

## 4. Execution plan (VM owns STEPs 1-13)

### STEP 0 — [USER] boot the VM + clone the LATEST repo

Manual, before Claude Code launches (per `exp 1 gitbash.txt` workflow):

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@<public-ip>

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

# Confirm RUNBOOK_EXP1.md (NOT EXP2) and recent commit
ls RUNBOOK_EXP1.md && git log -1 --format='%h %s'

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
- **GPU**: NVIDIA L4 (or better). VRAM total: report what `nvidia-smi`
  prints (E2E listing says 48 GB; if the actual card shows 24 GB, that
  is still acceptable — pre-flight memory budget §2 fits in 24 GB).
  VRAM free at startup: ≥ 20 GiB.
- **vCPUs**: ≥ 48
- **RAM free**: ≥ 100 GiB (we'll only use ~20 GiB peak)
- **Disk free**: ≥ 200 GiB

Report to user regardless. If GPU is missing or VRAM < 16 GiB, STOP.

### STEP 2 — [VM] Create the venv

```bash
cd /root/hcmarl_project
python3.12 -m venv venv
source venv/bin/activate
python --version           # 3.12.x
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
pip list | grep -iE "^(torch|gymnasium|numpy|cvxpy|osqp|scipy|pyyaml|pypdf|pytest)"
```

Torch version must include `+cu124` or `+cu121`. If pip replaced it
with the CPU wheel, re-run STEP 3 with `--force-reinstall`.

### STEP 5 — [VM] Pytest sanity

```bash
pytest -q 2>&1 | tail -5
```

Expected: **589 passed, 2 skipped, 0 failed**. 1-3 skips acceptable
(env-conditional). One harmless CVXPY warning from `ecbf_filter.py:309`
is expected and benign. If anything else fails, STOP.

### STEP 6 — [VM] The NINE non-negotiable pre-flight checks

```bash
echo "=== CHECK 1: git tip ==="
git log -1 --format='%h %ci %s'
echo

echo "=== CHECK 2: ECBF OFF in all 5 baselines ==="
grep -H '^\s*enabled:' \
  config/mappo_config.yaml \
  config/mappo_lag_config.yaml \
  config/macpo_config.yaml \
  config/happo_config.yaml \
  config/shielded_mappo_config.yaml
echo

echo "=== CHECK 3: ECBF ON in hcmarl_full_config ==="
grep -E '^\s*enabled:' config/hcmarl_full_config.yaml
echo

echo "=== CHECK 4: environment section populated in all 6 ==="
for f in config/hcmarl_full_config.yaml \
         config/mappo_config.yaml \
         config/mappo_lag_config.yaml \
         config/macpo_config.yaml \
         config/happo_config.yaml \
         config/shielded_mappo_config.yaml; do
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

echo "=== CHECK 5: logs + checkpoints clean for all 6 methods ==="
for m in hcmarl mappo mappo_lag macpo happo shielded_mappo; do
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

echo "=== CHECK 6: experiment_matrix has 6 methods, 10 seeds, no ippo ==="
python - <<'PY'
import yaml
m = yaml.safe_load(open("config/experiment_matrix.yaml"))
methods = list(m["headline"]["methods"].keys())
seeds = m["headline"]["seeds"]
expected_methods = {"hcmarl","mappo","mappo_lag","macpo","happo","shielded_mappo"}
assert set(methods) == expected_methods, f"method set mismatch: {set(methods)} vs {expected_methods}"
assert "ippo" not in methods, "ippo MUST NOT be in the headline matrix"
assert seeds == list(range(10)), f"seeds wrong: {seeds}"
print(f"  methods: {methods}")
print(f"  seeds:   {seeds}")
print("  OK")
PY
echo

echo "=== CHECK 7: new agent classes exist ==="
ls hcmarl/agents/{happo,macpo,shielded_mappo}.py
python - <<'PY'
from hcmarl.agents.happo import HAPPO
from hcmarl.agents.macpo import MACPO
from hcmarl.agents.shielded_mappo import ShieldedMAPPO
h = HAPPO(obs_dim=19, global_obs_dim=109, n_actions=6, n_agents=6, device="cpu")
m = MACPO(obs_dim=19, global_obs_dim=109, n_actions=6, n_agents=6, device="cpu")
print(f"  HAPPO actors: {len(h.actors)} (per-agent)")
print(f"  MACPO actors: {len(m.actors)} (per-agent)")
print(f"  MACPO buffer: {type(m.buffer).__name__}")
print("  OK")
PY
echo

echo "=== CHECK 8: HIGH determinism active in seed_everything ==="
grep -n "use_deterministic_algorithms\|CUBLAS_WORKSPACE_CONFIG\|matmul_precision" hcmarl/utils.py | head -10
python - <<'PY'
from hcmarl.utils import seed_everything
import inspect
src = inspect.getsource(seed_everything)
assert "use_deterministic_algorithms(True" in src, "MISSING: use_deterministic_algorithms(True"
assert "CUBLAS_WORKSPACE_CONFIG" in src,           "MISSING: CUBLAS_WORKSPACE_CONFIG"
assert 'matmul_precision("highest")' in src,       "MISSING: matmul_precision(highest)"
print("  HIGH determinism: all 3 mechanisms present in seed_everything OK")
PY
echo

echo "=== CHECK 9: dry-run banner ==="
python scripts/run_baselines.py \
    --methods hcmarl mappo mappo_lag macpo happo shielded_mappo \
    --dry-run 2>&1 | head -5
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    [sys.executable, "scripts/run_baselines.py",
     "--methods","hcmarl","mappo","mappo_lag","macpo","happo","shielded_mappo",
     "--dry-run"], text=True).splitlines()
want1 = "Headline grid: 6 methods x 10 seeds = 60 runs"
want2 = "Methods: ['hcmarl', 'mappo', 'mappo_lag', 'macpo', 'happo', 'shielded_mappo']"
assert out[0] == want1, f"LINE 1 wrong:\n  got:  {out[0]}\n  want: {want1}"
assert out[1] == want2, f"LINE 2 wrong:\n  got:  {out[1]}\n  want: {want2}"
print("  dry-run banner OK")
PY
echo

echo "All nine checks complete."
```

Expected outcome:
- CHECK 1 — recent commit hash starts `4ee2ef3` or later.
- CHECK 2 — five `enabled: false` lines (mappo, mappo_lag, macpo, happo, shielded_mappo).
- CHECK 3 — one `enabled: true` line (hcmarl).
- CHECK 4 — all 6 configs have non-zero entries for muscle_groups, theta_max, tasks.
- CHECK 5 — twelve `clean:` lines.
- CHECK 6 — methods set = expected, no ippo, seeds=[0..9].
- CHECK 7 — three agent files exist; each constructs cleanly.
- CHECK 8 — HIGH determinism mechanisms present.
- CHECK 9 — `6 methods x 10 seeds = 60 runs`; method list ordered.

**If any check fails, STOP. Post the full output. Do not proceed.**

### STEP 7 — [VM] Symlink `logs/` and `checkpoints/` into `Results 1/`

```bash
# Defensive cleanup in case logs/ or checkpoints/ exist as real dirs:
rm -rf logs checkpoints 2>/dev/null || true

# Create the deliverable tree, then point logs/ AND checkpoints/ at it.
mkdir -p "Results 1/logs" "Results 1/checkpoints"
ln -sfn "$(pwd)/Results 1/logs" logs
ln -sfn "$(pwd)/Results 1/checkpoints" checkpoints

ls -la logs checkpoints
test -L logs && test -L checkpoints \
  || { echo "ERROR: logs or checkpoints is not a symlink"; exit 1; }
```

After this step, `ls -la logs checkpoints` MUST print two symlink lines.
If either is a real directory, STOP and retry the `ln -sfn` line.

### STEP 7.5 — [VM] EXP0 50K GPU probe (CRITICAL — pre-resolve determinism)

This is the moment HIGH determinism either works or doesn't.
`torch.use_deterministic_algorithms(True, warn_only=False)` raises
immediately on the first non-deterministic CUDA op. If the full grid
launches and one method hits a non-deterministic op 70 minutes in,
that's wasted GPU money. Do the smallest possible probe FIRST.

```bash
# Use macpo as the probe -- it has the most code paths (CG, second-order
# autograd, line search, dual solver). If macpo's 50K probe runs clean,
# the simpler methods will too.
mkdir -p "Results 1/_exp0_gpu_probe"
timeout 600 python scripts/train.py \
    --config config/macpo_config.yaml \
    --method macpo \
    --seed 0 \
    --device cuda \
    --max-steps 50000 \
    --run-name _exp0_gpu_probe_macpo \
    2>&1 | tee "Results 1/_exp0_gpu_probe/macpo_50k.log"
echo "macpo probe exit=$?"

# Also probe hcmarl (which exercises ECBF QP + MMICRL pretrain).
timeout 900 python scripts/train.py \
    --config config/hcmarl_full_config.yaml \
    --method hcmarl \
    --seed 0 \
    --device cuda \
    --max-steps 50000 \
    --run-name _exp0_gpu_probe_hcmarl \
    2>&1 | tee "Results 1/_exp0_gpu_probe/hcmarl_50k.log"
echo "hcmarl probe exit=$?"
```

**Pass criterion:** both exit codes 0; both log tails show training
progress (no `RuntimeError: ... does not have a deterministic
implementation`); SPS reported in log >= 200 (this is conservative;
the smaller-than-expected SPS during the probe is because of MMICRL
pretrain on hcmarl and CG warmup on macpo).

**If a deterministic-op error fires:**
1. Identify the offending op from the traceback (e.g., `scatter_add`,
   `index_add`, `embedding_bag`).
2. Per §0.1 emergency policy, you may flip warn_only=False to
   warn_only=True in `hcmarl/utils.py::seed_everything`.
3. Document in your status report: "Determinism warn_only flipped due
   to <op> at <file:line>; within-run determinism preserved, cross-run
   bit-identity not guaranteed."
4. Re-run the probe; if it passes, proceed to STEP 8.

**If the probe SPS < 200 even without crashes:** STOP and escalate.
That suggests CPU-bound rollout or thread-cap misconfiguration.

After both probes pass, clean up the probe artefacts so they don't
contaminate the headline grid:
```bash
rm -rf "Results 1/logs/_exp0_gpu_probe_"* \
       "Results 1/checkpoints/_exp0_gpu_probe_"*
```

### STEP 8 — [VM] Create tmux session + launch the grid

```bash
# Tmux session (separate from the one Claude itself is in)
tmux kill-session -t exp1 2>/dev/null || true
pkill -f "scripts/train.py" 2>/dev/null || true
pkill -f "run_baselines.py" 2>/dev/null || true
sleep 1
tmux new -d -s exp1
tmux send-keys -t exp1 "cd /root/hcmarl_project && source venv/bin/activate" Enter
sleep 1
tmux list-sessions | grep exp1

# Launch the grid
tmux send-keys -t exp1 "python scripts/run_baselines.py \
  --methods hcmarl mappo mappo_lag macpo happo shielded_mappo \
  --device cuda \
  --fresh-logs \
  --max-parallel 6 \
  --budget-inr 1500 \
  --cost-per-hour 98.0 \
  --budget-margin 0.95 \
  2>&1 | tee 'Results 1/_exp1_run.log'" Enter
```

Within 30 seconds, `tmux capture-pane -t exp1 -p | tail -20` must show:
```
Headline grid: 6 methods x 10 seeds = 60 runs
Methods: ['hcmarl', 'mappo', 'mappo_lag', 'macpo', 'happo', 'shielded_mappo']
Seeds:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

If the banner mentions **5 methods, 50 runs, contains "ippo", or is
missing any of the 6 expected methods**, immediately `Ctrl-C` and STOP.

**Flags explained — do not improvise:**
- `--methods hcmarl mappo mappo_lag macpo happo shielded_mappo` — all
  6 in one grid. Filter is explicit so if experiment_matrix.yaml gains
  a method later it won't silently join.
- `--device cuda` — L4.
- `--fresh-logs` — **NON-NEGOTIABLE**. Wipes `logs/{method}/` +
  `checkpoints/{method}/` for each method before launching. Physical
  prevention against the 2026-04-20 "append-on-matching-header" bug.
  Without this flag, STOP.
- `--max-parallel 6` — **6 concurrent seeds**. Rationale:
  - Calibration point: 3-way on L4 with 25 vCPUs = 8.3 vCPUs per
    process, ran cleanly. Holding vCPUs-per-process constant: 50/8.3
    = 6.0 processes. Not 7 (would oversubscribe at 7.1 vCPUs/proc).
  - BLAS thread cap auto-set to `OMP_NUM_THREADS = 50/6 = 8` per
    process by run_baselines.py.
  - 48 GB VRAM / ~3 GB peak per process (HCMARL during MMICRL
    pretrain) = ~16-way GPU memory headroom. Not the constraint.
- `--budget-inr 1500` — per-run kill-switch (~15 hours per run @ ₹98/hr;
  way over expected ~37 min).
- `--cost-per-hour 98.0` — L4 on-demand at E2E. Required for kill-switch
  math.
- `--budget-margin 0.95` — kill at 95% of the budget.
- `tee Results 1/_exp1_run.log` — launcher stdout mirror.

Do NOT add `--resume` — clean-slate grid.
Do NOT drop `--methods` — default would re-scan experiment_matrix.yaml
(currently exact match, but defensive specificity matters).
Do NOT alter seed list — 10 seeds locked.

### STEP 9 — [VM] Automated status reports (every 20 minutes)

**You do not wait to be asked.** Post this block every 20 min:

```
### Status report — <UTC HH:MM> (elapsed: <Hh:MMm> since STEP 8 kickoff)

Runs done / in-flight / pending:   <done>/<inflight>/<pending>  (of 60)
Per-method SPS (rolling 50 ep):
  hcmarl:         <sps>
  mappo:          <sps>
  mappo_lag:      <sps>
  macpo:          <sps>
  happo:          <sps>
  shielded_mappo: <sps>
ETA to grid completion:            ~<Hh:MMm>
Wall-clock spend so far:           Rs ~<amount>  (@ Rs 98/hr)
lazy_agent trips since start:      <count>
budget_tripped events:             <count>
GPU mem in use (nvidia-smi):       <MiB> / 48 GiB total
CPU load (uptime):                 <load>
pytest state:                      green (STEP 5)

Last 5 lines of Results 1/_exp1_run.log:
  <paste>

Notes / minor fixes applied:       <describe or "none">
```

Field-gathering commands:
```bash
# n_done -- count seed dirs with >= 2 lines in training_log.csv
for m in hcmarl mappo mappo_lag macpo happo shielded_mappo; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    f=logs/$m/seed_$s/training_log.csv
    [ -f "$f" ] && [ "$(wc -l <"$f")" -ge 2 ] && echo "$m seed_$s"
  done
done | wc -l

# Per-method SPS (rolling, last 50 episodes if available)
python - <<'PY'
import csv, glob, os
for m in ("hcmarl","mappo","mappo_lag","macpo","happo","shielded_mappo"):
    sps_vals = []
    for s in range(10):
        f = f"logs/{m}/seed_{s}/training_log.csv"
        if not os.path.exists(f): continue
        with open(f) as h:
            rows = list(csv.DictReader(h))
        if len(rows) < 5: continue
        # Last 50 rows: derive sps from total_steps / total_time
        last = rows[-50:] if len(rows) > 50 else rows
        try:
            steps = int(last[-1]["global_step"]) - int(last[0]["global_step"])
            t = float(last[-1].get("wall_time_sec", 0)) - float(last[0].get("wall_time_sec", 0))
            if t > 0: sps_vals.append(steps / t)
        except (KeyError, ValueError):
            pass
    if sps_vals:
        print(f"  {m:20s} mean SPS = {sum(sps_vals)/len(sps_vals):.0f} (n={len(sps_vals)} seeds)")
    else:
        print(f"  {m:20s} no SPS data yet")
PY

# lazy/budget trips
grep -c "lazy-agent kill-switch" "Results 1/_exp1_run.log" 2>/dev/null || echo 0
grep -c "budget kill-switch"     "Results 1/_exp1_run.log" 2>/dev/null || echo 0

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

Additionally, post *immediately* (do not wait for 20-min tick) when:
- A run finishes or fails.
- A `lazy-agent kill-switch` or `budget kill-switch` fires.
- `run_baselines.py` prints `FAILED (exit code X)`.
- You apply a §8 minor fix.
- nvidia-smi shows GPU memory > 40 GB (potential OOM warning).

### STEP 10 — [VM] Final exit summary

When the launcher prints `All 60 jobs complete.`, post:

```
### EXP1 grid done — <UTC HH:MM> (total wall-clock: <Hh:MMm>)

| Method         | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | best_reward range |
|----------------|----|----|----|----|----|----|----|----|----|----|-------------------|
| hcmarl         |    |    |    |    |    |    |    |    |    |    |                   |
| mappo          |    |    |    |    |    |    |    |    |    |    |                   |
| mappo_lag      |    |    |    |    |    |    |    |    |    |    |                   |
| macpo          |    |    |    |    |    |    |    |    |    |    |                   |
| happo          |    |    |    |    |    |    |    |    |    |    |                   |
| shielded_mappo |    |    |    |    |    |    |    |    |    |    |                   |
(cells: D=DONE, F=FAIL, L=lazy-trip, B=budget-trip)

Kill-switch events:
  lazy_agent trips: <count>  (list)
  budget trips:     <count>  (list)

Failures: <list or "none">

Per-method SPS final:
  hcmarl:         <sps>
  mappo:          <sps>
  mappo_lag:      <sps>
  macpo:          <sps>
  happo:          <sps>
  shielded_mappo: <sps>

Total spend: Rs ~<amount>   (per-run budget: Rs 1,500; user hard-cap: Rs 2,500)
```

Best-reward gather:
```bash
python - <<'PY'
import json, os
for m in ("hcmarl","mappo","mappo_lag","macpo","happo","shielded_mappo"):
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
for m in ("hcmarl","mappo","mappo_lag","macpo","happo","shielded_mappo"):
    for s in range(10):
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

Pass: `audit exit=0`. If any run is SHORT or MISSING, STOP, list them,
wait for LOCAL to decide whether to re-launch specific seeds.

### STEP 12 — [VM] Add provenance + aggregation to `Results 1/`

```bash
# Frozen configs snapshot
mkdir -p "Results 1/_configs_snapshot"
cp config/experiment_matrix.yaml "Results 1/_configs_snapshot/"
cp config/hcmarl_full_config.yaml \
   config/mappo_config.yaml \
   config/mappo_lag_config.yaml \
   config/macpo_config.yaml \
   config/happo_config.yaml \
   config/shielded_mappo_config.yaml \
   "Results 1/_configs_snapshot/"

# Provenance
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
} > "Results 1/_provenance.txt"

# Aggregation summary
python - <<'PY'
import csv, json
from pathlib import Path
OUT = Path("Results 1/_aggregation_summary.csv")
fields = ["method","seed","total_steps","best_reward","final_cost_ema",
          "final_safety_rate","budget_tripped","lazy_tripped","sps_mean"]
rows = []
for m in ("hcmarl","mappo","mappo_lag","macpo","happo","shielded_mappo"):
    for s in range(10):
        p = Path(f"Results 1/logs/{m}/seed_{s}")
        summary = p / "summary.json"
        row = {"method": m, "seed": s}
        if summary.exists():
            d = json.loads(summary.read_text())
            for k in ("total_steps","best_reward","final_cost_ema",
                      "final_safety_rate","budget_tripped","lazy_tripped",
                      "sps_mean"):
                row[k] = d.get(k)
        rows.append(row)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"Aggregation CSV written: {OUT}")
PY

# Final index
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

**Pass criterion for STEP 12:** `_aggregation_summary.csv` has 60
non-empty rows; `Results 1/{6 methods}/seed_{0..9}/training_log.csv`
all exist. If any row is blank or any CSV is missing, STOP.

### STEP 13 — [VM] Stand down

```
EXP1 verified. Results 1/ assembled. Standing by for §0.2 close-out paste.
```

Then wait. Do not launch anything else.

---

## 5. Status-report cadence

Every **20 minutes** after STEP 8 kickoff. The grid runs ~8-12 hours;
the user may step away. Every 20 minutes is the observability guarantee.

Post *immediately* when: a run finishes; any kill-switch fires;
`run_baselines.py` reports FAILED; you applied a §8 minor fix; nvidia-smi
shows GPU memory > 40 GB; you encountered a non-minor blocker and
stopped.

---

## 6. Known failure modes (context for VM Claude)

### 6.1 Constraint-free baseline training (the env-section bug, 2026-04-20)
Cause: baseline configs missing `environment:` block → `safety_cost()`
returns 0 → policy learns degenerate single-task policy → entropy
collapses → lazy-agent trip at ~100K steps.
Prevention: fail-fast guard in `scripts/train.py` + populated env blocks
+ STEP 6 CHECK 4.

### 6.2 Contaminated CSV (append-on-matching-header bug, 2026-04-20)
Cause: `HCMARLLogger` appends to existing CSV if header matches.
Prevention: `--fresh-logs` flag + STEP 6 CHECK 5. STEP 8 ALWAYS uses
`--fresh-logs`.

### 6.3 PS-IPPO 50-60 SPS (the slow-baseline bug, 2026-04-25)
Cause: per-agent separate networks → n_agents CUDA launches per step.
**Resolution today**: PS-IPPO/IPPO dropped from headline matrix
entirely. Replaced with HAPPO + MACPO + Shielded-MAPPO.

### 6.4 Determinism not active (2026-04-25)
Cause: `seed_everything` only set `cudnn.deterministic=True`; matmul
TF32 was on (precision="high"); `use_deterministic_algorithms` was not
called. Per-seed runs diverged.
Prevention: B2 patch in `hcmarl/utils.py` (2026-05-02). All three
mechanisms active: see STEP 6 CHECK 8.

### 6.5 HAPPO M_running collapse to zero (caught in audit, 2026-05-02)
Cause: clamp range `[0.0, 2.0]` allowed M_running to vanish if any
ratio exactly hit 0; subsequent agents in the permutation got zero
gradient and silently didn't train.
Prevention: clamp tightened to `[0.5, 2.0]` in `hcmarl/agents/happo.py`.

### 6.6 MACPO division-by-zero on near-optimal policy (caught 2026-05-02)
Cause: when `g.norm() ≈ 0`, the CG returned zero, the dual gave lam=0,
the step formula computed `1/lam` which was Inf/NaN.
Prevention: zero-gradient guard at the top of `_agent_update` returns a
clean diagnostic dict if `g.norm() < 1e-8`.

### 6.7 MACPO is_lagrangian dispatch bug (caught pre-launch 2026-05-02)
Cause: `is_lagrangian = isinstance(agent, MAPPOLagrangian)` only — MACPO
fell through to the generic-buffer branch which calls `buffer.store(...)`
without `cost`/`cost_values`, raising TypeError on the first env step.
Prevention: `is_lagrangian = is_mappo_lag or is_macpo` in `train.py`.

### 6.8 HCMARL MMICRL MI-collapse on real WSD4FEDSRM (NOT a failure)
On real Path G data, MMICRL returns K with MI ≈ 0 (the data does not
exhibit multi-type structure). The MI-collapse guard at
`hcmarl/utils.py` falls back to per-worker config-default ceilings.
This is the designed behaviour, not a bug. HCMARL then runs with
uniform config ceilings = same as baselines on the real-data axis.
This is expected, documented, and is the central honesty point of the
paper (Exp 3 Part B validates the type-inference machinery on synthetic
K=3 instead).

### 6.9 Determinism-op crash mid-grid (PRE-RESOLVE)
If `torch.use_deterministic_algorithms(True, warn_only=False)` raises
on a CUDA op we missed in the audit, STEP 7.5 catches it before the
full launch. If somehow it slips past the probe and fires during the
grid, the §0.1 emergency policy authorises VM Claude to flip
`warn_only=False → warn_only=True` in `hcmarl/utils.py`. This is the
ONLY pre-authorised source edit.

---

## 7. Files, paths, layout (quick reference)

```
/root/hcmarl_project/
├── RUNBOOK_EXP1.md                  ← THIS FILE
├── config/
│   ├── hcmarl_full_config.yaml      # ECBF enabled (only one)
│   ├── mappo_config.yaml            # ECBF off
│   ├── mappo_lag_config.yaml        # ECBF off (cost_limit set)
│   ├── macpo_config.yaml            # ECBF off (delta_kl, cg_iters set)
│   ├── happo_config.yaml            # ECBF off (heterogeneous actors)
│   ├── shielded_mappo_config.yaml   # ECBF off, has shield: block
│   └── experiment_matrix.yaml       # 6 methods, seeds [0..9]
├── hcmarl/agents/
│   ├── mappo.py                     # MAPPO + T1 batched action selection
│   ├── mappo_lag.py                 # MAPPO + PID Lagrangian
│   ├── macpo.py                     # CG natural gradient + dual + line search
│   ├── happo.py                     # heterogeneous actors + sequential update
│   ├── shielded_mappo.py            # MAPPO + static-threshold shield
│   ├── ippo.py                      # PS-IPPO (kept for backward compat, NOT in matrix)
│   ├── networks.py                  # ActorNetwork, CriticNetwork, CostCriticNetwork
│   └── hcmarl_agent.py              # full HCMARL wrapper
├── scripts/
│   ├── train.py                     # MACPO is_lagrangian dispatch fix
│   └── run_baselines.py             # --fresh-logs + --max-parallel
├── logs/                            # SYMLINK -> Results 1/logs/ (set up in STEP 7)
├── checkpoints/                     # SYMLINK -> Results 1/checkpoints/ (STEP 7)
└── Results 1/                       # ← FINAL DELIVERABLE
    ├── logs/{6 methods}/seed_{0..9}/{training_log.csv, summary.json, mmicrl_results.json}
    ├── checkpoints/{6 methods}/seed_{0..9}/{checkpoint_*.pt, run_state.pt}
    ├── _exp0_gpu_probe/             # STEP 7.5 probe outputs (small)
    ├── _configs_snapshot/           # frozen 6 configs + matrix
    ├── _provenance.txt              # git, torch, hardware, determinism state
    ├── _aggregation_summary.csv     # one row per (method, seed)
    ├── _exp1_run.log                # launcher stdout (teed live)
    └── _INDEX.txt                   # file listing
```

**Why symlinks:** `scripts/train.py` and `hcmarl/logger.py` hardcode
`logs/` and `checkpoints/` as output roots. The symlinks redirect
those writes into `Results 1/` with zero source edits and zero data-
loss risk on grid interruption.

**Why both logs/ AND checkpoints/ go inside Results 1/:** reviewers
need the trained `.pt` weights for reproducibility. Size: ~50 MB ×
6 methods × 10 seeds = 3 GB.

---

## 8. Minor-blocker authority (what VM Claude MAY fix in place)

**MAY fix (report every one):**
- Missing pip package that requirements.txt should have pulled.
- Typo'd import in a **test** file that blocks collection (not source).
- Stale `.pyc` (`find . -name __pycache__ -exec rm -rf {} +`).
- CUDA visibility (`export CUDA_VISIBLE_DEVICES=0`).
- Tmux session recovery (new session, same name, ONLY if all CSVs are
  intact).
- **Determinism warn_only flip** per §6.9 / §0.1 emergency policy
  (the ONLY pre-authorised source edit).

**MUST NOT touch:**
- Any file under `hcmarl/` (source) EXCEPT the warn_only flip.
- `scripts/train.py`, `scripts/run_baselines.py`,
  `scripts/aggregate_learning_curves.py`.
- Any `config/*.yaml`.
- `tests/*.py` beyond trivial import-typo fixes.
- `requirements.txt` — use `--force-reinstall` if needed.
- `.git`, `git add/commit/push`.
- Kill-switches.
- Phase A constants (three_cc_r, ecbf_filter, nswf_allocator,
  real_data_calibration) — PDF-verified and SACRED.

---

## 9. Emergency procedures

| Event | Response |
|---|---|
| SSH drops mid-run | tmux keeps running. `tmux attach -t exp1`, resume reports. |
| pytest fails STEP 5 | STOP. Post last 30 lines. |
| Any STEP 6 check fails | STOP. Post full output. Wait for LOCAL. |
| CUDA not available after STEP 3 | Retry cu121. If still False, STOP. |
| STEP 7.5 probe deterministic-op crash | Identify op; flip warn_only; re-probe. Document in next status. |
| STEP 7.5 probe SPS < 200 | STOP. Escalate. |
| lazy-agent trip on any seed | Post immediately. Do NOT disable. Let grid finish. |
| budget trip on any seed | Post immediately. That seed halts cleanly; grid continues. |
| Hang with no log output > 20 min | `tmux capture-pane -t exp1 -p \| tail -200`. Don't kill. |
| nvidia-smi memory > 40 GB | Post immediately as warning. Continue unless OOM kills a process. |
| OOM kill on a single seed | Restart that seed manually with smaller batch_size? NO — STOP and escalate. |
| STEP 11 audit short/missing | Post the output. Wait for LOCAL. |
| E2E billing out of line | STOP status reports, user checks dashboard. |
| Rate limit on Claude | User has standby account. Relaunch claude, re-paste §0.1. |

---

## 10. Results-format summary (for the record)

Every file under `Results 1/` is:
- **Human-readable** — CSVs open in a spreadsheet; JSONs pretty-print.
- **Python-analyzable** — all numeric data in CSV / JSON, no squeezed
  aggregations at the VM stage (that's EXP4, locally).
- **Claude-interpretable** — plain text + structured data throughout.
- **Visualizable** — `training_log.csv` has one row per eval episode
  with (episode, global_step, reward, cost, safety_rate, peak_MF,
  per_agent_entropy_mean, per_agent_entropy_min, lazy_agent_flag, ...)
  ready for pandas/matplotlib/seaborn.

No visualization / analysis / interpretation on VM. That is Experiment
4, done locally on the laptop.

---

## 11. End-of-session pull commands (USER runs locally after §0.2)

User workflow per `exp 1 gitbash.txt`: tar+ssh, NOT rsync.

After VM replies `EXP1 complete. Standing by.`:

```bash
# PRIMARY — full Results 1/ (~3 GB) into Downloads
ssh -i ~/.ssh/id_ed25519 root@<public-ip> \
    "cd /root/hcmarl_project && tar -czf - 'Results 1'" \
  | tar -xzf - -C /c/Users/admin/Downloads/

# BELT-AND-BRACES #1 — independent mirror of logs/ (~30 MB CSVs/JSONs)
mkdir -p /c/Users/admin/Downloads/logs_exp1_mirror/
ssh -i ~/.ssh/id_ed25519 root@<public-ip> \
    "cd /root/hcmarl_project && tar -czhf - logs" \
  | tar -xzf - -C /c/Users/admin/Downloads/logs_exp1_mirror/

# BELT-AND-BRACES #2 — independent mirror of checkpoints/ (~3 GB .pt)
mkdir -p /c/Users/admin/Downloads/checkpoints_exp1_mirror/
ssh -i ~/.ssh/id_ed25519 root@<public-ip> \
    "cd /root/hcmarl_project && tar -czhf - checkpoints" \
  | tar -xzf - -C /c/Users/admin/Downloads/checkpoints_exp1_mirror/
```

The `-h` flag in the belt-and-braces tars dereferences the top-level
symlinks on the VM so that mirrors copy real files (not dangling
symlink markers).

After all three pulls succeed:
```bash
du -sh /c/Users/admin/Downloads/"Results 1"/
ls /c/Users/admin/Downloads/"Results 1"/logs/      # 6 method dirs
ls /c/Users/admin/Downloads/"Results 1"/checkpoints/  # 6 method dirs
cat /c/Users/admin/Downloads/"Results 1"/_INDEX.txt | head -30
```

Expected: `Results 1/` ~3 GB, 6 method subdirs in both logs/ and
checkpoints/, each with 10 seed subdirs, each with `training_log.csv`
and `summary.json`. HCMARL seeds also have `mmicrl_results.json`.

After verification:
1. Send §0.2 close-out paste to VM; VM acknowledges.
2. Destroy the E2E node from the dashboard.
3. Confirm billing has stopped.

---

## 12. End-of-session checklist (user runs after §0.2 paste)

- [ ] VM final summary (§10 format) posted with 60-row method × seed table
- [ ] STEP 11 audit `exit=0` confirmed
- [ ] STEP 12 `Results 1/_INDEX.txt` exists with 60+ seed entries
- [ ] §11 primary `tar+ssh` pull of `Results 1/` completed; folder has
      logs/ + checkpoints/ + metadata, ~3 GB
- [ ] §11 mirror #1 of `logs/` completed
- [ ] §11 mirror #2 of `checkpoints/` completed
- [ ] `ls /c/Users/admin/Downloads/"Results 1"/logs/{hcmarl,mappo,mappo_lag,macpo,happo,shielded_mappo}/seed_*/training_log.csv`
      returns 60 non-empty paths
- [ ] §0.2 close-out sent; VM acknowledged
- [ ] E2E node destroyed in dashboard
- [ ] Billing shows node charge stopped

Only after all nine boxes tick is this session "done."

---

## 13. What EXP1 deliberately does NOT do (scope discipline)

| Item | Why deferred | To |
|---|---|---|
| Visualisation / plotting | VM time is expensive; matplotlib on headless VM is a footgun | EXP4 (laptop) |
| Statistical aggregation (IQM, bootstrap CI) | Needs all 60 CSVs present first | EXP4 (laptop) |
| Ablation grid | Separate run, 5 rungs × 10 seeds, different launcher | EXP2 (next VM session) |
| Path G real-data eval | Already done in EXP3 Part 1 | -- |
| Synthetic K=3 MMICRL validation | CPU-side sanity check, ARI=1.0 done | -- |
| Synthetic K=3 HCMARL vs no_MMICRL | EXP3 Part B | next VM session |
| Sensitivity analysis (Path G ±20%/±50%) | Uses checkpoints from EXP1; runs LOCALLY post-EXP1 | LOCAL post-EXP4 |
| Two-axis (worker, seed) bootstrap | Aggregation step, needs all CSVs | EXP4 (laptop) |
| Paper writing | LOCAL-only, not a VM job | POST-EXP4 |

Keep EXP1 narrow: produce 60 clean training CSVs and the `Results 1/`
deliverable. Everything else comes later.

---

## 14. Pre-critic / pre-resolve summary (TL;DR for VM Claude)

The two prior attempts failed on:
1. **Configuration drift** (env block missing, ECBF wrong, IPPO logic
   broken, no determinism). RESOLVED by STEP 6 CHECKS 1-9 and the audit
   work of 2026-05-02.
2. **Subtle agent bugs** (HAPPO M_running collapse, MACPO div-by-zero,
   MACPO buffer dispatch). RESOLVED by the three blocker fixes landed
   in commits 45baa8b, 235f63b, 4ee2ef3.

The remaining risks for this third attempt are:
- **Determinism crash on a CUDA op missed in audit** — caught by STEP
  7.5 50K probe. Emergency authority to flip warn_only.
- **MACPO slow** — expected ~300-500 SPS (vs 1500 for MAPPO). Not a
  blocker, just slower wall-clock for those 10 seeds.
- **L4 thermal throttling** at hour 1+ — monitor in 20-min reports.
  Reduce `--max-parallel` 6→4 if clocks drop > 10%.
- **GPU OOM** — unlikely on 48 GB but watch the 40 GB warning line.

If those risks materialise, the runbook tells you exactly what to do.
The third attempt is the one that ships. Every step exists because
something earlier broke without it.
