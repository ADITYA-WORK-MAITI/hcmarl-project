# RUNBOOK_HEADLINES — 4 methods × 10 seeds × 2M on live E2E L4 (2026-04-20)

> **If you are Claude Code running inside the rented VM terminal, this entire file is your briefing. Read every single line. Do not skim, do not jump sections, do not summarise in your head and skip ahead. Read top to bottom. The §0.1 boot prompt you were pasted already told you what to do when you finish — follow it exactly. Do not wait for any separate instruction.**

---

## 0. Paste-in prompts (for the user)

Today uses **two Claude agents** (same split as the earlier probe/watch session):

- **LOCAL** — runs on the laptop (this session). Holds full MEMORY.md context. Owns scope decisions, git state, RUNBOOK edits, aggregation interpretation, the final TMLR-defensible claim language.
- **VM** — runs as Claude Code inside the same E2E L4 VM that ran today's probe + 1M watch (IP 164.52.192.64). Venv is already built, pytest is already green, latest commit is `3a1c6d3` (T7 + T1a + T2 + T2b throughput stack). Owns: git pull, batch launch, monitoring, completion reporting.

The user (Aditya) sits at the laptop, talks to LOCAL here, and pastes ready-made blocks from LOCAL into the VM-side Claude Code session when instructed.

Below are the **three** prompts the user will paste into VM-side Claude across this headline run:

- **§0.1 Boot prompt** — pasted once, right after the VM Claude session is launched (after LOCAL has pushed the 2M config edits to GitHub). VM Claude runs continuously through the batch launch and monitoring loop, and stops at the §14 completion report.
- **§0.2 Mid-batch status prompt** — optional, pasted if LOCAL wants a snapshot after wave ~5 of 10. Tells VM Claude to read live CSVs and report without interrupting training.
- **§0.3 Post-completion close-out** — pasted after LOCAL has interpreted the 40-run completion report. Tells VM Claude measurements are done and the user will do the `scp` pull and destroy-node manually.

### 0.1 Boot prompt (paste into VM-side Claude Code immediately after `claude` launches)

```
You are the VM-side Claude for the HC-MARL HEADLINE BATCH on an already-bootstrapped
E2E L4 GPU node (IP 164.52.192.64). The venv exists at /root/hcmarl_project/venv,
pytest is already green on this host, and today's probe + 1M watch are already
archived under /root/hcmarl_project/logs/. This paste IS your full starting instruction.
Execute continuously.

Your complete briefing is at /root/hcmarl_project/RUNBOOK_HEADLINES.md on this VM.
Read every line of that file top to bottom. Do not skim, do not skip, do not summarise.

After you finish reading, execute STEPs 3 through 7 of the RUNBOOK_HEADLINES
continuously, in order, without asking me for confirmation between steps:

  STEP 3  git pull --rebase; confirm latest commit message starts with
          "Headlines: cut total_steps from 3M to 2M". If that commit is not the
          tip, STOP and tell me — LOCAL hasn't pushed yet.
  STEP 4  verify venv still works (python -c "import torch; print(torch.cuda.is_available())")
          — must print True.
  STEP 5  create a fresh tmux session called `headlines`, activate the venv in it,
          then launch the 40-run batch via scripts/run_baselines.py with:
            --seeds 0 1 2 3 4 5 6 7 8 9
            --max-parallel 4
            --device cuda
            --budget-inr 200 --cost-per-hour 49 --budget-margin 0.95
          (The --budget-inr 200 is a PER-RUN kill-switch, not a batch kill. Any
          individual run going past ~3.9 hr wall time is aborted with a clean
          checkpoint. Predicted per-run wall is ~3 hr.)
          Redirect stdout+stderr to logs/headlines_launcher.log. Launch INSIDE tmux
          via `tmux send-keys`, not directly — the session must survive SSH drops.
  STEP 6  every ~2 hours until completion, tmux capture-pane of the `headlines` tmux
          and tail the last 30 lines of logs/headlines_launcher.log. Report to me
          the last DONE/FAILED lines you see. Do NOT attach to the tmux — use
          capture-pane from outside. Do NOT interrupt the batch.
  STEP 7  when `logs/headlines_launcher.log` shows "All 40 jobs complete" (or a
          failures list), collect per-run SPS + final cumulative_reward from each
          of logs/{hcmarl,mappo,ippo,mappo_lag}/seed_{0..9}/training_log.csv and
          post the completion table specified in §7 of RUNBOOK_HEADLINES. Then STOP.

Do NOT destroy the node. Do NOT run git commits. Do NOT edit tracked files. Do NOT
start any further training past the 40-run batch. If any step fails in a way this
runbook does not pre-authorise a fix for, STOP and tell me.

Begin now. When you have finished reading RUNBOOK_HEADLINES.md, reply with exactly
`Read. Executing from STEP 3.` and start.
```

### 0.2 Mid-batch status prompt (optional, pasted when LOCAL wants a snapshot)

```
Snapshot request — do NOT interrupt the running batch.

From outside the `headlines` tmux, run:
  tmux capture-pane -t headlines -p | tail -80
  tail -50 /root/hcmarl_project/logs/headlines_launcher.log
  ls -lh /root/hcmarl_project/logs/{hcmarl,mappo,ippo,mappo_lag}/seed_*/training_log.csv 2>/dev/null | tail -40
  date

For each finished run (training_log.csv has a row with global_step >= 2,000,000),
post: method, seed, final_step, wall_time, SPS = final_step / wall_time, and the
mean cumulative_reward over the last 100 episodes.

For each still-running run (latest row's global_step < 2,000,000), post: method,
seed, current global_step, and estimated remaining wall time assuming 188 SPS.

Then STOP and wait for my next paste. Do not touch the tmux.
```

### 0.3 Post-completion close-out prompt (paste after LOCAL interprets the §7 completion table)

```
Headlines are complete. Your work is done for this session.

The user will now execute the scp pull from the laptop (archive the 40 training_log.csv
files + per-run summary.json + per-run run_state.pt). Do NOT tar/zip anything on the
VM. Do NOT delete anything under /root/hcmarl_project/logs/. Do NOT destroy the node —
that is the user's decision and they may keep it alive to launch ablations separately.

Final acknowledgement: reply with exactly
  `Headlines archived. Standing by. Local-primary will close the session.`
and then wait. If the user asks a specific follow-up question answer it factually;
otherwise stay quiet until SSH is severed.
```

---

## 1. Two-agent division of labour

Do not cross these lines. Two agents editing the same file or executing the same action is worse than one agent with perfect information.

| Agent | Owns | Forbidden |
|---|---|---|
| **LOCAL** (this session) | Scope decisions (seeds, steps, parallel fan-out), config edits (total_steps 3M→2M), git commits + push, RUNBOOK_HEADLINES edits, MEMORY.md, logs/project_log.md, aggregation interpretation, final claim language | Direct SSH execution on the VM. LOCAL hands the user ready-to-paste blocks; VM-side Claude or the user executes them on the node. |
| **VM** (Claude Code on the rented L4) | `git pull` on the VM, launching the 40-run batch in tmux, passive monitoring via `tmux capture-pane` + `tail -f logs/`, reading finished CSVs on the remote, reporting numbers back | Scope decisions, destroying the E2E node, editing `config/*.yaml` or any `.py` on disk, making git commits on the VM, writing to RUNBOOK_HEADLINES, interrupting the running batch (no Ctrl-C, no kill on a running train.py), starting ablations, running anything past STEP 7 without §0.3. |
| **USER** (Aditya) | E2E dashboard (it stays alive — do NOT destroy), SSH-ing into the VM, pasting §0.1/§0.2/§0.3 between sessions, supplying the completion paste back to LOCAL, final scp pull of the 40 CSVs to the laptop, destroying the node ONLY after LOCAL confirms the scp worked | Running training outside tmux. Editing files directly on the VM — ask LOCAL. Destroying the node before `scp` pulls the CSVs. |

When in doubt about ownership: LOCAL holds MEMORY.md and is the brain. VM is the hands. USER is the physical interface to the dashboards and terminals.

---

## 2. Project context (compressed for a cold-start agent)

- **Project:** HC-MARL — Human-Centric Multi-Agent RL for fatigue-aware warehouse task allocation. Author: Aditya Maiti. Target: TMLR (OpenReview, rolling).
- **Repo:** `github.com/ADITYA-WORK-MAITI/hcmarl-project`, branch `master`, tip after today's LOCAL config edit is the "Headlines: cut total_steps from 3M to 2M" commit.
- **Core modules:** 3CC-r fatigue ODE, ECBF safety filter (inline analytical), Nash Social Welfare allocator, MMICRL pretrainer (normalizing flows), MAPPO/IPPO/MAPPO-Lag baselines, PettingZoo-style warehouse env.
- **Workload shape (measured today):** ~188 SPS per-seed at `--max-parallel 4`, ~752 SPS aggregate. Per-run MMICRL pretrain fixed cost ~40 s. Per-run wall at 2M: ~2.97 hr.
- **Tests:** 508 passed, 1 skipped, 0 failed on the VM venv. Must not re-run today — already green.
- **Scope today (headlines only):** 4 methods × 10 seeds × 2M steps = 40 runs. Predicted wall ~29.7 hr. Predicted cost ~Rs 1455. Remaining credits on the E2E account: Rs 1862.80 as of the scp archive step. Buffer after headlines: Rs ~407.
- **Ablations (25 runs)** are NOT on today's list — they run on sir's VM later. Do not touch them.

---

## 3. Today's scope (headline batch only)

Today we do **one batch on E2E** — 40 training runs end-to-end, read the per-seed final rewards + SPS, pull the CSVs to the laptop. That is all.

1. **LOCAL pre-flight**: edit the 4 headline method configs to drop `total_steps` from 3,000,000 to 2,000,000. Commit + push.
2. **VM pull**: `git pull --rebase` on the VM, confirm tip is the 2M commit.
3. **Launch**: `scripts/run_baselines.py --seeds 0 1 2 3 4 5 6 7 8 9 --max-parallel 4 --device cuda --budget-inr 200 --cost-per-hour 49 --budget-margin 0.95` inside tmux session `headlines`.
4. **Monitor**: VM Claude passively reports every ~2 hr from `tmux capture-pane` + tail of launcher log. No interruption.
5. **Completion**: VM Claude emits the §7 completion table. LOCAL reviews.
6. **Archive**: USER scps the 40 CSVs + summary.json + run_state.pt files down to the laptop.
7. **Destroy-or-keep**: USER decides based on remaining credits + ablation plan.

**Per-run kill-switch: Rs 200** (any individual run past ~3.9 hr wall is aborted — predicted is 3.0 hr so this is a runaway protector, not an expected stop).

Do not launch ablations today. Do not re-run the probe. Do not touch the archived watch_1m directory.

---

## 4. Current state (as of this session)

- **VM**: live at `root@164.52.192.64`, Ubuntu 24.04, NVIDIA L4 24 GB, venv at `/root/hcmarl_project/venv`, Python 3.12, pytest green (508/1/0).
- **Repo on VM**: `/root/hcmarl_project`, branch `master`, tip before LOCAL's pre-flight edit is `3a1c6d3` (T2b thread-cap patch). After LOCAL's pre-flight commit, tip will be the "Headlines: cut total_steps from 3M to 2M" commit.
- **Archived measurements under `/root/hcmarl_project/logs/`**:
    - `probe_500k/seed_0/` — 500K pre-patch baseline (87 SPS) — keep, do not delete.
    - `probe_50k/seed_0/` — 50K post-T7+T1a probe (294 SPS) — keep.
    - `watch_1m/seed_0/` — 1M watch run (352 SPS steady, `check_plateau.py` exit 0 PLATEAU, best_reward = -1478, final_reward = -1555, MMICRL collapsed to K=2 with MI=0.0, expected per memory). Keep.
- **SSH key**: `hcmarl-laptop` on E2E, `~/.ssh/id_ed25519` on the laptop.
- **Credits**: Rs 1862.80 remaining as of the archive step earlier today. At Rs 49/hr on-demand, that is ~38 hr of ceiling.
- **Optimizations live** (all in commits cbb0e31 / 7077ced / 3a1c6d3):
    - **T7** — TF32 matmul precision enabled in `hcmarl/utils.py::seed_everything`. No-op on non-TensorCore GPUs.
    - **T1a** — batched actor+critic forward in `MAPPO.get_actions()` and `MAPPOLag.get_actions()`. One CUDA launch per timestep instead of 6.
    - **T2** — `--max-parallel N` ProcessPoolExecutor fan-out in `scripts/run_baselines.py` and `scripts/run_ablations.py`.
    - **T2b** — OMP/MKL/OPENBLAS thread cap per child (auto = vcpu_count // max_parallel). Prevents 25×4 = 100-thread oversubscription on the 25-vCPU host.
- **MMICRL**: on Path G single-muscle profiles, MI collapses to 0.0 (expected — this is the Path G limitation documented in MEMORY). `rescale_to_floor: true` + `mi_collapse_threshold: 0.01` handle it: effective theta falls back to the floor per muscle, a warning is logged, training continues deterministically. Do NOT interpret "MMICRL MI collapse" in the batch log as a bug — it is expected behaviour on this calibration dataset.

---

## 5. Execution plan

Ownership tag on each step: **[USER]** = Aditya does it manually, **[LOCAL]** = LOCAL Claude (this session), **[VM]** = VM-side Claude (the paste-me-to-start session).

**Ordering invariant:** STEPs 1-2 are [LOCAL] + [USER] and must complete before VM-side Claude is even launched. The §0.1 boot prompt is pasted as the first action of STEP 2b, and VM-side Claude then owns STEPs 3-7 continuously.

### STEP 1 — Pre-flight config edit (LOCAL)
**[LOCAL] — makes these edits on the laptop, commits, and pushes.** Four one-line edits — drop `total_steps: 3000000` to `total_steps: 2000000` in the `training:` block of each headline method config:

- `config/hcmarl_full_config.yaml` (line 41)
- `config/mappo_config.yaml` (line 3)
- `config/ippo_config.yaml` (line 3)
- `config/mappo_lag_config.yaml` (line 3)

Do not touch the ablation configs (`config/ablation_no_*.yaml`) — those stay at 3M and will be edited the night before the sir-VM launch.

Commit message:
```
Headlines: cut total_steps from 3M to 2M across 4 method configs

Research Mode + Agarwal 2021 + Stooke 2020 reconciliation: with Rs 1862
credit ceiling and measured 188 per-seed SPS at --max-parallel 4, the only
config that hits Agarwal's N=10 floor for trustworthy IQM CIs AND fits the
budget is 10 seeds x 2M steps x 4 methods = Rs 1455. 3M x 10 seeds is over
budget (Rs 2176); 5 x 3M is "underpowered for ordering claims" per Agarwal.
2M is 20% of the 10M field norm — paper reports IQM + stratified bootstrap
CIs plus per-method plateau inspection on the last 500K of each run (Research
Mode's "flattens with overlapping CIs in the last 20%" criterion).

Ablation configs untouched (stay 3M) — will be matched to 2M the night before
sir-VM launch so headline + ablation step budgets stay equal across the
attribution ladder.
```

Then push to origin/master.

### STEP 2 — VM bring-up (USER → VM Claude)
**[USER]**:
1. SSH back into the VM: `ssh -i ~/.ssh/id_ed25519 root@164.52.192.64`.
2. `cd /root/hcmarl_project && source venv/bin/activate`.
3. Relaunch Claude Code: `claude`.
4. Paste the **§0.1 boot prompt** verbatim.

VM Claude will reply `Read. Executing from STEP 3.` and begin.

From this point on, STEPs 3 through 7 are owned by **[VM]** and run continuously.

### STEP 3 — Git pull + tip verification [VM]
```bash
cd /root/hcmarl_project
source venv/bin/activate
git pull --rebase
git log --oneline -3
```
The tip must be LOCAL's "Headlines: cut total_steps from 3M to 2M" commit. If it is not, STOP and tell LOCAL — LOCAL hasn't pushed yet.

Spot-check the 2M value on one config to prove the pull worked:
```bash
grep "total_steps:" config/hcmarl_full_config.yaml
# must print:  total_steps: 2000000
```

### STEP 4 — Environment sanity [VM]
```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader
free -h | head -2
df -h /root | tail -1
```
Pass criteria:
- `cuda True NVIDIA L4`
- GPU free memory ≥ 22 GiB (probe/watch runs already terminated — no ghost processes)
- RAM free ≥ 100 GiB
- `/root` free ≥ 200 GiB

If any is below spec, STOP and escalate. Do not `rm` anything to "free space" — the probe/watch CSVs are paper-critical and LOCAL has not yet confirmed they scp'd down.

### STEP 5 — Launch the 40-run batch [VM]

**Inside a fresh tmux session `headlines`:**
```bash
tmux new -d -s headlines
tmux send-keys -t headlines "cd /root/hcmarl_project && source venv/bin/activate" Enter
tmux send-keys -t headlines "nohup python scripts/run_baselines.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --max-parallel 4 \
  --device cuda \
  --budget-inr 200 --cost-per-hour 49 --budget-margin 0.95 \
  > logs/headlines_launcher.log 2>&1 &" Enter
```

Key launch-command facts to report back:
- `--seeds 0..9` overrides the matrix's default `[0,1,2,3,4]` for 10-seed Agarwal-floor coverage.
- `--max-parallel 4` triggers the T2 ProcessPoolExecutor fan-out + T2b auto thread-cap (25 vcpu // 4 = 6 threads per child).
- `--budget-inr 200` is a **per-run** hard kill (≈ 3.88 hr wall). Predicted per-run wall is ~2.97 hr, so this is a runaway protector, not an expected stop.
- `--drive-backup-dir` is NOT used — we are NOT on Colab; the laptop's `scp` is the archive path.
- `nohup ... &` inside tmux survives both SSH drops and tmux detach.

Within ~60 s of launch, the launcher log should show:
```
Headline grid: 4 methods x 10 seeds = 40 runs
Methods: ['hcmarl', 'mappo', 'ippo', 'mappo_lag']
Seeds:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Thread cap per child: OMP/MKL/OPENBLAS_NUM_THREADS=6 (max_parallel=4)

Launching 40 runs with --max-parallel=4 (concurrent train.py subprocesses)
```
VM Claude confirms this string is present in `logs/headlines_launcher.log` and reports it to LOCAL. If the launcher exits immediately with a non-zero code, STOP and paste the full launcher log to LOCAL.

### STEP 6 — Passive monitoring every ~2 hr [VM]
```bash
# every ~2 hours, from OUTSIDE the tmux:
tmux capture-pane -t headlines -p | tail -60
tail -50 /root/hcmarl_project/logs/headlines_launcher.log
ls -lh /root/hcmarl_project/logs/{hcmarl,mappo,ippo,mappo_lag}/seed_*/training_log.csv 2>/dev/null | tail -40
pgrep -af "train.py" | grep -v grep | wc -l   # should be 4 while the batch is running
date
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```
For each finished run (the launcher log prints `[run_id/40] <method> seed=<s> DONE`), note method+seed+exit. Count per wave — expect 4 DONE lines per ~3-hour wave, 10 waves total.

**Do NOT attach to the `headlines` tmux.** `capture-pane -p` reads the pane buffer without attaching; attaching risks accidental keystrokes into the launcher.

**Do NOT `kill -9` anything.** If a single run crashes, the launcher records it as FAILED in the log and moves on; the remaining 39 runs are unaffected. A single failed seed is recoverable post-batch (see §9).

### STEP 7 — Completion report [VM]
**Triggered when `logs/headlines_launcher.log` prints `All 40 jobs complete.` (or a failures list).**

Collect per-run stats and emit this table to LOCAL:

```
============================================================
HEADLINE BATCH COMPLETE — 40 runs, 4 methods x 10 seeds x 2M
============================================================

Launcher log tail:
<last 30 lines of logs/headlines_launcher.log>

Per-method summary (mean over 10 seeds):
  method      |  SPS mean  |  final_step mean  |  last_100ep_reward mean  |  failed
  ------------+-----------+-------------------+-------------------------+--------
  hcmarl      |   ...     |   2,000,000       |   ...                    |  0/10
  mappo       |   ...     |   2,000,000       |   ...                    |  0/10
  ippo        |   ...     |   2,000,000       |   ...                    |  0/10
  mappo_lag   |   ...     |   2,000,000       |   ...                    |  0/10

Per-seed table (40 rows):
  method,seed,final_step,wall_s,SPS,last_100ep_reward,run_state_present

Per-method plateau peek (cumulative_reward over last 400K of each seed):
  method      | mean_last_400K | mean_first_100 | plateau?
  (plateau = |last_400K - mean_across_200K_rolling| < 5% relative)

Aggregate wall time (newest summary.json minus oldest start_time): HH:MM:SS
Aggregate SPS (40 * 2_000_000 / aggregate_wall_seconds): ...
Failures (if any): [(method, seed, exit_code), ...]
Disk used by new logs/{method}/seed_*: <du -sh>
```

Extraction commands (VM Claude runs these and pastes the output):
```bash
python - <<'PY'
import json, csv, glob, os, statistics
methods = ["hcmarl", "mappo", "ippo", "mappo_lag"]
for m in methods:
    rows = []
    for seed in range(10):
        csv_path = f"logs/{m}/seed_{seed}/training_log.csv"
        summ_path = f"logs/{m}/seed_{seed}/summary.json"
        if not os.path.exists(csv_path):
            rows.append((m, seed, "MISSING"))
            continue
        with open(csv_path) as f:
            r = list(csv.DictReader(f))
        last = r[-1]
        last100 = [float(x["cumulative_reward"]) for x in r[-100:]]
        sps = float(last["global_step"]) / float(last["wall_time"]) if float(last["wall_time"]) > 0 else 0
        rows.append((m, seed, int(last["global_step"]), float(last["wall_time"]), sps, statistics.mean(last100)))
    print(m, rows)
PY
```
(If the wall_time column name in our CSV differs, grep the header first: `head -1 logs/hcmarl/seed_0/training_log.csv`. Existing columns per Batch D include `global_step`, `wall_time`, `cumulative_reward`, `cost_ema`, `entropy`, etc.)

Post the full table to LOCAL. Then STOP. Do NOT start anything else. Wait for §0.3.

---

## 6. Decision gates (who decides what)

| Decision | Who |
|---|---|
| Abort an individual crashed run | [VM] may report, but the launcher already skips it — no manual action needed |
| Abort the whole batch early | [LOCAL] — only if a systemic bug shows up (e.g. all 10 HCMARL seeds producing identical -inf rewards by step 100K) |
| Re-launch a failed seed post-batch | [LOCAL] decides, [VM] executes per §9 |
| 2M vs 3M-extension per method | [LOCAL] only, only after §7 table is in hand + plateau peek shows a method is still trending. Default is no extension — the buffer is for crash recovery, not per-method extension. |
| Destroy the E2E node | [USER] — only after [LOCAL] confirms all 40 CSVs are on the laptop. Not required to destroy immediately; node can stay alive to run ablations later if budget allows. |

---

## 7. Forbidden actions for VM-side Claude

- Destroying the node. That is always [USER].
- Editing `config/*.yaml` on disk. Configs are tracked files. If a change is needed, report it and [LOCAL] makes the edit and pushes.
- Editing `scripts/*.py` or any `.py` in `hcmarl/` on disk.
- Running `git add`, `git commit`, `git push`.
- Running `pip install` of anything not in this runbook.
- Running `scripts/run_ablations.py`. Today is headlines-only.
- Running `scripts/aggregate_learning_curves.py` on the VM before LOCAL has reviewed the §7 table. Aggregation is a LOCAL-side step after the scp.
- Interrupting the `headlines` tmux: no Ctrl-C, no `tmux kill-session`, no `pkill train.py`.
- Deleting anything under `/root/hcmarl_project/logs/` — the archived probe + 1M-watch directories are paper-critical.
- Proceeding past STEP 7 without the §0.3 paste.

---

## 8. Cost table (today, headlines only)

| Stage | Wallclock | Pre-GST cost | With 18% GST (if not on credits) |
|---|---|---|---|
| Pre-flight + pull + sanity (STEPs 1-4) | ~5 min | Rs 4 | Rs 5 |
| 40-run batch (STEP 5 launch → STEP 7 completion) | ~29.7 hr | Rs 1,455 | Rs 1,717 |
| Monitoring overhead (STEP 6, idle-billed by node) | included above | — | — |
| **Headline total** | **~30 hr** | **~Rs 1,459** | **~Rs 1,722** |

Remaining credits after headlines (expected): Rs 1,862 − Rs 1,459 = **~Rs 403 buffer**.

**If aggregate wall exceeds 33 hr** (i.e. observed per-seed SPS < 155, well below measured 188), [LOCAL] is alerted via the STEP 6 monitoring reports and may manually kill the `headlines` tmux to avoid overshooting the Rs 1,500 soft target. The per-run `--budget-inr 200` kill-switch is a per-run protector only, not a batch total.

---

## 9. Emergency procedures

| Event | Response |
|---|---|
| SSH drops mid-batch | Training is in tmux + nohup; it keeps going. Reconnect: `ssh -i ~/.ssh/id_ed25519 root@164.52.192.64`, then `tmux ls` (expect `headlines` still attached to a PID). VM Claude runs STEP 6 capture-pane to resume monitoring. |
| One seed crashes with a traceback | The launcher prints `[run_id/40] <method> seed=<s> FAILED (exit <rc>)` and moves on to the next pending run. No action during the batch. After §7 completion, [LOCAL] reviews and either (a) accepts 9/10 for that method and reports N=9 for that cell (caveat in the paper's stats section), or (b) re-launches just that (method, seed) via `python scripts/train.py --config config/<method>_config.yaml --method <method> --seed <s> --device cuda --run-name <method>` from a fresh tmux. Use `run_state.pt` auto-resume if the crashed run had reached ≥ 100K. |
| All 4 concurrent runs in a wave crash in the same place | Something systemic (GPU context exhaustion, disk full, OOM). STOP — `pgrep -af train.py` and capture-pane the tmux, send the full output to [LOCAL]. Do NOT restart blindly. |
| GPU OOM | L4 has 24 GB. Per-process footprint at N=6 workers is ~300 MB, so 4 concurrent ≈ 1.2 GB — OOM here is not an expected failure mode. If it happens: tmux capture, `nvidia-smi` snapshot, escalate. |
| Disk fills up | The 1M watch csv is 354 KB. 40 runs × ~700 KB each ≈ 28 MB for all CSVs. Even with checkpoints, 40 × 30 MB ≈ 1.2 GB. L4 plan has 250 GB. Disk fill is not expected; if observed, something is wrong with checkpoint naming — STOP and escalate. |
| Budget kill-switch trips (a run writes `checkpoint_budget_halt.pt`) | The per-run budget was set to Rs 200 (≈ 3.9 hr wall). Predicted wall is ~3 hr. A trip means that seed took 30% longer than predicted — probably a stuck MMICRL pretrain or a lazy-agent loop. Capture the summary.json (`"budget_tripped": true`), report to LOCAL, and let the launcher continue to the next queued run. |
| `headlines_launcher.log` shows the thread-cap print from T2b with a different number | T2b auto-caps at `total_vcpus // max_parallel`. On E2E L4 with 25 vCPUs + max_parallel=4 this should print `OMP/MKL/OPENBLAS_NUM_THREADS=6`. If it prints something else, the host reported a different vCPU count — tolerable, but note the value in the report. |
| MMICRL log lines say "MI collapse detected, falling back to theta floor" | Expected on Path G single-muscle profiles. Documented in MEMORY.md and in the config comment. NOT a bug. Do not intervene. |
| Claude Code rate limit mid-monitor | Training is not affected — it is in tmux + nohup. [USER] switches to a spare Anthropic account, re-launches `claude`, re-pastes §0.1. VM Claude re-reads RUNBOOK_HEADLINES and resumes monitoring from STEP 6. |
| Node disappears from E2E dashboard | Catastrophic — training + CSVs are gone. Re-launch a new L4 node, pay a fresh Rs ~1500, redo the whole batch. First check the dashboard billing page to confirm the disappearance wasn't an admin action. |

---

## 10. What happens after today

Tomorrow (or whenever sir's VM access is confirmed), the **25-run ablation batch** runs on sir's VM:

1. Night before: [LOCAL] edits `config/ablation_no_{nswf,ecbf,mmicrl}.yaml` from `total_steps: 3000000` to `total_steps: 2000000` so the ablation step budget matches today's headlines. Commit + push.
2. [USER] bootstraps sir's VM (apt install, python3.12-venv, torch from CUDA wheel index, `pip install -r requirements.txt`, `pytest -q` must print `508 passed, 1 skipped, 0 failed`).
3. [USER] launches a VM-side Claude on sir's VM and pastes an analogous §0.1 boot prompt pointing at `scripts/run_ablations.py --seeds 0 1 2 3 4 --max-parallel 4 --device cuda --budget-inr 200` (25 runs × 2M ≈ 18.6 hr on an L4-class GPU).
4. Ablation launcher uses the same matrix — the 5 rungs in `experiment_matrix.yaml::ablation.rungs` — so no config edits on run_ablations side are needed.
5. On completion: [USER] scps ablation CSVs down to the laptop under `logs/ablation_<rung>/seed_<s>/`.
6. [LOCAL] runs `python scripts/aggregate_learning_curves.py --out results/` on the laptop — walks both headline and ablation grids, emits IQM + stratified bootstrap 95% CIs at anchors {500K, 1M, 2M}.
7. [LOCAL] + [USER] review results, [LOCAL] drafts the TMLR figures + claim language.

If sir's VM path fails, fallback: [USER] pays out-of-pocket for ~19 hr of E2E L4 time (~Rs 1,100 incl. GST) to run ablations here. Stays within the Rs 4,118 running-total cap declared in the 2026-04-18 log entry.

---

## 11. End-of-today checklist (read before destroying the node)

- [ ] [LOCAL] edited + committed + pushed the 2M step-count change
- [ ] [VM] confirmed the pull landed the 2M commit (STEP 3)
- [ ] [VM] confirmed `cuda True NVIDIA L4` + 22+ GiB VRAM free (STEP 4)
- [ ] [VM] posted the "Launching 40 runs…" confirmation to LOCAL (STEP 5)
- [ ] [VM] posted at least two mid-batch snapshots (STEP 6)
- [ ] [VM] posted the §7 completion table to LOCAL (exit-success on the launcher log, or a failures list)
- [ ] [LOCAL] reviewed the completion table, decided no method needs 3M extension
- [ ] [USER] scp'd the 40 CSVs + 40 summary.json + (optionally) 40 run_state.pt files to `/c/Users/admin/Desktop/hcmarl_project/logs/` under `{hcmarl,mappo,ippo,mappo_lag}/seed_{0..9}/`
- [ ] [USER] verified the scp target exists + sizes are non-zero (`ls -lh` on the laptop side)
- [ ] [LOCAL] appended today's session to `logs/project_log.md` and committed
- [ ] [LOCAL] updated `MEMORY.md` with the SPS numbers, aggregate wall, per-method headline IQMs, any crashed seeds, any methodology caveats for the paper
- [ ] [USER] decides: destroy now, or keep alive for ablations (if sir's VM is not ready + budget allows)

Only after all eleven boxes are ticked is today "done."
