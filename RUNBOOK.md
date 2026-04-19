# RUNBOOK — HC-MARL probe + 1M watch on E2E L4 (2026-04-19)

> **If you are Claude Code running inside the rented VM terminal, this entire file is your briefing. Read every single line. Do not skim, do not jump sections, do not summarise in your head and skip ahead. Read top to bottom. The §0.1 boot prompt you were pasted already told you what to do when you finish — follow it exactly. Do not wait for any separate instruction.**

---

## 0. Paste-in prompts (for the user)

Today uses **two Claude agents**:

- **LOCAL** — runs on the laptop (this session). Holds full MEMORY.md context. Owns scope decisions, git state, RUNBOOK edits, plateau interpretation.
- **VM** — runs as Claude Code inside the rented E2E L4 VM. Owns bootstrap, training launch, monitoring, reporting.

The user (Aditya) sits at the laptop, talks to LOCAL here, and pastes ready-made blocks from LOCAL into the VM terminal when instructed. There is no second local session today; a third Anthropic account is kept as a standby rate-limit backup only (not an active agent).

Below are the **three** prompts the user will paste into VM-side Claude across the session:

- **§0.1 Boot prompt** — pasted once, right after the manual `git clone` in STEP 6. VM Claude then runs continuously through STEP 14a and stops.
- **§0.2 Post-14a continuation** — pasted after LOCAL has interpreted the probe SPS. Tells VM Claude to run STEPs 14c + 14d and stop.
- **§0.3 Post-14d close-out** — pasted after LOCAL has interpreted the plateau verdict. Tells VM Claude measurements are done and the user will do STEPs 16-17 manually.

### 0.1 Boot prompt (paste into VM-side Claude Code immediately after `claude` launches)

```
You are the VM-side Claude for the HC-MARL execution session on an E2E L4 GPU node.
This paste IS your full starting instruction — no further instruction is coming from
me for roughly the next 25-30 minutes. Execute continuously.

Your complete briefing is at /root/hcmarl_project/RUNBOOK.md on this VM. Read every
line of that file top to bottom. Do not skim, do not skip, do not summarise.

After you finish reading, execute STEPs 8 through 14a of the RUNBOOK continuously,
in order, without asking me for confirmation between steps:

  STEP 8   hardware sanity check (nvidia-smi, lscpu, free -h, df -h)
  STEP 9   create python3.12 venv at /root/hcmarl_project/venv
  STEP 10  pip install torch from the CUDA wheel index (cu124, fall back to cu121)
  STEP 11  pip install -r requirements.txt, verify torch is still the CUDA build
  STEP 12  pytest -q   (pass criterion: 508 passed, 1 skipped, 0 failed)
  STEP 13  create tmux session `probe` and activate the venv inside it
  STEP 14a launch the 500K probe inside the `probe` tmux, wait ~15-20 min for it
           to finish, then extract Wallclock / Final step / SPS / lazy_agent events
           / MMICRL MI collapse warnings as the table in §5 STEP 14a.

When STEP 14a finishes, post the full result table and then STOP. Do NOT start the
1M watch. Wait for my next paste (§0.2) before continuing.

If any of STEP 8-13 fails in a way the RUNBOOK does not pre-authorise a fix for
(hardware below spec, pytest red, cuda:False after both cu124 and cu121 attempts,
any training crash), STOP immediately and tell me what failed. Do not edit tracked
config or source files to "fix" things — those edits are owned by local-primary.

Begin now. When you have finished reading RUNBOOK.md, reply with exactly
`Read. Executing from STEP 8.` and start.
```

### 0.2 Post-14a continuation prompt (paste after LOCAL interprets the probe SPS)

```
SPS is acceptable. Proceed with STEPs 14c and 14d of the RUNBOOK:

  STEP 14c  launch the 1M watch-curve in a NEW tmux session called `watch`
            (do NOT reuse the `probe` tmux — keep that as a record).
            Wait ~30-60 min for it to finish.
  STEP 14d  run `python scripts/check_plateau.py logs/hcmarl/watch_1m/training_log.csv`
            followed by `echo "exit=$?"`, then print the header and last 5 rows
            of logs/hcmarl/watch_1m/training_log.csv.

When STEP 14d finishes, post:
  - the full stdout of check_plateau.py
  - the exit code line (exit=N)
  - the CSV header + last 5 rows

Then STOP. Do not touch anything else. Wait for my next paste (§0.3).
```

### 0.3 Post-14d close-out prompt (paste after LOCAL interprets the plateau verdict)

```
Measurements are complete. Your work is done for this session.

The user will now execute STEP 16 (scp the CSVs down to the laptop) and STEP 17
(destroy the E2E node) manually from the laptop — those are [USER] steps, not yours.

Do NOT destroy the node. Do NOT delete anything under /root/hcmarl_project/logs/.
Do NOT start any more training. Do NOT make any git commits.

Final acknowledgement: reply with exactly
  `Measurements complete. Standing by. Local-primary will close the session.`
and then wait. If the user asks a specific follow-up question answer it factually;
otherwise stay quiet until SSH is severed.
```

---

## 1. Two-agent division of labour

Do not cross these lines. Two agents editing the same file or executing the same action is worse than one agent with perfect information.

| Agent | Owns | Forbidden |
|---|---|---|
| **LOCAL** (this session) | Scope decisions (3M vs 5M, seed counts, kill/continue calls), git commits, RUNBOOK.md edits, MEMORY.md, logs/project_log.md, plateau interpretation, paper-side writing | Direct SSH execution on the VM. LOCAL hands the user ready-to-paste blocks; VM-side Claude or the user executes them. |
| **VM** (Claude Code on the rented L4) | Bootstrap execution on the VM, launching training in tmux, monitoring live logs, running `scripts/check_plateau.py`, inspecting CSVs on the remote, reporting numbers back | Scope decisions, destroying the E2E node, editing tracked configs on disk, editing any `.py` file on disk, making git commits on the VM, writing to RUNBOOK.md, running anything past STEP 14d without the §0.3 paste. |
| **USER** (Aditya) | E2E dashboard clicks (Launch, Destroy, top-up), SSH-ing into the VM, `git clone`-ing the repo before Claude Code launches, pasting between sessions, supplying the node IP, final approval on any spend, destroying the node after STEP 16 verified | Running training outside tmux. Destroying the node before `scp` pulls the CSVs. Editing files directly on the VM — ask LOCAL. |

When in doubt about ownership: LOCAL holds MEMORY.md and is the brain. VM is the hands. USER is the physical interface to the dashboards and terminals.

---

## 2. Project context (compressed for a cold-start agent)

- **Project:** HC-MARL — Human-Centric Multi-Agent RL for fatigue-aware warehouse task allocation. Author: Aditya Maiti. Target: TMLR (OpenReview, rolling).
- **Repo:** `github.com/ADITYA-WORK-MAITI/hcmarl-project`, branch `master`, tip published to origin is the 2026-04-19 pre-execution readiness commit.
- **Core modules:** 3CC-r fatigue ODE (Xia/Frey-Law 3-compartment), ECBF safety filter (analytical QP, inline), Nash Social Welfare allocator, MMICRL pretrainer (normalizing flows), MAPPO/IPPO/MAPPO-Lag baselines, PettingZoo-style warehouse env.
- **Workload shape:** 95-99 % CPU-bound, GIL-bound single-process PyTorch loop, tiny nets (~250K params across 6 agents), GPU is >=L4 for determinism + compliance, not throughput.
- **Tests:** 508 passed, 1 skipped, 0 failed on Python 3.12 + local venv. Must stay green on any new host.
- **Scope for the 45-run batch:** 3M steps x {4 headline methods x 5 seeds + 5 ablation rungs x 5 seeds} = 45 runs, ~1.5 hr each, ~67.5 hr total. Locked in `config/experiment_matrix.yaml`.
- **TMLR scaling-study:** NOT required. Confirmed via prior Research Mode pass (2026-04-18).

---

## 3. Today's scope (measurement only)

Today we do **two runs on E2E**, read two numbers, destroy the node, then tomorrow move to sir's VM for the 45-run batch. That is all.

1. **500K probe** with `config/probe_500k.yaml` -> measures real SPS on this L4 host -> tells us if 3M-per-run is budgetable.
2. **1M watch-curve** with `config/watch_1m.yaml` -> measures reward-curve shape -> tells us if 3M steps is enough or we need 5M.
3. **Plateau check** with `scripts/check_plateau.py` on the 1M CSV -> emits PLATEAU / STILL_CLIMBING / REGRESSING verdict.
4. `scp` the CSVs down to the laptop.
5. **Destroy the E2E node.**

**Spend ceiling today: Rs 100.** Anything past Rs 150 means something went wrong — stop and escalate to LOCAL.

Do not launch any of the 45-run batch on E2E today. That happens tomorrow on sir's VM.

---

## 4. Current state (as of this session)

- Repo tip with today's pre-execution readiness commit is already on GitHub — `git clone` pulls the execution-ready state.
- `requirements.txt` fixed: `torch>=2.6.0` and `gymnasium>=0.29.0` uncommented. Dead `pettingzoo` reference removed.
- `config/probe_500k.yaml` — 500K probe config (single seed, faithful pipeline).
- `config/watch_1m.yaml` — 1M watch config (single seed, used for the plateau check).
- `scripts/check_plateau.py` — plateau verdict script (exit codes 0 PLATEAU / 1 STILL_CLIMBING / 2 REGRESSING / 3 NOT_ENOUGH_DATA).
- `config/experiment_matrix.yaml` — headline seeds locked at `[0, 1, 2, 3, 4]`.
- Rs 2,000 E2E infra credits intact, minus Rs 17.15 burned yesterday on an aborted bootstrap.
- SSH key `hcmarl-laptop` (fingerprint `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILk/a/yMwFx8Vzo5LNlishRe2LK3jCXisYVVTNC4XWmf e2e-hcmarl`) already uploaded to the E2E account.

---

## 5. Execution plan

Ownership tag on each step: **[USER]** = Aditya does it manually, **[LOCAL]** = LOCAL Claude (this session), **[VM]** = VM-side Claude (the paste-me-to-start session).

**Ordering invariant:** STEPs 1-7 are all [USER] and must complete before VM-side Claude is even launched. The §0.1 boot prompt is pasted as the first action of STEP 7, and VM-side Claude then owns STEPs 8-14a continuously.

### STEP 0 — Repo hygiene
**[LOCAL] — already done.** Today's pre-execution readiness commit is on origin/master. The `git clone` in STEP 6 will pull it.

### STEP 1 — Get node IP from E2E
**[USER].** Login -> MyAccount -> Compute -> GPU -> Launch GPU -> Nvidia-L4 -> Plan #1 (25 vCPU / 110 GB / 24 GB VRAM / 250 GB SSD) -> On-Demand Rs 49/hr -> attach SSH key `hcmarl-laptop` -> tick "Disable Password-based SSH login" -> Launch -> wait for Running -> **copy IP**.

If "This GPU plan is temporarily not available" appears, refresh and retry. Yesterday it cleared on the second attempt.

Note: E2E uses SSH-key auth only; there is no username/password to collect. The "username" is `root`. The "authentication" is your `~/.ssh/id_ed25519` matching the uploaded public key.

### STEP 2 — First SSH handshake
**[USER] in Git Bash on the Windows laptop:**
```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@<NEW_IP>
```
This drops you into a `root@hostname:~#` shell on the VM. STEPs 3-7 happen from inside this shell.

### STEP 3 — SKIP (ssh-copy-id not needed)
Your `~/.ssh/id_ed25519.pub` is already registered as the `hcmarl-laptop` key on E2E. Regenerating would orphan the registered key. Passwordless SSH already works from STEP 2.

### STEP 4 — SKIP (no VS Code)
VS Code Remote-SSH adds no value for a 60-minute measurement session. Git Bash terminal only.

### STEP 5 — Install OS-level tools on the VM
**[USER] inside the VM shell from STEP 2:**
```bash
apt-get update -q
apt-get install -y -q curl ca-certificates git tmux htop python3.12-venv python3-pip nodejs npm
# nvtop is nice-to-have; skip silently if not in this Ubuntu image
apt-get install -y -q nvtop || true
git --version && tmux -V && python3.12 --version && node --version && npm --version
```
If Ubuntu 24.04 doesn't have `nvtop` in apt, VM Claude can `pip install nvitop` inside the venv later. Not a blocker.

### STEP 6 — Git clone the repo (must happen BEFORE Claude Code launches)
**[USER] inside the VM shell:**
```bash
cd /root
git clone https://github.com/ADITYA-WORK-MAITI/hcmarl-project.git hcmarl_project
cd hcmarl_project
git log --oneline -5
ls RUNBOOK.md && wc -l RUNBOOK.md
```
The last two commands confirm `/root/hcmarl_project/RUNBOOK.md` exists — because VM-side Claude will read it as its first action in STEP 8. If RUNBOOK.md is missing, the clone went wrong; re-run the clone before proceeding.

### STEP 7 — Install + launch Claude Code, paste §0.1 boot prompt
**[USER] inside the VM shell, in `/root/hcmarl_project`:**
```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version
claude   # starts the CLI; log in with the second Anthropic account
```
After `claude` launches and the login completes, paste the **§0.1 boot prompt** verbatim. VM-side Claude will reply `Read. Executing from STEP 8.` and begin.

From this point on, STEPs 8 through 14a are owned by **[VM]** and run continuously.

### STEP 8 — Hardware sanity check (was 5.1)
**[VM]:**
```bash
nvidia-smi | head -25
lscpu | grep -E "Model name|Thread|Socket|CPU MHz|CPU max MHz"
free -h
df -h /root
```
Report back to the user:
- GPU model (must be `NVIDIA L4`)
- VRAM free (must be >= 23 GiB)
- Driver + CUDA version from `nvidia-smi` header
- Number of threads x sockets
- Max CPU MHz (single-thread turbo matters for GIL-bound work)
- RAM free (must be >= 100 GiB)
- Disk free on /root (must be >= 200 GiB)

If any of these are below spec, STOP and escalate to LOCAL.

### STEP 9 — Create Python venv (NOT miniconda)
**[VM]:**
```bash
cd /root/hcmarl_project
python3.12 -m venv venv
source venv/bin/activate
python --version   # expect 3.12.x
pip install -q -U pip wheel
```

### STEP 10 — Install torch from the CUDA wheel index FIRST
**[VM]:**
```bash
# nvidia-smi header shows CUDA driver 12.x. cu124 wheels are forward-compatible.
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```
**Pass criteria:** `cuda available: True` and `device name: NVIDIA L4`.

If `cuda available: False`, retry once with the cu121 wheel index:
```bash
pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```
If cu121 also reports False, STOP and escalate — driver is too old.

### STEP 11 — Install the rest of requirements
**[VM]:**
```bash
pip install -q -r requirements.txt
pip list | grep -iE "^(torch|gymnasium|numpy|cvxpy|osqp|scipy|pyyaml|matplotlib|wandb)"
```
Verify torch is still the CUDA build (version string contains `+cu124` or `+cu121`). If pip swapped it for the CPU wheel, re-run STEP 10 with `--force-reinstall`.

### STEP 12 — Pytest sanity
**[VM]:**
```bash
pytest -q 2>&1 | tail -20
```
**Pass criteria:** `508 passed, 1 skipped, 0 failed` exactly. One known harmless CVXPY warning from `ecbf_filter.py:309` ("Solution may be inaccurate") is expected in 8 pipeline tests and does not fail them.

If any test fails, STOP and escalate. Do not start the probe on a broken env.

### STEP 13 — tmux session for the probe
**[VM]:**
```bash
tmux new -d -s probe
tmux send-keys -t probe "cd /root/hcmarl_project && source venv/bin/activate" Enter
```
The `-d` flag creates the tmux detached so the VM-Claude session doesn't get absorbed into the tmux. `send-keys` lets VM Claude control tmux panes without attaching.

### STEP 14 — The measurements (split into 14a-14d, with two explicit hand-off pauses)

#### 14a — 500K probe [VM]
**Inside the `probe` tmux session:**
```bash
tmux send-keys -t probe "time python scripts/train.py --config config/probe_500k.yaml --method hcmarl --seed 0 --device cuda --run-name probe_500k 2>&1 | tee logs/probe_500k.log" Enter
```
Wait ~15-20 minutes. Monitor either way:
```bash
tail -n 50 /root/hcmarl_project/logs/probe_500k.log
# or
tmux capture-pane -t probe -p | tail -n 50
```

**When it finishes, extract:**
- Wallclock from the `real Xm Y.Zs` line of `time`.
- Step count from the last `[Ep ... | Step ...]` line (should reach ~500 000).
- lazy_agent events: `grep -c lazy_agent /root/hcmarl_project/logs/probe_500k.log`
- MMICRL MI collapse: `grep -ic "MI collapse" /root/hcmarl_project/logs/probe_500k.log`

**Report to LOCAL (VM Claude posts this and STOPS):**
| Metric | Value |
|---|---|
| Wallclock | Xm Y.Zs |
| Final step | ~500 000 |
| SPS = 500000 / wallclock_seconds | compute |
| lazy_agent events | count |
| MMICRL MI collapse | yes/no |

**HAND-OFF POINT #1.** VM Claude stops here and waits for the **§0.2 continuation prompt**.

#### 14b — Decision gate [LOCAL]
Given SPS, LOCAL interprets:

| SPS | 3M/run | 45 runs | Verdict |
|---|---|---|---|
| >= 555 | <= 1.5 hr | <= 67.5 hr | 3M comfortable |
| 415-555 | 1.5-2.0 hr | 67-90 hr | 3M workable, tight at 90 |
| 333-415 | 2.0-2.5 hr | 90-112 hr | 3M needs seed cut or sir's VM must be faster |
| < 333 | > 2.5 hr | > 112 hr | 3M mandatory, consider cutting ablation seeds |

If SPS < 300, LOCAL may pause before the 1M watch to replan scope. Otherwise LOCAL tells the user "proceed", and the user pastes §0.2 into VM Claude.

#### 14c — 1M watch-curve [VM]
**Triggered by the §0.2 paste. In a new tmux session so the probe tmux stays as a record:**
```bash
tmux new -d -s watch
tmux send-keys -t watch "cd /root/hcmarl_project && source venv/bin/activate" Enter
tmux send-keys -t watch "time python scripts/train.py --config config/watch_1m.yaml --method hcmarl --seed 0 --device cuda --run-name watch_1m 2>&1 | tee logs/watch_1m.log" Enter
```
Wait ~30-60 minutes depending on SPS.

#### 14d — Plateau verdict [VM]
**After 14c finishes:**
```bash
python scripts/check_plateau.py logs/hcmarl/watch_1m/training_log.csv
echo "exit=$?"
echo "header:"
head -1 /root/hcmarl_project/logs/hcmarl/watch_1m/training_log.csv
echo "last 5 rows:"
tail -5 /root/hcmarl_project/logs/hcmarl/watch_1m/training_log.csv
```
**Report to LOCAL:** full stdout, exit code, CSV header, last 5 rows.

- Exit 0 / PLATEAU -> 3M steps is enough. Option 4 locked at 5 seeds x 4 headline + 5 seeds x 5 ablation = 45 runs x 3M.
- Exit 1 / STILL_CLIMBING -> 5M needed. Scope needs replan.
- Exit 2 / REGRESSING -> abort. Debug before the batch.
- Exit 3 / NOT_ENOUGH_DATA -> run didn't reach 500K. Investigate a crash.

**HAND-OFF POINT #2.** VM Claude stops here and waits for the **§0.3 close-out prompt**.

### STEP 15 — Monitor (covered by 14a/14c)
Two techniques, both valid:
- **tmux detach + reattach:** `Ctrl-b d` to detach, `tmux attach -t probe` (or `watch`) to reattach. VM Claude uses `tmux capture-pane -p` instead.
- **rsync / scp the CSVs to the laptop** for offline analysis. See STEP 16.

### STEP 16 — Pull CSVs to the laptop BEFORE destroy
**[USER] in Git Bash on the Windows laptop (not inside SSH):**
```bash
scp -i ~/.ssh/id_ed25519 -r root@<NEW_IP>:/root/hcmarl_project/logs/hcmarl \
    /c/Users/admin/Desktop/hcmarl_project/logs/
```
The two run directories `probe_500k/` and `watch_1m/` (each with `training_log.csv` + `run_state.pt` + checkpoints) land under `logs/hcmarl/` on the laptop.

Verify on the laptop:
```bash
ls /c/Users/admin/Desktop/hcmarl_project/logs/hcmarl/probe_500k/
ls /c/Users/admin/Desktop/hcmarl_project/logs/hcmarl/watch_1m/
```
Both should show `training_log.csv` non-empty.

### STEP 17 — Destroy the E2E node
**[USER] in the E2E dashboard:**
1. MyAccount -> Compute -> Manage Nodes -> your L4 node -> `Destroy`.
2. Confirm the destroy dialog.
3. Refresh the Nodes page — state should be gone within 30 seconds.
4. Open Billing -> verify the running charge for this node has stopped.

**Do not destroy** until STEP 16 has completed and the CSVs are safely on the laptop. Once destroyed, the VM disk is unrecoverable.

---

## 6. Decision gates (who decides what)

| Decision | Who |
|---|---|
| Abort probe early if OOM / crash | [VM] Claude, report immediately |
| SPS too slow -> abort 1M watch | [LOCAL] based on 14a report |
| 3M vs 5M per run | [LOCAL] based on 14d plateau verdict |
| Final seed count (5 vs 3 vs 10) | [LOCAL] based on 14d + intra-run noise |
| Destroy the node | [USER] — only after [LOCAL] confirms measurements captured |
| Handoff to sir's VM tomorrow | [USER] + [LOCAL] together |
| Fallback to remaining E2E credit if sir's VM fails | [LOCAL] recommends, [USER] approves |

---

## 7. Forbidden actions for VM-side Claude

- Destroying the node. That is always [USER].
- Editing `config/*.yaml` on disk. Configs are tracked files. If a change is needed, report it and [LOCAL] makes the edit and pushes.
- Editing `scripts/train.py` or any `.py` in `hcmarl/` or `scripts/` on disk.
- Running `git add`, `git commit`, `git push`.
- Running `pip install` of anything not in this runbook without asking.
- Running the 45-run batch (`scripts/run_baselines.py`, `scripts/run_ablations.py`). Today is measurement-only.
- Modifying `logs/hcmarl/<run>/run_state.pt` or any checkpoint file.
- Deleting anything under `/root/hcmarl_project/logs/` before STEP 16 finishes.
- Proceeding past STEP 14a without the §0.2 paste, or past STEP 14d without the §0.3 paste.

---

## 8. Cost table (today)

| Stage | Wallclock | Pre-GST cost | With 18% GST |
|---|---|---|---|
| Bootstrap (STEPs 5-12) | ~6 min | Rs 5 | Rs 6 |
| 500K probe (14a) | ~15-20 min | Rs 12-16 | Rs 14-19 |
| 1M watch (14c) | ~30-45 min | Rs 25-37 | Rs 30-44 |
| Plateau check + scp + destroy (14d-17) | ~10 min | Rs 8 | Rs 10 |
| **Total** | **~65-80 min** | **~Rs 55-70** | **~Rs 65-80** |

Running total across both days: Rs 17.15 (yesterday aborted) + Rs 80 today = ~Rs 97. Leaves ~Rs 1,903 of the Rs 2,000 E2E free credits for the sir-VM fallback path.

---

## 9. Emergency procedures

| Event | Response |
|---|---|
| SSH drops mid-run | Training is in tmux; it keeps going. Reconnect: `ssh -i ~/.ssh/id_ed25519 root@<IP>` then `tmux attach -t probe` or `watch`. |
| Bootstrap heredoc hangs at `>` prompt | Your `EOF` line had trailing whitespace. Type `EOF` on its own and Enter. If still stuck, `Ctrl-C` and re-run the heredoc. |
| `cuda available: False` after torch install | STEP 10 retry block with cu121. If still CPU-only, driver is too old — report `nvidia-smi` CUDA version to [LOCAL]. |
| pytest fails with < 508 passed | Stop. Do not proceed. Paste the last 30 lines of pytest output to [LOCAL]. |
| Training crashes mid-probe | tmux pane has the traceback. Capture it: `tmux capture-pane -t probe -p > /tmp/probe_crash.txt` then `tail -100 /tmp/probe_crash.txt`. Escalate to [LOCAL]. |
| OOM on GPU | Unlikely at n_workers=6 on 24 GB L4. If it happens: capture tmux pane, escalate, do not retry blindly. |
| Claude Code on VM hits rate limit | User switches to the standby third Anthropic account, re-launches `claude`, re-pastes §0.1 (VM Claude re-reads RUNBOOK and resumes from wherever it left off based on tmux state). |
| E2E billing looks higher than expected | Stop all runs. [USER] checks Billing page. Escalate to [LOCAL] for recomputation. |
| Node gets destroyed accidentally before STEP 16 | The CSVs are gone. Relaunch a fresh L4 node, re-bootstrap, re-run probe + 1M. Budget impact ~Rs 80 more. |

---

## 10. What happens after today

Tomorrow (2026-04-20) the 45-run batch runs on sir's VM (assuming sir has sent GPU specs + access details). The handoff is:

1. [USER] receives credentials from sir.
2. [USER] starts a new VM-side Claude session on sir's VM (standby third Anthropic account), paste §0.1 boot prompt with the path updated to wherever the clone lands.
3. [VM] executes STEPs 8-12 on the new host (bootstrap).
4. [VM] launches `scripts/run_baselines.py` and `scripts/run_ablations.py` in tmux. Both read `config/experiment_matrix.yaml` and respect today's 3M-or-5M decision.
5. Runs execute ~3 days unattended.
6. On completion, [VM] runs `scripts/aggregate_learning_curves.py --out results/`.
7. [LOCAL] reviews results, [USER] prepares for 2026-04-28 internal evaluation.

If sir's VM path fails (no GPU, wrong spec, access broken, etc.), fallback plan:
- [USER] launches a new E2E L4 node using remaining credits (~33 hr usable).
- That covers the headline grid alone (4 methods x 5 seeds x 1.5 hr = 30 hr). Ablations slip to a second session or get cut to 3 seeds.

---

## 11. End-of-today checklist (read before destroying the node)

- [ ] Probe SPS reported to [LOCAL]
- [ ] 1M watch plateau verdict reported to [LOCAL] with exit code
- [ ] Last 5 rows of watch_1m CSV reported to [LOCAL]
- [ ] §0.3 close-out paste sent to VM Claude (VM now stood down)
- [ ] [USER] ran `scp` pull in STEP 16
- [ ] [USER] verified CSVs exist on laptop (`ls logs/hcmarl/probe_500k/` and `ls logs/hcmarl/watch_1m/`)
- [ ] [LOCAL] confirmed scope decision for tomorrow (3M or 5M, seed counts)
- [ ] [USER] destroys the node in E2E dashboard
- [ ] [USER] verifies billing charge has stopped
- [ ] [LOCAL] appends today's session to `logs/project_log.md` and commits

Only after all ten boxes are ticked is today "done."
