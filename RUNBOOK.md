# RUNBOOK — HC-MARL probe + 1M watch on E2E L4 (2026-04-19)

> **If you are Claude Code running inside the rented VM terminal, this entire file is your briefing. Read every single line. Do not skim, do not jump sections, do not summarise in your head and skip ahead. Read top to bottom, then stop and say `Read.` Wait for the user's first instruction before acting.**

---

## 0. Paste-in prompts (for the user)

Three Claude sessions are coordinating today. The user will paste ONE of these at the top of each session.

### 0.1 VM-side Claude boot prompt (paste into Claude Code running in the VM terminal, immediately after `git clone` finishes)

```
You are the VM-side Claude for the HC-MARL execution session. Your full briefing
is at /root/hcmarl_project/RUNBOOK.md on this VM. Read every line of that file
from top to bottom. Do not skim, do not skip, do not summarise. When you reach
the end, reply with exactly `Read.` and wait for my first instruction.
```

### 0.2 Local-secondary Claude boot prompt (optional third session on the laptop, for spill-over work)

```
You are the local-secondary Claude. Your role is parallel work that is
independent of the VM session: reading returned CSVs, generating plots,
drafting paper paragraphs. Do not edit repo files without my explicit say-so
— the local-primary Claude owns git state. When I send you a task, confirm
you understand the scope and then execute.
```

### 0.3 Local-primary Claude (this session, me) needs no prompt — already has full MEMORY.md context.

---

## 1. Three-agent division of labour

Do not cross these lines. Two agents editing the same file or executing the same action is worse than one agent with perfect information.

| Agent | Owns | Forbidden |
|---|---|---|
| **Local-primary** (me) | Scope decisions (3M vs 5M, seed counts, kill/continue calls), git commits, RUNBOOK.md edits, MEMORY.md, logs/project_log.md, scp pulls from laptop-side, plateau interpretation | Direct SSH execution on the VM. I hand the user ready-to-paste blocks; VM-side Claude or the user executes them. |
| **VM-side** | Bootstrap execution on the VM, launching training in tmux, monitoring live logs, running scripts/check_plateau.py, inspecting CSVs on the remote, reporting numbers back | Scope decisions, destroying the E2E node, editing tracked configs on disk, making git commits on the VM, writing to RUNBOOK.md. |
| **Local-secondary** (optional) | CSV analysis of files already pulled to the laptop, figure generation, paper drafting | Any edit to the repo working tree. Any git command. Any file the primary has touched in the last hour. |
| **User** (Aditya) | E2E dashboard clicks (Launch, Destroy, top-up), pasting between sessions, supplying the node IP, final approval on any spend, approving the sir-VM handoff tomorrow | Running training outside tmux. Destroying the node before `scp` pulls the CSVs. |

When in doubt about ownership: the agent that holds MEMORY.md context is the brain. The agent in the VM is the hands.

---

## 2. Project context (compressed for a cold-start agent)

- **Project:** HC-MARL — Human-Centric Multi-Agent RL for fatigue-aware warehouse task allocation. Author: Aditya Maiti. Target: TMLR (OpenReview, rolling).
- **Repo:** `github.com/ADITYA-WORK-MAITI/hcmarl-project`, branch `master`, tip `23e2b73` (2026-04-19 pre-execution readiness).
- **Core modules:** 3CC-r fatigue ODE (Xia/Frey-Law 3-compartment), ECBF safety filter (analytical QP, inline), Nash Social Welfare allocator, MMICRL pretrainer (normalizing flows), MAPPO/IPPO/MAPPO-Lag baselines, PettingZoo-style warehouse env.
- **Workload shape:** 95–99 % CPU-bound, GIL-bound single-process PyTorch loop, tiny nets (~250K params across 6 agents), GPU is ≥L4 for determinism + compliance, not throughput.
- **Tests:** 508 passed, 1 skipped, 0 failed on Python 3.12 + local venv. Must stay green on any new host.
- **Scope for the 45-run batch:** 3M steps × {4 headline methods × 5 seeds + 5 ablation rungs × 5 seeds} = 45 runs, ~1.5 hr each, ~67.5 hr total. This is locked in `config/experiment_matrix.yaml`.
- **TMLR scaling-study:** NOT required. Confirmed via prior Research Mode pass (2026-04-18).

---

## 3. Today's scope (measurement only)

Today we do **two runs on E2E**, read two numbers, destroy the node, then tomorrow move to sir's VM for the 45-run batch. That is all.

1. **500K probe** with `config/probe_500k.yaml` → measures real SPS on this L4 host → tells us if 3M-per-run is budgetable.
2. **1M watch-curve** with `config/watch_1m.yaml` → measures reward-curve shape → tells us if 3M steps is enough or we need 5M.
3. **Plateau check** with `scripts/check_plateau.py` on the 1M CSV → emits PLATEAU / STILL_CLIMBING / REGRESSING verdict.
4. `scp` the CSVs down to the laptop.
5. **Destroy the E2E node.**

**Spend ceiling today: ₹100.** Anything past ₹150 means something went wrong — stop and escalate to local-primary.

Do not launch any of the 45-run batch on E2E today. That happens tomorrow on sir's VM.

---

## 4. Current state (as of 12:30 PM IST, 2026-04-19)

- Repo tip `23e2b73` already on GitHub — `git clone` pulls the execution-ready state.
- `requirements.txt` fixed: `torch>=2.6.0` and `gymnasium>=0.29.0` uncommented. Dead `pettingzoo` reference removed.
- `config/probe_500k.yaml` — 500K probe config.
- `config/watch_1m.yaml` — 1M watch config.
- `scripts/check_plateau.py` — plateau verdict script.
- `config/experiment_matrix.yaml` — headline seeds locked at `[0, 1, 2, 3, 4]`.
- ₹2,000 E2E infra credits intact. ₹17.15 burned yesterday on aborted bootstrap.
- SSH key `hcmarl-laptop` (fingerprint `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILk/a/yMwFx8Vzo5LNlishRe2LK3jCXisYVVTNC4XWmf e2e-hcmarl`) already uploaded to the E2E account.

---

## 5. Execution plan — follows PLAN.txt numbering with refinements applied

Ownership tag on each step: **[USER]** = Aditya does it, **[LOCAL]** = local-primary Claude, **[VM]** = VM-side Claude.

### STEP 0 — Repo hygiene
**[LOCAL] — already done.** Commit `23e2b73` pushed to origin/master. The VM's `git clone` will pull this.

### STEP 1 — Get node IP from E2E
**[USER].** Login → MyAccount → Compute → GPU → Launch GPU → Nvidia-L4 → Plan #1 (25 vCPU / 110 GB / 24 GB VRAM / 250 GB SSD) → On-Demand ₹49/hr → attach SSH key `hcmarl-laptop` → tick "Disable Password-based SSH login" → Launch → wait for Running → **copy IP**.

If "This GPU plan is temporarily not available" appears, refresh and retry. Yesterday it cleared on the second attempt.

Note: E2E uses SSH-key auth only; there is no username/password to collect. The "username" is `root`. The "authentication" is your `~/.ssh/id_ed25519` matching the uploaded public key.

### STEP 2 — First SSH handshake
**[USER] in Git Bash on the Windows laptop:**
```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@<NEW_IP>
```
This drops you into a `root@hostname:~#` shell on the VM. From here, the VM terminal is where STEP 5+ happens.

### STEP 3 — SKIP (ssh-copy-id not needed)
Your `~/.ssh/id_ed25519.pub` is already registered as the `hcmarl-laptop` key on E2E. Regenerating would orphan the registered key. Passwordless SSH already works from step 2.

### STEP 4 — SKIP (no VS Code)
VS Code Remote-SSH adds no value for a 60-minute measurement session. Git Bash terminal only.

### STEP 5 — VM terminal open, install Claude Code
**[USER] inside the VM shell from STEP 2:**
```bash
# Install Node.js (required for Claude Code)
apt-get update -q && apt-get install -y -q curl ca-certificates nodejs npm

# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version
```
Then run `claude` to start the CLI, log in with your second Anthropic account, and paste the boot prompt from §0.1 above.

Once VM-side Claude replies `Read.`, all further steps 5.1 through 15 are owned by **[VM]**.

### STEP 5.1 — Hardware sanity check
**[VM]:**
```bash
nvidia-smi | head -25
lscpu | grep -E "Model name|Thread|Socket|CPU MHz|CPU max MHz"
free -h
df -h /root
```
Report back to the user:
- GPU model (must be `NVIDIA L4`)
- VRAM free (must be ≥ 23 GiB)
- Driver + CUDA version from `nvidia-smi` header
- Number of threads × sockets
- Max CPU MHz (single-thread turbo matters for GIL-bound work)
- RAM free (must be ≥ 100 GiB)
- Disk free on /root (must be ≥ 200 GiB)

If any of these are below spec, stop and escalate to local-primary.

### STEP 6 — Install monitoring tools
**[VM]:**
```bash
apt-get install -y -q htop nvtop tmux
htop --version && nvtop --version && tmux -V
```
If `nvtop` is not in the default apt index (Ubuntu 24.04 should have it), fall back to `pip install nvitop` after the venv is set up in STEP 8.

### STEP 7 — Git clone the repo
**[VM]:**
```bash
cd /root
git clone https://github.com/ADITYA-WORK-MAITI/hcmarl-project.git hcmarl_project
cd hcmarl_project
git log --oneline -5
```
Verify the top commit is `23e2b73 Pre-execution readiness: requirements unblock + probe/plateau + runbook`. If it is not, stop and escalate — the user may need to force-refresh GitHub caches or the repo went private.

### STEP 8 — Create Python venv (NOT miniconda)
**[VM]:**
```bash
apt-get install -y -q python3.12-venv python3-pip
cd /root/hcmarl_project
python3.12 -m venv venv
source venv/bin/activate
python --version   # expect 3.12.x
pip install -q -U pip wheel
```

### STEP 9 — Install torch from the CUDA wheel index FIRST
**[VM]:**
```bash
# `nvidia-smi` header shows CUDA driver 12.x. cu124 wheels are
# forward-compatible. If this pip install fails, try cu121 next.
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```
**Pass criteria:** `cuda available: True` and `device name: NVIDIA L4`. If `cuda available: False`, the driver-wheel mismatch is the cause — re-install with `--index-url https://download.pytorch.org/whl/cu121` and re-check.

### STEP 10 — Install the rest of requirements
**[VM]:**
```bash
pip install -q -r requirements.txt
pip list | grep -iE "^(torch|gymnasium|numpy|cvxpy|osqp|scipy|pyyaml|matplotlib|wandb)"
```
Verify torch is still the CUDA build (version should contain `+cu124` or similar). If pip swapped it for the CPU wheel, re-run STEP 9.

### STEP 11 — Pytest sanity
**[VM]:**
```bash
pytest -q 2>&1 | tail -20
```
**Pass criteria:** `508 passed, 1 skipped, 0 failed` exactly. One known harmless warning from `ecbf_filter.py:309` (CVXPY "Solution may be inaccurate") is expected during 8 pipeline tests and does not fail them.

If any test fails, stop and escalate. Do not start the probe on a broken env.

### STEP 12 — tmux session for the probe
**[VM]:**
```bash
tmux new -d -s probe
tmux send-keys -t probe "cd /root/hcmarl_project && source venv/bin/activate" Enter
```
The `-d` flag creates tmux detached so the VM-Claude session doesn't get absorbed into the tmux. The send-keys pattern lets VM Claude control tmux panes without attaching.

### STEP 13 — SKIP W&B login today
Both `probe_500k.yaml` and `watch_1m.yaml` have `logging.use_wandb: false`. No W&B credentials are needed today. W&B will be configured tomorrow on sir's VM if the 45-run batch uses it (decision deferred).

### STEP 14 — The measurements (split into 14a–14d)

#### 14a — 500K probe
**[VM], inside the probe tmux session:**
```bash
tmux send-keys -t probe "time python scripts/train.py --config config/probe_500k.yaml --method hcmarl --seed 0 --device cuda --run-name probe_500k 2>&1 | tee logs/probe_500k.log" Enter
```
Wait approximately 15–20 minutes. VM Claude can monitor by reading the last 50 lines of the log:
```bash
tail -n 50 /root/hcmarl_project/logs/probe_500k.log
```
or by peeking at the tmux pane:
```bash
tmux capture-pane -t probe -p | tail -n 50
```

**At the end, extract:**
- Wallclock from the `real Xm Y.Zs` line of `time`.
- Step count from the last `[Ep ... | Step ...]` line (should reach ≈ 500000).
- Any `lazy_agent_flag` warnings (grep the log).
- Any MMICRL `MI collapse` warnings.

**Report to local-primary:**
| Metric | Value |
|---|---|
| Wallclock | Xm Y.Zs |
| Final step | ≈ 500 000 |
| SPS = 500000 / wallclock_seconds | compute |
| lazy_agent events | count |
| MMICRL MI collapse | yes/no |

Local-primary uses this to decide whether to proceed to 14c (1M watch) or abort and replan.

#### 14b — Decision gate (local-primary)
**[LOCAL].** Given SPS, local-primary interprets:

| SPS | 3M/run | 45 runs | Verdict |
|---|---|---|---|
| ≥ 555 | ≤ 1.5 hr | ≤ 67.5 hr | 3M comfortable |
| 415–555 | 1.5–2.0 hr | 67–90 hr | 3M workable, tight at 90 |
| 333–415 | 2.0–2.5 hr | 90–112 hr | 3M needs seed cut or sir's VM must be faster |
| < 333 | > 2.5 hr | > 112 hr | 3M mandatory, consider cutting ablation seeds |

If SPS < 300, stop and replan scope before spending more on the 1M watch.

#### 14c — 1M watch-curve
**[VM], new tmux session so the probe tmux is preserved as record:**
```bash
tmux new -d -s watch
tmux send-keys -t watch "cd /root/hcmarl_project && source venv/bin/activate" Enter
tmux send-keys -t watch "time python scripts/train.py --config config/watch_1m.yaml --method hcmarl --seed 0 --device cuda --run-name watch_1m 2>&1 | tee logs/watch_1m.log" Enter
```
Wait approximately 30–60 minutes depending on SPS.

#### 14d — Plateau verdict
**[VM], after 14c finishes:**
```bash
python scripts/check_plateau.py logs/hcmarl/watch_1m/training_log.csv
echo "exit=$?"
```
**Report to local-primary:** the full output plus the `exit=N` code.

- Exit 0 / PLATEAU → 3M steps is enough. Option 4 locked at 5 seeds × 4 headline + 5 seeds × 5 ablation = 45 runs × 3M.
- Exit 1 / STILL_CLIMBING → 5M needed. Scope needs replan.
- Exit 2 / REGRESSING → abort. Debug before the batch.
- Exit 3 / NOT_ENOUGH_DATA → run didn't reach 500K. Investigate a crash.

Also read and report the last 5 rows of `logs/hcmarl/watch_1m/training_log.csv`:
```bash
echo "header:"
head -1 /root/hcmarl_project/logs/hcmarl/watch_1m/training_log.csv
echo "last 5 rows:"
tail -5 /root/hcmarl_project/logs/hcmarl/watch_1m/training_log.csv
```

### STEP 15 — Monitor (already covered by 14a/14c)
Two techniques, both valid:
- **tmux detach + reattach:** `Ctrl-b d` to detach, `tmux attach -t probe` to reattach. VM Claude uses `tmux capture-pane -p` instead.
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
1. MyAccount → Compute → Manage Nodes → your L4 node → `Destroy`.
2. Confirm the destroy dialog.
3. Refresh the Nodes page — state should be gone within 30 seconds.
4. Open Billing → verify the running charge for this node has stopped.

**Do not destroy** until STEP 16 has completed and the CSVs are safely on the laptop. Once destroyed, the VM disk is unrecoverable.

---

## 6. Decision gates (who decides what)

| Decision | Who |
|---|---|
| Abort probe early if OOM / crash | [VM] Claude, report immediately |
| SPS too slow → abort 1M watch | [LOCAL] based on 14a report |
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

---

## 8. Cost table (today)

| Stage | Wallclock | Pre-GST cost | With 18% GST |
|---|---|---|---|
| Bootstrap (steps 5–11) | ~6 min | ₹5 | ₹6 |
| 500K probe | ~15–20 min | ₹12–16 | ₹14–19 |
| 1M watch | ~30–45 min | ₹25–37 | ₹30–44 |
| Plateau check + scp + destroy | ~10 min | ₹8 | ₹10 |
| **Total** | **~65–80 min** | **~₹55–70** | **~₹65–80** |

Running total across both days: ₹17.15 (yesterday aborted) + ₹80 today = ~₹97. Leaves ~₹1,903 of the ₹2,000 E2E free credits for the sir-VM fallback path.

---

## 9. Emergency procedures

| Event | Response |
|---|---|
| SSH drops mid-run | Training is in tmux; it keeps going. Reconnect: `ssh -i ~/.ssh/id_ed25519 root@<IP>` then `tmux attach -t probe` or `watch`. |
| Bootstrap heredoc hangs at `>` prompt | Your `EOF` line had trailing whitespace. Type `EOF` on its own and Enter. If still stuck, `Ctrl-C` and re-run the heredoc. |
| `cuda available: False` after torch install | Re-install torch with `--index-url https://download.pytorch.org/whl/cu121`. If still CPU-only, the driver is too old — report the `nvidia-smi` CUDA version to [LOCAL]. |
| pytest fails with < 508 passed | Stop. Do not proceed. Paste the last 30 lines of pytest output to [LOCAL]. |
| Training crashes mid-probe | tmux pane will have the traceback. VM Claude captures it: `tmux capture-pane -t probe -p > /tmp/probe_crash.txt`. Then `tail -100 /tmp/probe_crash.txt`. Escalate to [LOCAL]. |
| OOM on GPU | Unlikely at n_workers=6 on 24 GB L4, but if it happens: tmux pane has the traceback. Escalate; do not retry blindly. |
| Claude Code on VM hits rate limit | User switches to their third account for VM session, re-pastes §0.1 boot prompt, VM Claude resumes from the RUNBOOK. |
| E2E billing looks higher than expected | Stop all runs. [USER] checks Billing page. Escalate to [LOCAL] for recomputation. |
| Node gets destroyed accidentally before STEP 16 | The CSVs are gone. Relaunch a fresh L4 node, re-bootstrap, re-run probe + 1M. Budget impact ~₹80 more. |

---

## 10. What happens after today

Tomorrow (2026-04-20) the 45-run batch runs on sir's VM (assuming sir has sent GPU specs + access details). The handoff is:

1. [USER] receives credentials from sir.
2. [USER] starts a new VM-side Claude session on sir's VM (third Anthropic account), paste §0.1 boot prompt with the path updated to wherever the clone lands.
3. [VM] executes steps 5.1 through 11 on the new host (bootstrap).
4. [VM] launches `scripts/run_baselines.py` and `scripts/run_ablations.py` in tmux. Both read `config/experiment_matrix.yaml` and respect today's 3M-or-5M decision.
5. Runs execute ~3 days unattended.
6. On completion, [VM] runs `scripts/aggregate_learning_curves.py --out results/`.
7. [LOCAL] reviews results, [USER] prepares for 2026-04-28 internal evaluation.

If sir's VM path fails (no GPU, wrong spec, access broken, etc.), fallback plan:
- [USER] launches a new E2E L4 node using remaining credits (~33 hr usable).
- That covers the headline grid alone (4 methods × 5 seeds × 1.5 hr = 30 hr). Ablations slip to a second session or get cut to 3 seeds.

---

## 11. End-of-today checklist (read before destroying the node)

- [ ] Probe SPS reported to [LOCAL]
- [ ] 1M watch plateau verdict reported to [LOCAL] with exit code
- [ ] Last 5 rows of watch_1m CSV reported to [LOCAL]
- [ ] [USER] ran `scp` pull in STEP 16
- [ ] [USER] verified CSVs exist on laptop (`ls logs/hcmarl/probe_500k/` and `ls logs/hcmarl/watch_1m/`)
- [ ] [LOCAL] confirmed scope decision for tomorrow (3M or 5M, seed counts)
- [ ] [USER] destroys the node in E2E dashboard
- [ ] [USER] verifies billing charge has stopped
- [ ] [LOCAL] appends today's session to `logs/project_log.md` and commits

Only after all ten boxes are ticked is today "done."
