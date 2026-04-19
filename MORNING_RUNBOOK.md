# RUNBOOK — 2026-04-19 (today) and 2026-04-20 (sir's VM)

## Today's scope on E2E (₹60–100 total)

This is a **measurement session only**. No batch runs on E2E. Goal: get real SPS and a
1M reward curve, decide 3M-vs-5M and final seed count, then **delete the node**. The
45-run batch happens tomorrow on sir's VM (or comes back to E2E free credit if his
setup fails).

**Spend ceiling today:** ~₹100. Stops long before ₹1,900 of the ₹2,000 free credits.

---

## STEP 0 — Before you SSH: re-tar the project (10 sec)

Git Bash on your Windows laptop, from `C:\Users\admin\Desktop`:

```bash
cd /c/Users/admin/Desktop

tar --exclude='hcmarl_project/venv' \
    --exclude='hcmarl_project/data' \
    --exclude='hcmarl_project/checkpoints' \
    --exclude='hcmarl_project/figures' \
    --exclude='hcmarl_project/REFERENCES' \
    --exclude='hcmarl_project/logs/hcmarl' \
    --exclude='hcmarl_project/logs/mappo' \
    --exclude='hcmarl_project/logs/ippo' \
    --exclude='hcmarl_project/logs/mappo_lag' \
    --exclude='hcmarl_project/diagrams' \
    --exclude='hcmarl_project/.git' \
    -czf hcmarl.tar.gz hcmarl_project

ls -lh hcmarl.tar.gz   # expect ~1.4 MB
```

Alternative: once tonight's commit is pushed to GitHub, you can skip tar and do
`git clone https://github.com/ADITYA-WORK-MAITI/hcmarl-project.git` on the VM
instead. Tar is faster and has no auth dependency, so it's the default.

---

## STEP 1 — Launch L4 on E2E

1. Login → MyAccount → Compute → GPU → Launch GPU
2. Pick **Nvidia-L4**, **Plan #1 (25 vCPU / 110 GB / 24 GB VRAM)**, **On-Demand ₹49/hr**
3. SSH key: attach `hcmarl-laptop` (already uploaded yesterday)
4. Tick "Disable Password-based SSH login"
5. Launch → wait for status "Running" → copy the public IP

If "plan temporarily not available": refresh and retry. Yesterday the error cleared on the second attempt.

Billing starts the second state flips to Running. Move fast from here.

---

## STEP 2 — Upload + bootstrap (ONE PASTE, ~4 min)

Substitute the new IP on the first line, then paste the whole block into Git Bash:

```bash
NEW_IP=164.52.X.X

scp -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 \
    /c/Users/admin/Desktop/hcmarl.tar.gz root@$NEW_IP:/root/

ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@$NEW_IP << 'EOF'
set -e
echo "=== nvidia-smi ==="
nvidia-smi | head -20
echo "=== apt ==="
apt-get update -q && apt-get install -y -q python3.12-venv python3-pip tmux htop nvtop
echo "=== extract ==="
cd /root && tar -xzf hcmarl.tar.gz && cd hcmarl_project
echo "=== venv ==="
python3.12 -m venv venv && source venv/bin/activate
pip install -q -U pip wheel
echo "=== torch (CUDA wheel FIRST so requirements.txt is satisfied by CUDA build) ==="
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
echo "=== requirements ==="
pip install -q -r requirements.txt
echo "=== torch check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
echo "=== pytest ==="
pytest -q 2>&1 | tail -20
EOF
```

**Pass criteria (all three must hold):**
- `nvidia-smi` shows `NVIDIA L4` with 24 GB free
- torch check prints `cuda: True` (NOT `cuda: False`)
- pytest tail: `508 passed, 1 skipped` (0 failed)

If any fail → STOP. Don't burn ₹49/hr running the probe on a broken env.

---

## STEP 3 — 500K probe inside tmux (~15 min, ₹12)

```bash
ssh -i ~/.ssh/id_ed25519 root@$NEW_IP
# on remote:
cd /root/hcmarl_project && source venv/bin/activate
tmux new -s probe
# inside tmux:
time python scripts/train.py \
    --config config/probe_500k.yaml \
    --method hcmarl \
    --seed 0 \
    --device cuda \
    --run-name probe_500k 2>&1 | tee logs/probe_500k.log
# detach tmux with: Ctrl-b then d
```

**What you're measuring:** the `real X.Xm` line at the end.

| Wallclock | SPS | 3M/run | 5M/run | 45 runs @ 3M | Verdict |
|---|---|---|---|---|---|
| ≤ 15 min | ≥ 555 | ~1.5 hr | ~2.5 hr | 67.5 hr ≈ ₹3,900 | **3M fits anywhere** |
| 16–20 min | 415–555 | ~1.8–2.0 hr | ~3.0–3.3 hr | 81–90 hr | 3M needs tight seed count |
| 21–25 min | 333–415 | ~2.1–2.5 hr | ~3.5–4.2 hr | 94–112 hr | Drop to 3M + 5 headline + 3 ablation seeds |
| > 25 min | < 333 | > 2.5 hr | > 4.2 hr | > 112 hr | 3M mandatory, cut ablations harder |

Also check: `logs/probe_500k/training_log.csv` reached `global_step` ≈ 500000 and has no `lazy_agent_flag=True` rows.

---

## STEP 4 — 1M watch-curve inside tmux (~30 min, ₹25)

Edit the probe config in-place on the remote to extend to 1M, then rerun under a new run-name:

```bash
# still SSH'd into the node:
cd /root/hcmarl_project && source venv/bin/activate
sed -i 's/total_steps: 500000/total_steps: 1000000/' config/probe_500k.yaml
grep total_steps config/probe_500k.yaml   # verify it now reads 1000000

tmux new -s watch
time python scripts/train.py \
    --config config/probe_500k.yaml \
    --method hcmarl \
    --seed 0 \
    --device cuda \
    --run-name watch_1m 2>&1 | tee logs/watch_1m.log
# detach with Ctrl-b d
```

**Plateau check** (copy-paste once the run finishes):

```bash
python scripts/check_plateau.py logs/hcmarl/watch_1m/training_log.csv
```

Exit codes map directly to your scope decision:
- `0` / PLATEAU → **3M per run is enough.** Commit Option 4.
- `1` / STILL_CLIMBING → **5M needed.** Scope balloons ~67% — you'll need to cut seeds to 3 per method/rung, or accept partial results, or fall back to E2E free credits.
- `2` / REGRESSING → **Do NOT launch the batch.** Something is wrong — inspect the curve, check for NaN, check lazy-agent flag. Debug before spending more.
- `3` / NOT_ENOUGH_DATA → run didn't reach 500K steps; re-examine logs for a crash.

Also eyeball the CSV for sanity:
- `cumulative_reward` climbing (noisy is fine, monotonically flat is bad)
- `safety_rate` close to 1.0 after ~200K steps
- `peak_fatigue` ≤ ~0.5
- `lazy_agent_flag` never `True`

---

## STEP 5 — Delete the node (CRITICAL)

Once you've read the probe wallclock and the plateau verdict and written them down somewhere permanent:

1. **Note the numbers** — SPS, wallclock, plateau ratio, cumulative_reward at 1M. These decide tomorrow's scope.
2. **E2E dashboard → Manage Nodes → your L4 → Destroy** (confirm twice).
3. Verify in Billing that the running charge has stopped.

₹49/hr idle = ₹1,176/day burned doing nothing. Destroy decisively.

**Do NOT** destroy before running the plateau check — you lose the CSV when the node is destroyed. If you want to keep the logs, `scp` them down first:

```bash
# from Git Bash on Windows, BEFORE destroying the node:
scp -i ~/.ssh/id_ed25519 -r root@$NEW_IP:/root/hcmarl_project/logs/hcmarl/watch_1m \
    /c/Users/admin/Desktop/hcmarl_project/logs/hcmarl/
scp -i ~/.ssh/id_ed25519 -r root@$NEW_IP:/root/hcmarl_project/logs/hcmarl/probe_500k \
    /c/Users/admin/Desktop/hcmarl_project/logs/hcmarl/
```

---

## STEP 6 — Update scope based on the two numbers

In this conversation, paste me:
- the `real X.Xm` wallclock from step 3
- the VERDICT line from step 4's `check_plateau.py`
- peak cumulative_reward at step 1000000 (last row of watch_1m CSV)

I'll tell you: 3M vs 5M, final seed counts, and the revised budget math for tomorrow.

---

## Tomorrow (2026-04-20) — sir's VM

Once sir sends GPU model + SSH/portal details:

1. Bootstrap on his VM using the same one-paste heredoc (step 2 above), adjusting only the SSH target.
2. Launch `scripts/run_baselines.py` in tmux (headline grid, 20 runs).
3. Launch `scripts/run_ablations.py` in tmux (ablation grid, 25 runs).
4. Both read `config/experiment_matrix.yaml` — single source of truth for seeds and step counts.
5. Aggregate with `scripts/aggregate_learning_curves.py` when all 45 runs finish.

If sir's setup fails: relaunch L4 on E2E using remaining ~₹1,940 credit. That covers ~33 compute hours pre-GST, which is roughly 22 runs at 1.5 hr each — enough for the headline grid alone. Ablation grid would then go on a fresh session or be cut.

---

## Quick reference — today's budget

| Stage | Time | ₹ pre-GST | ₹ with 18% GST |
|---|---|---|---|
| Bootstrap | ~4 min | 3 | 4 |
| 500K probe | ~15 min | 12 | 14 |
| 1M watch | ~30 min | 25 | 30 |
| Plateau check + scp + destroy | ~10 min | 8 | 10 |
| **Total today** | **~60 min** | **~₹48** | **~₹58** |

Leaves ₹1,940+ of your E2E free credits intact as insurance against sir's VM failing tomorrow.

---

## Safety / gotchas

- **Always run in tmux.** If SSH drops, the training job keeps going. Yesterday's rate-limit cutoff killed a ₹17 bootstrap because nothing was in tmux.
- **Heredoc EOF trap:** bash heredocs fail silently if the terminating `EOF` has trailing whitespace. If bootstrap hangs at a `>` prompt, it means your paste clipped the `EOF` line — type `EOF` alone on its own line and press Enter.
- **`pip install torch` BEFORE `pip install -r requirements.txt`**, not after. The CUDA wheel must win the dependency resolution first; otherwise pip will pull the CPU build from PyPI and `requirements.txt` will "satisfy" torch from that CPU build.
- **`weights_only=False` loader is already patched** (2026-04-17 execution hardening). Checkpoints survive PyTorch 2.6+ restore without warnings.
- **Resume is bit-identical** — if the node preempts mid-run (E2E on-demand doesn't preempt, but general habit), `run_state.pt` + RNG round-trip make resumption lossless.
