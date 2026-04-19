# Claude Research Mode Prompt — Alternative GPU Cloud Platforms for HC-MARL (non-E2E)

## THE ASK

I am an independent researcher in India submitting a paper (HC-MARL: Human-Centric Multi-Agent Reinforcement Learning) to **TMLR (Transactions on Machine Learning Research)** — rolling submissions, no hard deadline. My planned primary compute was **E2E Networks Delhi-NCR L4 GPU @ ₹49/hr on-demand (₹17/hr spot) with ₹2,000 signup credits**. Two E2E-specific problems have now killed that plan:

1. **L4 is out-of-stock at E2E**. I received a "we'll contact you shortly" modal when I clicked Request. Chennai region on E2E has no GPU tab at all; only Delhi-NCR has GPU inventory, and L4 is unavailable there. The remaining Delhi-NCR GPUs (A30, A100, H100, etc.) either cost more per hour than L4 without being faster for my workload, or blow through my budget.
2. **E2E is the only Indian provider I've tried. I now want to investigate every credible alternative platform.**

**I need you to do deep research and return a final recommendation.** Identify the **single best platform + GPU + pricing plan** where I can complete the full experiment matrix (described below) under my budget ceiling, with zero hidden costs. Taxes (GST etc.) are fine to include — I just need them disclosed up-front. Hidden conversion fees, forced minimum top-ups, egress charges that aren't disclosed, mandatory subscription upgrades, or "credit expires in 7 days"-style traps are NOT fine.

---

## MY BUDGET

- **₹2,500 personal out-of-pocket ceiling. Absolute hard cap.** No slack.
- **Plus any free credits the platform gives on signup** (these are not counted against my ₹2,500 — they're a bonus).
- Thus effective budget = ₹2,500 + (free credits, which I want you to maximize via platform choice).
- Payment: I'm in India. UPI / Razorpay / Indian debit cards must work. Some Indian debit cards fail on Stripe-only platforms (AWS, GCP, Azure, Jarvislabs) — flag if you find a Stripe-only platform and tell me whether there's a workaround.

## THE GPU REQUIREMENT

**I need a GPU at least as fast as the Nvidia L4 for ML compute (FP32 ~30 TFLOPS, bf16/fp16 tensor cores).** Acceptable classes: L4, L40, L40S, A10, A10G, A30 (debatable — check FP32), A40, A100, A6000, A6000 Ada, H100, RTX 4090, RTX A6000. T4 is NOT acceptable (it's ~8 TFLOPS FP32, half of L4 — I've already rejected it).

**Critical honest caveat that matters for recommendation**: My workload is **95–99% CPU-bound** per my prior code audits (Python env loop + CVXPY/OSQP + ODE integration dominate; GPU is idle 95% of the time). Faster GPUs don't actually buy me speedup — they just burn money. **What matters far more than GPU choice is host vCPU count and single-thread clock.** A ≥16-vCPU host with an L4-class GPU beats a 4-vCPU host with an A100. Factor this into the recommendation: I'm willing to take an L4/L40S/A10 over a faster GPU if it comes with more vCPUs and cheaper ₹/hr. **But the GPU must be ≥L4 class regardless** — per my instruction above, not T4.

## PLATFORMS TO INVESTIGATE (and find more)

User-named starting set:
1. **Ace Clouds** — unclear what this is, user thinks they saw it mentioned; possibly Ace Cloud Hosting (India) or AceCloud
2. **Vast.ai** — peer-to-peer GPU marketplace, variable vCPU/host quality, free credits status uncertain
3. **Jarvislabs** — Indian, acquired by E2E December 2025, Stripe-only payment historically
4. **YottaShakti** — user thinks this is different from Yotta Data Services (Yotta is enterprise-only H100 clusters); YottaShakti may be their self-serve arm, verify status

Plus: **find every other credible platform that offers ≥L4-class GPUs at low ₹/hr and accepts Indian payment**. Candidates to research: RunPod (community + secure cloud), Lambda Labs, Paperspace / DigitalOcean Gradient, TensorDock, LeaderGPU, CoreWeave, Crusoe, Fluidstack, Hyperstack, Modal, Lightning.ai (paid tier), Nerc / Nscale, Genesis Cloud, Massed Compute, Shadeform, Datacrunch, OVHcloud, Hetzner (if GPU offered), Ola Krutrim Cloud (new Indian entrant 2024–2025), Sify AI Cloud, Reliance Jio AI Cloud (if live), Tata Communications / TCS AI Cloud, NxtGen, CtrlS, Nutanix — anything else that surfaces.

For each viable platform, produce the following:

| Field | What to find |
|---|---|
| Platform name + URL | |
| Whether it still exists & is open for signup as of November 2026 | Verify, don't assume |
| GPU models offered that are ≥L4 class | List each |
| vCPU count on the cheapest instance with that GPU | Critical — mine is CPU-bound |
| ₹/hr on-demand (convert USD at ~₹85/USD if USD-priced) | |
| ₹/hr spot / preemptible / community (if offered) | |
| Free signup credits offered (amount + expiry + any conditions) | **Dig hard — some give $100, $300, even $500** |
| Payment methods accepted | UPI? Razorpay? Indian debit/credit? Stripe? Crypto? |
| Hidden costs | Egress, storage, snapshot, IP fees, mandatory subscriptions, minimum balance, credit expiry traps |
| GST/VAT treatment | Applied on top, or included, or not applicable (if non-Indian) |
| Spot preemption SLA (notice period, restart behavior, checkpoint persistence) | |
| Indian availability | Some global platforms decline Indian cards silently |
| Verdict: fits my budget for full matrix? | Explicit yes/no with numbers |

## THE EXPERIMENT MATRIX (scope options, with explicit preference ordering)

Four scope options. Ranked from most preferred to least preferred — pick the most preferred one that fits the budget:

1. **Option 1 (most preferred): 5M steps + 10 seeds for ablations**
   - Headline: 4 methods × 10 seeds = 40 runs × 5M steps
   - Ablation: 5 rungs × 10 seeds = 50 runs × 5M steps
   - Total: 90 runs × 5M steps
2. **Option 2: 5M steps + 5 seeds for ablations**
   - Headline: 4 methods × 10 seeds = 40 runs × 5M steps
   - Ablation: 5 rungs × 5 seeds = 25 runs × 5M steps
   - Total: 65 runs × 5M steps
3. **Option 3: 3M steps + 10 seeds for ablations**
   - Headline: 4 methods × 10 seeds = 40 runs × 3M steps
   - Ablation: 5 rungs × 10 seeds = 50 runs × 3M steps
   - Total: 90 runs × 3M steps
4. **Option 4 (least preferred): 3M steps + 5 seeds for ablations**
   - Headline: 4 methods × 10 seeds = 40 runs × 3M steps
   - Ablation: 5 rungs × 5 seeds = 25 runs × 3M steps
   - Total: 65 runs × 3M steps

**Training and baselines are ALWAYS 10 seeds. The 10↔5 flex is only for ablations.** This is non-negotiable — reducing headline seeds below 10 kills the rliable IQM claim.

**Expected per-run wall-clock: ~1.5 hr on L4-class GPU at 3M steps, ~2.5 hr at 5M steps** (per prior throughput audit at 1,000–1,500 env steps/sec; actual will vary by host vCPUs).

## VERIFIED: TMLR DOES NOT REQUIRE A SCALING STUDY

I had a prior Research Mode audit (file #4 of 9, compass_artifact_wf-5162293e) that explicitly says: *"Do not run the N ∈ {4, 6, 8, 12} scaling study. TMLR's correctness criterion evaluates whether claims are supported, not whether experiments are exhaustive."* Please re-verify this against current TMLR submission guidelines at jmlr.org/tmlr if anything has changed, but I'm treating it as settled: **no scaling study, fixed N=6.** Do not size the budget for an N-sweep.

---

## THE PROJECT (context the platforms matter for)

**HC-MARL (Human-Centric Multi-Agent RL)** is a cooperative MARL framework for warehouse worker task allocation with physiological fatigue constraints. Four components:

1. **3CC-r fatigue ODE** — 3-compartment coupled differential equation (Looft et al. 2018 parameters) tracking muscle activation, fatigue, and recovery per agent per muscle group. Integrated at 1-min resolution for 480-step episodes (8-hr shift).
2. **ECBF safety filter** — Exponential Control Barrier Function with analytical inlined form (NOT a CVXPY/OSQP QP solve per step; that was the previous implementation — it is now closed-form scalar math in `hcmarl/envs/pettingzoo_wrapper.py` lines 205–218). Still CPU work, but microseconds per agent.
3. **NSWF allocator** — Nash Social Welfare Function for fair task-capacity allocation across 6 workers.
4. **MMICRL** — Multi-modal Inverse Constrained RL with normalizing flows (MADE/MAF), demoted to a diagnostic / pretraining module. Held-out NLL for K-selection (Watanabe 2013 — BIC invalid for singular flows).

Base learner: **MAPPO** (actor-critic, hidden_dim=64, critic_hidden_dim=128, batch_size=256, n_epochs=10). 6 agents, 6 muscle groups. Sequential (not vectorized) GPU forwards in `get_actions()` — one of the real bottlenecks alongside env step.

Python 3.13.3, PyTorch 2.6 (with `weights_only=False` patch on all checkpoint loaders). `cudnn.deterministic=True` → ~1.5× slowdown, bit-exact reproducibility required for TMLR correctness claim. `torch.load`, `run_state.pt` (counters + RNG + theta_max) paired with every checkpoint for bit-identical resume.

**Test suite**: 508 passed / 1 skipped / 0 failed. All 4 agent classes save/load optimizer state. Batch A (slack-augmented CBF-QP + strict OSQP status), Batch B (determinism), Batch C (long-run stability), Batch D (rliable IQM + stratified bootstrap CIs + 5-way attribution ladder + lazy-agent kill-switch), Batch E (MMICRL validity), Batch F (NIOSH rebuttal armor) all green.

### Execution files (what the platform will actually run)

- `scripts/train.py` (~1008 lines, main training loop). Built-in budget kill-switch: `--budget-inr N --cost-per-hour X --budget-margin 0.95` physically halts training when wall-clock cost hits 95% of cap.
- `scripts/run_baselines.py` — matrix-driven from `config/experiment_matrix.yaml`. Iterates 4 methods × 10 seeds.
- `scripts/run_ablations.py` — matrix-driven. Iterates 5 rungs × {5 or 10} seeds.
- `scripts/aggregate_learning_curves.py` — post-hoc rliable IQM + stratified bootstrap 95% CI at anchors [500K, 1M, 2M, 3M].
- `config/experiment_matrix.yaml` — single source of truth for scope.
- `config/hcmarl_full_config.yaml`, `config/mappo_config.yaml`, `config/ippo_config.yaml`, `config/mappo_lag_config.yaml`, `config/ablation_no_{ecbf,nswf,mmicrl}.yaml` — per-method hyperparameters.

Single-process training loop (no SubprocVecEnv for now — adding it before submission is high-risk). Environment: custom PettingZoo `ParallelEnv` wrapper.

---

## DELIVERABLES I WANT BACK FROM YOUR RESEARCH

1. **Ranked shortlist of top 3 platforms** — each with the full field table above, written so I can verify every number.
2. **Explicit verdict for each scope option (1–4)** — which platform + which option fits, under budget, with how much reserve. Order them per my preference: try Option 1 first; if that doesn't fit on any platform, try Option 2; then 3; then 4.
3. **Free credit deep dive** — which platforms give the largest free credits, exactly how to claim them, expiry rules, whether they can be combined with spot pricing, any gotchas (e.g., "free credits cannot be used for GPU instances" or "credits expire after 7 days of first use"). This is a lever I care about a lot.
4. **Hidden-cost audit** — for each top-3 platform, enumerate every non-compute fee you can find. Storage (persistent disk ₹/GB/mo, ephemeral disk, snapshot fees), network egress (₹/GB, free-tier egress caps), public IP charges, instance-create/delete fees, minimum balance requirements, subscription gating ("GPU only available on Pro tier"), forced commit terms, auto-renewal traps, and currency-conversion markups if billed in USD and paid via an Indian card.
5. **Payment friction assessment** — for each top-3 platform, confirm with current evidence whether an Indian HDFC/ICICI/SBI/Axis debit card + UPI/Razorpay will work. If Stripe-only, flag it and note whether Razorpay-on-top or any alternative gateway is available.
6. **Re-verify TMLR requirements** — one paragraph confirming TMLR has no scaling-study mandate and that my N=6 fixed configuration is acceptable for a correctness-based review. Flag anything new in TMLR's guidelines (2026) that would change this.
7. **Final recommendation in one paragraph** — the single platform + GPU + pricing plan I should sign up for tonight, the scope option it enables, the total cost estimate (compute + GST + any storage + egress), the free credits that apply, the expected reserve after the matrix completes, and the exact reason this beats every other candidate.

---

## HARD RULES FOR YOUR RESEARCH

- **Do not hallucinate pricing.** Every ₹/hr or $/hr number must come from a URL you can cite, checked as live during your research. Include the URL.
- **Do not assume "all ≥L4 GPUs are interchangeable for my workload"** — confirm the host vCPU count and list it.
- **Do not recommend a platform just because it's cheap per-hour if vCPUs are 2–4** — it won't actually reduce my wall-clock, and it violates my CPU-bound constraint.
- **Do not recommend T4.** Already rejected.
- **Do not assume Stripe-only platforms are fine for Indian users.** Flag Stripe-only loudly.
- **Do not skip free credits.** If a platform doesn't advertise them on the pricing page, check their /signup or /get-started pages, student programs, startup programs, and recent blog posts.
- **Do not skip the hidden-cost audit.** Egress fees on one wrong platform have eaten entire budgets before. Verify it.
- **Indian-specific friction matters more than raw price.** A platform that's ₹5/hr cheaper but declines my card is worthless.

I'm under deadline pressure and budget pressure. I need the tightest, most honest recommendation you can produce. Thank you.
