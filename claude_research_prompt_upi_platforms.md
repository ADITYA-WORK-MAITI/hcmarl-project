# Claude Research Mode prompt — UPI-explicit GPU cloud platforms for HC-MARL TMLR submission

> **This prompt supersedes `claude_research_prompt_alt_platforms.md`.** The previous prompt was scoped to crypto-pay / Stripe alternatives on Vast.ai. All three Vast.ai payment rails (Stripe, Crypto.com Pay, BitPay) have since been verified dead for unaffiliated Indian individuals. We have also dropped E2E Networks (out of L4 stock, KYC friction). The scope here is **pure platform discovery**: find GPU cloud providers that explicitly accept **UPI** (Unified Payments Interface — India's real-time bank-to-bank rail), at ≥L4-class GPU speed, under a hard ₹2,500 personal-out-of-pocket ceiling + whatever free credits the platform offers on signup.

---

## A. What you are researching (summary)

A single researcher in India (Delhi, GMT+5:30) needs to run **45–90 RL training jobs** for a TMLR journal submission over the next ~10–14 days. His local laptop is not fast enough to run the production matrix in the available time. All payment rails he has tried on the two leading low-cost platforms (Vast.ai, E2E Networks) have failed — either because Indian debit cards silently fail Stripe, KYC walls block invoices, or no UPI channel exists. He is asking you to find **other GPU cloud platforms that explicitly accept UPI**, compare them on total cost (taxes included, no hidden fees), and advise on which to sign up for. He wants a full map of Indian-friendly + global platforms he may not have heard of, with verified free-credit amounts and verified UPI support.

The seed list he has heard of (but whose UPI/L4 status he has NOT verified): **Ace Clouds, Vast.ai, Jarvis Labs, YottaShakti**. You should treat this list as a starting point, not an answer. Discover more.

**Budget ceiling: ₹2,500 personal out-of-pocket** (all-in, including GST/VAT/taxes). Any free signup credits are additive and should be enumerated aggressively — he specifically wants to know if any platform offers large-value free credits for first-time signup, academic/research programs, or GitHub Student Pack linkage.

---

## B. The project (so you understand the compute shape)

**Project:** HC-MARL — Human-Centric Multi-Agent Reinforcement Learning for fatigue-aware warehouse task allocation. Single-author submission to **TMLR** (Transactions on Machine Learning Research, OpenReview, rolling).

**Core modules** (all in `hcmarl/` Python package, torch 2.6 + numpy + cvxpy + osqp):
- **3CC-r fatigue ODE** — 3-compartment Xia/Frey-Law model, Euler-stepped per agent per env-step.
- **ECBF safety filter** — Exponential Control Barrier Function wrapped as a small QP (`h(x_next) ≥ 0`), solved analytically (inline, no CVXPY/OSQP per step in production — the analytical form is inlined for throughput; CVXPY is only used in tests).
- **Nash Social Welfare allocator** — closed-form per-step task allocation based on a fairness objective.
- **MMICRL pretrainer** — Multi-modal Imitation from Constrained Reward Learning using normalizing flows (MADE/MAF) for worker-type theta discovery. Runs once before RL as a pretrain phase (~5–15 minutes, negligible cost vs. 1.5–2.6 hr per run).
- **MAPPO base learner** — standard 4-method set: `hcmarl`, `mappo`, `ippo`, `mappo_lag` (MAPPO-Lagrangian).

**Environment:** 6-agent warehouse task-allocation env (PettingZoo ParallelEnv, custom), 480 steps/episode, ~10,417 PPO updates per 5M-step run.

**Workload shape — THIS IS THE CRITICAL ENGINEERING FACT:**
- **95–99% CPU-bound.** The neural networks are tiny (actor hidden_dim=64, critic hidden_dim=128, ~250K params total across 6 agents). The GPU sits at <1% utilization. Wall-clock is dominated by env stepping: dict-of-dicts Python hot loop, ODE integration, and ECBF evaluation.
- **Single-process PyTorch training loop.** No SubprocVecEnv; no Ray; no env parallelism. A subprocess rewrite is out of scope pre-submission.
- **GIL-bound.** Cloud vCPU count does NOT convert to parallelism; what matters is **single-thread CPU speed** of one core, plus maybe ≥4 vCPUs for OS / checkpoint I/O / MMICRL pretrain overlap.
- **The GPU is effectively a compliance requirement** (PyTorch 2.6 `cudnn.deterministic=True` for bit-identity; also the small MLPs still run fastest on GPU because the CPU path would consume the hot-loop core). **≥L4 class is required** — meaning FP32 ≥ 30 TFLOPS, ≥ 24 GB VRAM, Ada or Ampere architecture. **T4 is rejected** (prior pilot showed insufficient throughput; reviewer also rejected T4 on kernel-launch overhead grounds).

**Per-run wall-clock estimate** (from audited compute plan, `compass_artifact_wf-6aa451f5-...md` in this session's research archive):
- Expected throughput: **~1,270 env-steps/sec** on an L4 with a modern Xeon/EPYC host.
- Pessimistic: 825 sps. Optimistic: 2,055 sps.
- Per 5M-step run: **2.6 hr expected** (1.1 hr upper / 5.3 hr lower).
- Per 3M-step run: **~1.5 hr expected.**
- Per MMICRL pretrain: ~5–15 minutes, one-time before RL.

**Throughput hedge:** A prior research pass warned the 1,270 sps estimate may be 5–10× optimistic on cheaper/older Xeon shapes, and that you should verify by requesting a 500K-step probe run on any shortlisted provider before committing.

**Training pipeline files the researcher will actually run on the platform:**
- `scripts/train.py` — single-run entrypoint. Accepts `--method`, `--seed`, `--total_steps`, `--run-name`, `--drive-backup-dir`, `--checkpoint_interval`.
- `scripts/run_baselines.py` — drives 4 methods × 10 seeds from `config/experiment_matrix.yaml`.
- `scripts/run_ablations.py` — drives 5-rung build-up ladder × (5 or 10) seeds.
- `scripts/aggregate_learning_curves.py` — post-hoc: reads logs, emits rliable IQM + 95% stratified bootstrap CI figures.
- `config/hcmarl_full_config.yaml`, `config/mappo_config.yaml`, `config/ippo_config.yaml`, `config/mappo_lag_config.yaml` — one per headline method.
- `config/ablation_no_{ecbf,nswf,mmicrl,reperfusion,divergent}.yaml` — one per ablation rung.
- `config/experiment_matrix.yaml` — the matrix driver (seeds × methods × step budgets).

**Checkpointing:** Already implemented — `run_state.pt` every 100K steps with full optimizer/RNG/global_step. PyTorch 2.6 `weights_only=False` patch applied (all 4 agent loaders). Bit-identical resume verified. So **spot / preemptible instances are safe** as long as the filesystem path persists (a 100K-step loss at 1,270 sps ≈ 79 seconds of wall-time, trivial).

**Test suite:** 509 pytests, 508 pass + 1 skip. Must stay green on any new host.

---

## C. The execution dilemma (the scenario you are advising on)

### What has been tried and ruled out

| Platform | Attempted payment rail | Outcome |
|---|---|---|
| **E2E Networks** | UPI (native) — platform supports it | Out of L4 stock for self-service signups as of 2026-04-17; enterprise KYC queue; he walked away. |
| **Vast.ai** | Stripe (Indian debit card) | Silent fail — Stripe returns success to Vast but card issuer declines behind the scenes. Zero feedback to the user. |
| **Vast.ai** | Crypto.com Pay | Proprietary `crypto.com://pay?...` deep-link scheme. QR only. Requires the Crypto.com **retail app** (unavailable on Indian Play Store per RBI crypto rules). Not interoperable with standard USDT TRC20 wallets. |
| **Vast.ai** | BitPay | Mandatory KYC wall for new accounts regardless of invoice size. Clicking "Maybe Later" loops back to the same wall after login. Dead end. |

The prior research pass already suggested forex virtual cards (Niyo Global, Fi Money), RunPod, Lambda Labs, Coinbase, etc. — **the user has explicitly rejected these pivots** and wants only platforms that **accept UPI natively** (not "UPI via a virtual card that dumps into Stripe", not "you can pay with a crypto wallet that you can fund via UPI", not "you can add INR from UPI to a prepaid wallet that then pays the platform"). **UPI must be a first-class, listed, documented payment method on the billing page.** Taxes (GST, 18% in India) are fine. Hidden costs are not.

### What "hidden cost" means here — enumerate for each candidate

For every platform you shortlist, explicitly answer yes/no + amount for:
- **Egress / data-out charges.** (Many providers charge per-GB for downloading logs, checkpoints, results.)
- **Block-storage / persistent-disk fees beyond boot disk.** Does the instance bill storage separately? Hourly or daily? Survives instance-stop?
- **Public IP / static IP fees.**
- **Snapshot / image storage fees.**
- **Minimum balance / top-up minimums.** (Some require ₹500 or $10 minimum top-up, which wastes credit.)
- **Credit expiry.** Do free credits expire after 7/30/90 days? After instance first launch, or after signup? This is the most common trap for free credits.
- **Credit usage restrictions.** Some free credits only work on specific GPU types, specific regions, or require monthly subscription activation.
- **Mandatory subscription / platform fee.** Does the provider charge a monthly platform fee on top of hourly GPU?
- **Idle / reservation charges.** Some providers bill for reserved-but-not-running instances.
- **GST-on-top vs GST-inclusive.** The sticker price must reveal this. Prefer GST-inclusive quotes. If you cannot determine from the provider docs, flag it.
- **Forex / cross-border fees.** Does the provider bill in USD and add 1–3% FX markup on UPI → USD conversion? Is this disclosed?
- **Preemption semantics on spot.** No SLA is fine, but you must know: (a) notice period (0 sec / 30 sec / 2 min), (b) whether the filesystem survives preemption, (c) whether preemption resets your "free credit month" or consumes credit even for the preempted compute.

If you cannot verify any of these from public docs within a single-page browsable source (pricing page, billing FAQ, support article), flag it as "unknown — requires confirmation before top-up" rather than assuming.

---

## D. The hard rules on compute config

1. **GPU floor: ≥L4 class.**
   - Acceptable: NVIDIA L4 (24 GB, 30 TFLOPS FP32, Ada), A10 (24 GB, 31 TFLOPS, Ampere), A10G, RTX 4090 (24 GB, 82 TFLOPS, Ada), RTX A6000 (48 GB, 38 TFLOPS, Ampere), L40 / L40s, A40 (48 GB, 37 TFLOPS, Ampere), A100 (40 GB or 80 GB, 19.5 TFLOPS FP32 but 312 TFLOPS TF32), H100.
   - **Rejected:** T4 (8.1 TFLOPS FP32, Turing), P100, V100 for cost reasons, K80, GTX-series consumer.
2. **Host CPU: ≥8 vCPUs strongly preferred, ≥4 vCPUs absolute minimum**, with a modern (≥2019) Xeon Gold / EPYC Milan+ host. Single-thread turbo ≥3.0 GHz matters because the workload is GIL-bound.
3. **RAM: ≥16 GB** on the instance.
4. **Persistent storage:** need ≥50 GB that survives instance stop/restart, to hold `run_state.pt` across a multi-day queue of runs. On-instance tmpfs / wiped-on-kill disks are OK only if paired with cheap/free rsync to object storage.
5. **Region:** Indian region strongly preferred for UPI-billed providers (GST compliance, lower egress). US/EU regions acceptable if the provider supports UPI regardless of region.

---

## E. Scope options in preference order (DO NOT REORDER THESE — the user was explicit)

**Rule 1: Training + baselines are ALWAYS 10 seeds. The 10 vs 5 flexibility is ONLY for ablations.**
**Rule 2: 4 headline methods × 10 seeds = 40 headline runs (fixed).**
**Rule 3: 5 ablation rungs × {5 or 10} seeds = 25 or 50 ablation runs.**

| Priority | Option | Headline runs | Ablation runs | Total runs | Est. GPU-hours @ 1,270 sps | Est. cost @ spot ₹20/hr | Est. cost @ on-demand ₹58/hr |
|---|---|---|---|---|---|---|---|
| **1 (preferred)** | **5M steps + 10 ablation seeds** | 40 × 2.6 hr = 104 | 50 × 2.6 hr = 130 | 90 | **234 hr** | ~₹4,680 | ~₹13,572 |
| **2** | **5M steps + 5 ablation seeds** | 104 | 25 × 2.6 hr = 65 | 65 | **169 hr** | ~₹3,380 | ~₹9,802 |
| **3** | **3M steps + 10 ablation seeds** | 40 × 1.5 hr = 60 | 50 × 1.5 hr = 75 | 90 | **135 hr** | ~₹2,700 | ~₹7,830 |
| **4** | **3M steps + 5 ablation seeds** | 60 | 25 × 1.5 hr = 37.5 | 65 | **97.5 hr** | ~₹1,950 | ~₹5,655 |

**Your target:** identify the highest-priority option that fits under `₹2,500 personal out-of-pocket + platform's documented free credits`. If Option 1 fits on a single platform with large free credits, that is the answer. If it requires splitting across two platforms (e.g., signup on A for free credits, top up on B with UPI), that is fine and should be proposed. If only Option 3 or 4 fits even with free credits, say so explicitly and do not inflate estimates to make Option 1 look feasible.

**Throughput hedge:** If any platform's host has unclear CPU specs, assume the pessimistic 825 sps estimate (5M at 5.3 hr, 3M at ~3.2 hr) and reprice. Flag the uncertainty. Do not rely on the expected-case number alone.

---

## F. TMLR scaling-study verification (explicit ask)

The user wants you to **re-verify from TMLR's published criteria that a scaling study (sweeping N_agents ∈ {3, 4, 6, 8, 12}) is NOT required** for acceptance. Prior research concluded:

- TMLR's acceptance criterion is **correctness of claims**, not significance / novelty / exhaustiveness of benchmarks.
- HC-MARL makes no scalability claims — the paper studies a fixed N=6 warehouse setup.
- Adding a 3-seed × 2M-step scaling study alongside 10-seed × 5M main experiments would create a conspicuous statistical asymmetry that reviewers would flag as second-class evidence.
- A survey of 2024–2025 TMLR and JAAMAS MARL papers showed systematic N-sweeps only in papers whose core contribution is scalability.

Please verify this is still correct as of today (2026-04-18). Check:
1. TMLR's current submission guidelines (jmlr.org/tmlr) for any mandatory experiment-design requirements.
2. The TMLR OpenReview page for recent (2025–2026) MARL accept papers — do they all include N-sweeps?
3. Any updates to TMLR's Action Editor guidance or certification criteria (Featured, Expert Certification) that would flip the answer.

If verification confirms: state clearly that **scaling is not required** and the 4 scope options above are the correct planning envelope. If verification flips (i.e., TMLR *has* tightened requirements since the prior research pass): say so, describe the new requirement, and recompute the scope options to include the scaling runs, showing how the budget math changes.

---

## G. Candidate platforms to investigate (start here, but cast a wider net)

The user's seed list:
1. **Ace Clouds** (ace-clouds.com / ace-cloud.com — verify URL; may be US "Ace Cloud Hosting" which is enterprise QuickBooks hosting and NOT a GPU provider — disambiguate carefully).
2. **Vast.ai** — already dead on all 3 payment rails for this user. You may flag it only if Vast has introduced a new UPI rail since April 2026.
3. **Jarvis Labs** (jarvislabs.ai) — acquired by E2E in Dec 2025 per prior research; status of self-service UPI billing post-acquisition is unknown; verify.
4. **YottaShakti / Yotta Data Services** (yotta.com, shakti.yotta.com) — Indian provider, possibly enterprise-only; verify self-service + UPI.

Additional platforms to actively investigate (India-focused + global with UPI):
- **Ola Krutrim** (cloud.olakrutrim.com) — Indian AI cloud, launched 2024.
- **Sify CloudInfinit** and **Tata Communications** (GPU-as-a-service offerings in India).
- **Reliance Jio Cloud AI / JioCloud** — announced GPU services.
- **CoreWeave India** (coreweave.com) — if they have an Indian entity with UPI billing.
- **Nscale** (nscale.com) — European GPU cloud; verify UPI.
- **Together.ai / Together AI** (together.ai) — serverless + dedicated; verify UPI.
- **Fireworks.ai**, **Modal Labs**, **Replicate**, **Baseten** — dedicated-instance pricing and UPI status.
- **Paperspace / DigitalOcean GPU** — DO accepts UPI in some Indian accounts; verify.
- **Lightning.ai** — free tier is 4-hour reset, but verify paid tier + UPI.
- **Crusoe Cloud** (crusoe.ai) — low-cost US provider; verify UPI.
- **Hyperstack** (hyperstack.cloud) — NexGen Cloud's GPU product; verify UPI.
- **RunPod** — prior user rejected as non-UPI, but re-verify — RunPod has an "India" payment option via Razorpay in some accounts.
- **Ori Industries** (ori.co) — UK provider; verify UPI / Indian billing entity.
- **Sesterce**, **Genesis Cloud**, **Latitude.sh**, **Atlantic.Net** — verify.
- **Alibaba Cloud India / AWS India / Azure India / GCP India** — large clouds have Indian entities that sometimes accept UPI for pay-as-you-go; check current status, check if they offer L4 / A10 / A100 in Indian regions, and check their free-tier credit programs.
- **Kaggle / Colab / Lightning Studios / Hugging Face Spaces** — these are training platforms that give *free* GPU access. Check current allocations: **Kaggle gives 30 hours/week of T4 or P100 (rejected), but check if they have L4 or better in 2026**; **Colab Pro/Pro+ at ₹850–1,900/mo in Indian pricing, with UPI via Google Play Store billing (Indian Google accounts); verify this still works in 2026**; **Lightning Studios** (formerly Lightning.ai) offers 22 free GPU hours/month on L4 with a Studio subscription; verify current policy.

For each platform, produce a row in a single summary table answering:

| Field | What to find |
|---|---|
| Name | — |
| UPI accepted? | Yes / No / Unknown (if yes, cite the billing-docs URL) |
| Best GPU offered ≥L4-class | Model, VRAM, FP32 TFLOPS |
| On-demand price | INR/hr all-in (GST + taxes) |
| Spot / preemptible price | INR/hr all-in, and preemption semantics |
| vCPU count on that GPU tier | — |
| Persistent storage policy | survives restart? egress fees? |
| Free signup credits | amount + expiry + restrictions |
| Academic / researcher / student credits | GitHub Student Pack / .edu / referral programs |
| Hidden costs (full enumeration per §C.2) | — |
| KYC burden | PAN? Aadhaar? Business-only? Time to approval? |
| Indian GST invoice provided? | relevant for researcher's potential reimbursement later |
| Verdict for this workload | GREEN (sign up now) / YELLOW (acceptable fallback) / RED (skip) |

---

## H. Free-credit deep dive (separate section — user explicitly asked)

Some providers offer **large-value first-signup or academic credits** that effectively make the whole matrix free. Investigate aggressively — these are the highest-leverage finds in this research.

Known programs to verify the current 2026 status of:
- **Google Cloud Platform $300 signup credit** (90 days) — does it still work for Indian-address accounts? Does Indian UPI activate it?
- **Microsoft Azure $200 signup credit** (30 days) — Indian-card acceptance in 2026?
- **AWS $100–$300 signup credit equivalents** + **AWS Activate** (non-equity startup program, up to $5,000–$100,000) — does a single researcher / unincorporated project qualify in 2026?
- **Oracle Cloud Free Tier** — has an always-free tier but no L4; verify if 2026 offers include GPU trials.
- **Lambda Labs credits** — any first-signup bonus in 2026?
- **Paperspace Gradient free tier + signup credits.**
- **Lightning.ai** — free Studio compute + paid add-ons.
- **Hugging Face** — PRO subscription ($9/mo) includes some GPU credits; verify UPI billing via Stripe Indian entity.
- **NVIDIA NGC / NVIDIA Inception** — research / startup program; may provide DGX Cloud trial credits; verify 2026 eligibility for a single academic researcher.
- **GitHub Student Developer Pack** — many cloud providers bundle free credits to verified students; verify if the researcher's university email (he has `aditya.03819051622@ipu.ac.in`) activates these.
- **JetBrains / Kaggle / Notion AI** programs — less likely for GPU but check.
- **Indian government / AICTE / NIELIT / C-DAC / NSDC / Meity** startup or research cloud programs — sometimes announced with large free GPU credits; check current status.
- **Ola Krutrim / Tata / Jio / Reliance** — Indian providers sometimes run launch-promo credits (e.g., 50–100% credit match on top-up) to capture market share; check current 2026 promotions.
- **Referral programs** — some providers (e.g., Vultr, DO, Lambda, RunPod) give $20–$200 in referral credit. Identify any that this researcher could claim without needing a referrer.

For each free-credit program, report:
- Credit amount (USD or INR).
- Expiry (calendar days from signup / first instance launch / credit grant).
- GPU restrictions (some credits exclude L4+ or limit to T4).
- Region restrictions.
- Eligibility (individual / .edu / startup / business-entity-required).
- Does claiming the credit require a card / UPI top-up first? (Some "free" credits require a $1 card authorization — which fails for Indian debit cards on Stripe, defeating the purpose.)
- Does the credit stack with other promos / referral bonuses?

**Flag any credit program that requires GitHub Student Pack verification** — the researcher has a `.ac.in` university email and can almost certainly qualify; the GH Student Pack unlocks DigitalOcean $200, MongoDB $200, Name.com domain, Stripe Atlas discount, Microsoft Azure $100, etc., several of which stack with cloud free credits.

---

## I. Deliverables — what you output back

1. **TMLR scaling-study verification verdict** (one paragraph): confirmed not required / new requirement since April 2026.
2. **Summary ranking table** of 10–20 candidate platforms with the columns in §G, sorted by `(UPI accepted?) × (free credit value) × (L4-class availability) ÷ (price per hour all-in)`.
3. **Top-3 recommendation** with full reasoning:
   - Primary (GREEN).
   - Spot/preemption alternative (YELLOW) if primary is on-demand only.
   - Fallback (YELLOW) if primary's free-credit program has a restriction that might bite.
4. **Scope-option feasibility mapping**: for each of the 4 options in §E, which platform(s) + credit combinations make it fit under ₹2,500 OOP.
5. **Hidden-cost enumeration** per §C.2 for the top 3 platforms only (one bullet per hidden-cost type).
6. **Signup path** for the #1 recommendation: exact steps to take, including what KYC document to have ready, typical approval time for an Indian individual without a business entity, and the specific URL of the billing/add-funds page where UPI should appear.
7. **Dry-run verification protocol**: before the researcher commits the full matrix, how to run a 500K-step probe (~15 min, ~₹15 on most platforms) to verify throughput vs. the 1,270 sps expected number. Include the exact `scripts/train.py` invocation with appropriate args.
8. **One hard "do not do this" list**: platforms the researcher has already verified do not work for him (Vast.ai all 3 rails, E2E out of stock, forex virtual cards, crypto workarounds), so your answer does not redundantly propose them.

---

## J. Style / answer constraints

- **Cite sources.** Every UPI-accepted claim must have a direct URL to the provider's billing-methods page or a screenshot-backed support article. Do not hallucinate UPI support — if you cannot find a first-party confirmation, mark it "unknown, requires pre-signup verification." The researcher has burned three hours on hallucinated payment rails already.
- **Do not propose forex virtual cards, Wise/Revolut routes, Niyo Global, crypto-funding workarounds, or any pivot that goes `UPI → INR wallet → USD card → Stripe → platform`.** The user has explicitly rejected these. They fail silently and add a forex spread the user cannot afford. UPI direct-to-platform only.
- **Be blunt about uncertainty.** If the Indian GPU-cloud landscape has shifted since your knowledge cutoff and you cannot verify a platform's current 2026 pricing or UPI support, say so. "Verify live before top-up" is a legitimate answer.
- **Tax-awareness:** assume 18% Indian GST on top of any INR-quoted price unless the provider explicitly states GST-inclusive. Assume 1–3% FX spread on any USD-billed platform. Present all-in rupee numbers.
- **Do not re-pitch the Vast.ai A40-interruptible plan from the prior research pass.** That plan is dead for this researcher.
- **Length:** as long as it takes to be exhaustive on platforms (§G table has 15–20 rows) and free credits (§H is its own deep-dive). Do not pad. Short on §I conclusions; long on §G / §H data.

---

## K. What success looks like

The researcher hits `Enter` on your recommended platform's signup form, funds ₹2,500 via UPI within 24 hours, verifies the free-credit grant, runs the 500K-step probe to confirm throughput, and within ~5 days has 40 headline runs + 25 ablation runs (Option 4) or ideally 50 ablation runs (Option 3) completed, aggregated, and plotted. The TMLR manuscript is then drafted around real numbers, not estimates. No additional out-of-pocket spend. No surprise bills from hidden egress / storage / IP fees. No KYC-blocked payment attempt.

Your research output is the gate between "have a plan" and "have a paper." Take it seriously.
