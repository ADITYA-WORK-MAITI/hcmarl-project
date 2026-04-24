# Results 1/ — Experiment 1 VM deliverable

This folder is populated by the VM during Experiment 1. Training writes
DIRECTLY into `Results 1/logs/<method>/seed_<s>/` AND
`Results 1/checkpoints/<method>/seed_<s>/` from the first step via two
symlinks set up in `RUNBOOK_EXP1.md` STEP 7:

```
/root/hcmarl_project/logs         -> /root/hcmarl_project/Results 1/logs
/root/hcmarl_project/checkpoints  -> /root/hcmarl_project/Results 1/checkpoints
```

Metadata (provenance, frozen configs, aggregation CSV, file index) is
added by STEP 12 after the grid finishes.

## At experiment end this folder contains

```
Results 1/
├── logs/
│   ├── hcmarl/seed_{0..9}/
│   │   ├── training_log.csv
│   │   ├── summary.json
│   │   └── mmicrl_results.json         (HCMARL only)
│   ├── mappo/seed_{0..9}/
│   │   ├── training_log.csv
│   │   └── summary.json
│   ├── ippo/seed_{0..9}/                (parameter-shared IPPO, Yu et al. 2022)
│   │   ├── training_log.csv
│   │   └── summary.json
│   └── mappo_lag/seed_{0..9}/
│       ├── training_log.csv
│       └── summary.json
├── checkpoints/
│   ├── hcmarl/seed_{0..9}/
│   │   ├── checkpoint_<step>.pt        (one per checkpoint_interval)
│   │   ├── checkpoint_best.pt
│   │   ├── checkpoint_final.pt
│   │   └── run_state.pt                (counters + RNG for bit-exact resume)
│   ├── mappo/seed_{0..9}/               (same structure)
│   ├── ippo/seed_{0..9}/                (same)
│   └── mappo_lag/seed_{0..9}/           (same)
├── _configs_snapshot/
│   ├── experiment_matrix.yaml
│   ├── hcmarl_full_config.yaml
│   ├── mappo_config.yaml
│   ├── ippo_config.yaml
│   └── mappo_lag_config.yaml
├── _provenance.txt                     (git hash, torch version, hardware)
├── _aggregation_summary.csv            (40 rows: method, seed, best_reward,
│                                        final_cost_ema, final_safety_rate,
│                                        budget_tripped, lazy_tripped)
├── _exp1_run.log                       (launcher stdout, teed live)
└── _INDEX.txt                          (file listing)
```

Expected on-disk size: **~2-3 GB** (2 GB checkpoints + ~30 MB CSVs/JSONs
+ metadata).

## Pulling this folder to the laptop

All three pulls land in `/c/Users/admin/Downloads/`. Move them wherever
locally after the pulls complete.

```bash
# PRIMARY — single pull gets everything (~2-3 GB)
rsync -avzP -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/"Results 1" \
    /c/Users/admin/Downloads/

# BELT-AND-BRACES #1 — independent mirror of logs/ (~50 MB)
rsync -avzP -L -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/logs/ \
    /c/Users/admin/Downloads/logs_exp1_mirror/

# BELT-AND-BRACES #2 — independent mirror of checkpoints/ (~2 GB)
rsync -avzP -L -e "ssh -i ~/.ssh/id_ed25519" \
    root@<public-ip>:/root/hcmarl_project/checkpoints/ \
    /c/Users/admin/Downloads/checkpoints_exp1_mirror/
```

After these three commands `/c/Users/admin/Downloads/` has three
independent copies of the same training output:

- `Results 1/` — canonical deliverable (CSVs + JSONs + .pt weights + metadata)
- `logs_exp1_mirror/` — second copy of CSVs + JSONs + MMICRL results
- `checkpoints_exp1_mirror/` — second copy of .pt weights

The `-L` flag on the two mirror commands dereferences the top-level
symlinks on the VM so the pull copies real files (the primary pull does
not need `-L` because `Results 1/logs/` and `Results 1/checkpoints/`
are the symlink *targets*, i.e., real directories on the VM).

## Format notes (per user instruction)

Every file in this folder is:

- **Human-readable** — CSVs open in any spreadsheet; JSONs pretty-print.
- **Python-analyzable** — all numeric data in CSV / JSON with labelled
  columns. `training_log.csv` has one row per eval episode with
  `(episode, global_step, reward, cost, safety_rate, peak_MF, ...)`.
- **Claude-interpretable** — plain text + structured data throughout.
- **Visualizable** — tabular rows ready for pandas / matplotlib / seaborn.

No visualization / analysis / interpretation happens on the VM. Those
are deferred to **Experiment 4** on the laptop.
