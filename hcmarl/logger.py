"""
HC-MARL Phase 2 (#27): W&B Integration Logger
Logs 9 metrics per step: violation_rate, cumulative_cost, safety_rate,
tasks_completed, cumulative_reward, jain_fairness, peak_fatigue,
forced_rest_rate, constraint_recovery_time.
"""
import numpy as np
from typing import Dict, List, Optional, Any
import csv
import os
from collections import defaultdict


class HCMARLLogger:
    """Unified logger supporting W&B, CSV, and console output."""

    METRIC_NAMES = [
        "violation_rate", "cumulative_cost", "safety_rate",
        "tasks_completed", "cumulative_reward", "jain_fairness",
        "peak_fatigue", "forced_rest_rate", "constraint_recovery_time",
    ]

    def __init__(self, log_dir="logs", use_wandb=False, wandb_project="hcmarl",
                 wandb_entity=None, run_name=None, config=None):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.step_count = 0
        self.episode_count = 0
        self.history = defaultdict(list)
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self._csv_initialized = False

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, entity=wandb_entity,
                           name=run_name, config=config)
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, falling back to CSV-only logging")
                self.use_wandb = False

    def log_step(self, metrics: Dict[str, float]):
        self.step_count += 1
        for k, v in metrics.items():
            self.history[k].append(v)
        if self.use_wandb:
            self.wandb.log(metrics, step=self.step_count)

    def log_episode(self, metrics: Dict[str, float]):
        self.episode_count += 1
        metrics["episode"] = self.episode_count
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
                writer.writeheader()
            self._csv_initialized = True
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in metrics.items()})
        if self.use_wandb:
            self.wandb.log(metrics, step=self.episode_count)

    def compute_episode_metrics(self, episode_data: Dict) -> Dict[str, float]:
        """Compute all 9 HC-MARL metrics from raw episode data."""
        n_steps = episode_data.get("n_steps", 1)
        n_workers = episode_data.get("n_workers", 1)
        n_muscles = episode_data.get("n_muscles", 6)
        total_slots = n_steps * n_workers * n_muscles

        metrics = {}
        metrics["violation_rate"] = episode_data.get("total_violations", 0) / max(1, total_slots)
        metrics["cumulative_cost"] = float(episode_data.get("total_violations", 0))
        metrics["safety_rate"] = episode_data.get("safe_steps", 0) / max(1, n_steps)
        metrics["tasks_completed"] = float(episode_data.get("tasks_completed", 0))
        metrics["cumulative_reward"] = float(episode_data.get("total_reward", 0.0))

        tasks_per_worker = episode_data.get("tasks_per_worker", np.ones(n_workers))
        tpw = np.array(tasks_per_worker)
        n = len(tpw)
        metrics["jain_fairness"] = float((tpw.sum()**2) / (n * (tpw**2).sum() + 1e-8)) if tpw.sum() > 0 else 1.0
        metrics["peak_fatigue"] = float(episode_data.get("peak_fatigue", 0.0))
        metrics["forced_rest_rate"] = episode_data.get("forced_rests", 0) / max(1, n_steps * n_workers)
        recovery = episode_data.get("recovery_times", [])
        metrics["constraint_recovery_time"] = float(np.mean(recovery)) if recovery else 0.0
        return metrics

    def close(self):
        if self.use_wandb:
            self.wandb.finish()
