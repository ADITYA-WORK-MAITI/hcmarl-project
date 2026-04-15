"""Statistical aggregation helpers for headline + ablation reporting.

Implements the rliable recipe (Agarwal et al. NeurIPS 2021) without taking a
hard dependency on the `rliable` package:
    - Interquartile Mean (IQM): mean of the middle 50% of scores, a robust
      point estimate that ignores the top/bottom quartile noise floor.
    - Stratified bootstrap 95% CI: resample within-method, compute IQM
      on each resample, take 2.5th and 97.5th percentiles.

The function signatures match rliable.library.get_interval_estimates so a
future migration is a one-line swap. Numpy is the only dependency.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def iqm(scores: np.ndarray) -> float:
    """Interquartile mean: arithmetic mean of values between the 25th and
    75th percentiles. For fewer than 4 samples, falls back to the plain
    mean (no middle-50% to take)."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(arr.mean())
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    mask = (arr >= q1) & (arr <= q3)
    if not mask.any():
        return float(arr.mean())
    return float(arr[mask].mean())


def stratified_bootstrap_iqm_ci(
    scores: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int | None = 20260416,
) -> Tuple[float, float]:
    """Stratified bootstrap 95% CI for the IQM.

    Stratification: samples are resampled WITH replacement from the same
    method's score vector. With a single method this is ordinary bootstrap;
    with multiple methods, call this per-method and report each CI.

    Args:
        scores: 1-D array of per-seed scores for a single method.
        n_resamples: number of bootstrap resamples.
        ci: confidence level (default 0.95).
        seed: rng seed for reproducibility.

    Returns:
        (lo, hi) tuple — the CI bounds on IQM.
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    estimates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        estimates[i] = iqm(arr[idx])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(estimates, 100.0 * alpha))
    hi = float(np.percentile(estimates, 100.0 * (1.0 - alpha)))
    return (lo, hi)


def aggregate_by_method(
    scores_by_method: Dict[str, np.ndarray],
    n_resamples: int = 10_000,
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper: {method: 1-D scores} -> {method: {iqm, ci_lo, ci_hi}}."""
    out: Dict[str, Dict[str, float]] = {}
    for method, scores in scores_by_method.items():
        point = iqm(scores)
        lo, hi = stratified_bootstrap_iqm_ci(scores, n_resamples=n_resamples)
        out[method] = {"iqm": point, "ci_lo": lo, "ci_hi": hi,
                       "n_seeds": int(np.asarray(scores).size)}
    return out
