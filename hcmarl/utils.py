"""
Shared utility functions for HC-MARL.

Provides configuration loading, logging helpers, and numerical utilities
used across all modules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml


# =========================================================================
# Logging
# =========================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently formatted logger.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# =========================================================================
# Configuration
# =========================================================================

def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def resolve_project_root() -> Path:
    """Find the project root directory (contains setup.py or .git).

    Walks up from this file's location until it finds a marker file.

    Returns:
        Path to the project root.

    Raises:
        RuntimeError: If the root cannot be found.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "setup.py").exists() or (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Cannot locate project root (no setup.py or .git found).")


# =========================================================================
# Numerical utilities
# =========================================================================

def clip_and_normalise(x: np.ndarray) -> np.ndarray:
    """DEPRECATED — do not use for physiological state.

    This function clips to [0,1] then divides by sum, which corrupts
    the physics of the 3CC-r ODE (see audit C-1/C-2/C-5).  Use the
    conservation-preserving guard instead:
        MA = max(0, MA); MF = max(0, MF); MR = 1 - MA - MF

    Kept only for backward compatibility with non-ODE callers.
    """
    import warnings
    warnings.warn(
        "clip_and_normalise is deprecated — use conservation-preserving guard",
        DeprecationWarning,
        stacklevel=2,
    )
    x = np.clip(x, 0.0, 1.0)
    total = x.sum()
    if total > 0.0:
        x = x / total
    return x


def safe_log(x: float, floor: float = 1e-20) -> float:
    """Compute log(x) with a floor to avoid log(0).

    Used in the NSWF objective: ln(U(i,j) - Di).

    Args:
        x: Value to take log of. Must be positive for meaningful result.
        floor: Minimum value to clamp to before taking log.

    Returns:
        log(max(x, floor))
    """
    return float(np.log(max(x, floor)))


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Check if a square matrix is positive definite.

    Used for QP feasibility verification in ECBF.

    Args:
        matrix: Square numpy array.

    Returns:
        True if all eigenvalues are positive.
    """
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues > 0))
    except np.linalg.LinAlgError:
        return False


def build_per_worker_theta_max(mmicrl_results, config_theta_defaults, n_workers, method):
    """Construct per-worker theta_max dict from MMICRL results, clamped to
    the config floor per muscle.

    Single source of truth shared by train.py and evaluate.py so the env
    built at training time and the env built at evaluation time impose
    identical thresholds on the policy.

    MMICRL may return thresholds below what the env dynamics can satisfy
    (e.g. Path G single-muscle demos produce shoulder=0.21 while
    theta_min_max requires >= 0.627). We clamp each learned theta up to
    the corresponding config floor so constraints are structurally
    feasible.

    Args:
        mmicrl_results: Dict from MMICRL.fit() with 'theta_per_type' and
            'type_proportions', or None.
        config_theta_defaults: Dict {muscle: float} from config.environment.theta_max.
        n_workers: Number of workers.
        method: Only 'hcmarl' uses MMICRL thresholds; other methods get
            the flat config dict.

    Returns:
        Either a per-worker dict {'worker_0': {muscle: theta}, ...}
        clamped to floors, or the flat config_theta_defaults for non-hcmarl
        methods, or config_theta_defaults when MMICRL produced nothing.
    """
    if method != "hcmarl" or not mmicrl_results:
        return config_theta_defaults

    theta_per_type = mmicrl_results.get("theta_per_type", {})
    type_proportions = mmicrl_results.get("type_proportions", [])
    if not theta_per_type:
        return config_theta_defaults

    type_keys = sorted(theta_per_type.keys(), key=lambda k: int(k))
    n_types = len(type_keys)
    theta_max = {}

    if type_proportions and len(type_proportions) == n_types:
        counts = np.round(np.array(type_proportions) * n_workers).astype(int)
        diff = n_workers - counts.sum()
        counts[np.argmax(counts)] += diff
        worker_type_map = []
        for t_idx, count in enumerate(counts):
            worker_type_map.extend([t_idx] * count)
        for w in range(n_workers):
            type_k = type_keys[worker_type_map[w]]
            raw_theta = theta_per_type[type_k]
            clamped = {}
            for muscle, val in raw_theta.items():
                floor = config_theta_defaults.get(muscle, 0.5)
                clamped[muscle] = max(float(val), float(floor))
            theta_max[f"worker_{w}"] = clamped
    else:
        conservative = {}
        for type_k in type_keys:
            for muscle, val in theta_per_type[type_k].items():
                floor = config_theta_defaults.get(muscle, 0.5)
                clamped_val = max(float(val), float(floor))
                if muscle not in conservative or clamped_val < conservative[muscle]:
                    conservative[muscle] = clamped_val
        theta_max = {f"worker_{w}": dict(conservative) for w in range(n_workers)}

    return theta_max


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch seeding (deferred import, only available in Phase 3+)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # M-1: deterministic=True slows GPU ~1.5x but is required for
            # reproducibility in research. benchmark=False ensures consistent
            # algorithm selection across runs.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger = get_logger(__name__)
            logger.info("CUDA seed set; cudnn.deterministic=True (reproducible but ~1.5x slower)")
    except ImportError:
        pass
