"""
HC-MARL Data Loaders — Real Demonstration Datasets
====================================================
Load real human demonstration data from published datasets for
use with MM-ICRL and the HC-MARL training pipeline.

Supported datasets:
  1. RoboMimic (Mandlekar et al., CoRL 2021) — multi-proficiency human teleop demos
  2. D4RL Adroit (Fu et al., 2020) — real human CyberGlove teleop demos
  3. PAMAP2 (Reiss & Stricker, 2012) — IMU + heart rate physical activity data

Each loader returns data in the HC-MARL standard format:
  List[Dict] where each dict has:
    - 'states': np.ndarray of shape (T, state_dim)
    - 'actions': np.ndarray of shape (T,) or (T, action_dim)
    - 'worker_id': int or str identifying the demonstrator
    - 'metadata': dict with dataset-specific info
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings


SUPPORTED_DATASETS = {
    "robomimic": {
        "url": "https://robomimic.github.io/",
        "description": "Multi-proficiency human teleop demos (Mandlekar et al. CoRL 2021)",
        "format": "HDF5",
        "tasks": ["Lift", "Can", "Square", "Transport", "ToolHang"],
    },
    "d4rl_adroit": {
        "url": "https://github.com/Farama-Foundation/D4RL",
        "description": "Real human CyberGlove teleop demos (Fu et al. 2020)",
        "format": "HDF5 via d4rl API",
        "tasks": ["pen-human-v1", "hammer-human-v1", "door-human-v1", "relocate-human-v1"],
    },
    "pamap2": {
        "url": "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring",
        "description": "IMU + heart rate from 9 subjects, 18 activities (Reiss & Stricker 2012)",
        "format": "CSV/DAT",
        "tasks": ["lying", "sitting", "standing", "walking", "running", "cycling",
                  "rope_jumping", "ironing", "vacuum_cleaning", "ascending_stairs",
                  "descending_stairs", "Nordic_walking"],
    },
}


def _validate_path(path: str, dataset_name: str) -> Path:
    """Validate that data path exists."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {path}\n"
            f"Download {dataset_name} from: {SUPPORTED_DATASETS[dataset_name]['url']}"
        )
    return p


def load_robomimic_demos(
    path: str,
    task: str = "Lift",
    proficiency: Optional[str] = None,
    max_demos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load RoboMimic multi-proficiency human demonstration data.

    Args:
        path: Path to RoboMimic HDF5 file (e.g., low_dim_v141.hdf5)
        task: Task name (Lift, Can, Square, Transport, ToolHang)
        proficiency: Filter by proficiency level ('worse', 'okay', 'better') or None for all
        max_demos: Maximum number of demos to load

    Returns:
        List of trajectory dicts in HC-MARL standard format

    Reference:
        Mandlekar et al. "What Matters in Learning from Offline Human
        Demonstrations for Robot Manipulation." CoRL 2021.
        Site: https://robomimic.github.io/
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py required for RoboMimic loading. Install: pip install h5py"
        )

    data_path = _validate_path(path, "robomimic")
    trajectories = []

    with h5py.File(data_path, "r") as f:
        demos_group = f.get("data", f)
        demo_keys = sorted([k for k in demos_group.keys() if k.startswith("demo")])

        if max_demos:
            demo_keys = demo_keys[:max_demos]

        for demo_key in demo_keys:
            demo = demos_group[demo_key]

            # Extract observations and actions
            obs = demo["obs"]
            states = np.array(obs.get("object", obs.get("flat", list(obs.values())[0])))
            actions = np.array(demo["actions"])

            # Get proficiency label if available
            attrs = dict(demo.attrs) if hasattr(demo, 'attrs') else {}
            demo_proficiency = attrs.get("proficiency", "unknown")

            if proficiency and demo_proficiency != proficiency:
                continue

            trajectories.append({
                "states": states.astype(np.float32),
                "actions": actions.astype(np.float32),
                "worker_id": attrs.get("operator_id", demo_key),
                "metadata": {
                    "dataset": "robomimic",
                    "task": task,
                    "proficiency": demo_proficiency,
                    "demo_key": demo_key,
                    "n_steps": len(actions),
                },
            })

    if not trajectories:
        raise ValueError(
            f"No demos found in {path}. "
            f"Check file structure or proficiency filter='{proficiency}'."
        )

    return trajectories


def load_d4rl_demos(
    env_name: str = "pen-human-v1",
    max_demos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load D4RL Adroit human demonstration data.

    Args:
        env_name: D4RL environment name. Human demos:
            pen-human-v1, hammer-human-v1, door-human-v1, relocate-human-v1
        max_demos: Maximum number of demos to load

    Returns:
        List of trajectory dicts in HC-MARL standard format

    Reference:
        Fu et al. "D4RL: Datasets for Deep Data-Driven Reinforcement
        Learning." arXiv 2004.07219, 2020.
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError(
            "gymnasium required for D4RL loading. Install: pip install gymnasium"
        )

    try:
        import d4rl  # noqa: F401 — registers envs as side effect
    except ImportError:
        raise ImportError(
            "d4rl required. Install: pip install d4rl\n"
            "Or use Minari (modern replacement): pip install minari"
        )

    env = gym.make(env_name)
    dataset = env.get_dataset()

    # D4RL stores flat arrays; split into trajectories at terminal/timeout flags
    states = dataset["observations"]
    actions = dataset["actions"]
    terminals = dataset.get("terminals", np.zeros(len(states), dtype=bool))
    timeouts = dataset.get("timeouts", np.zeros(len(states), dtype=bool))

    trajectories = []
    traj_start = 0

    for i in range(len(states)):
        if terminals[i] or timeouts[i] or i == len(states) - 1:
            end = i + 1
            traj = {
                "states": states[traj_start:end].astype(np.float32),
                "actions": actions[traj_start:end].astype(np.float32),
                "worker_id": f"d4rl_operator_{len(trajectories) % 5}",
                "metadata": {
                    "dataset": "d4rl",
                    "env_name": env_name,
                    "terminal": bool(terminals[i]),
                    "n_steps": end - traj_start,
                },
            }
            trajectories.append(traj)
            traj_start = end

            if max_demos and len(trajectories) >= max_demos:
                break

    return trajectories


def load_pamap2(
    path: str,
    subjects: Optional[List[int]] = None,
    activities: Optional[List[int]] = None,
    window_size: int = 256,
    stride: int = 128,
    include_hr: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load PAMAP2 Physical Activity Monitoring dataset.

    Extracts IMU (acceleration, gyroscope) and heart rate data,
    windowed into fixed-length segments for trajectory format.

    Args:
        path: Path to PAMAP2 Protocol/ directory containing subject*.dat files
        subjects: List of subject IDs (1-9) to load, or None for all
        activities: List of activity IDs to load, or None for all
        window_size: Number of timesteps per trajectory window (at 100Hz)
        stride: Window stride for overlap
        include_hr: Whether to include heart rate in state vector

    Returns:
        List of trajectory dicts in HC-MARL standard format

    Reference:
        Reiss & Stricker. "Introducing a New Benchmarked Dataset for
        Activity Recognition." ISWC 2012.
        URL: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

    PAMAP2 column layout (54 columns):
        0: timestamp  1: activityID  2: heart_rate
        3-19: hand IMU (temp, acc*3, acc_16g*3, gyro*3, mag*3, orientation*4)
        20-36: chest IMU
        37-53: ankle IMU
    """
    data_path = _validate_path(path, "pamap2")
    trajectories = []

    # Activity labels from PAMAP2 documentation
    activity_names = {
        1: "lying", 2: "sitting", 3: "standing", 4: "walking",
        5: "running", 6: "cycling", 7: "Nordic_walking",
        12: "ascending_stairs", 13: "descending_stairs",
        16: "vacuum_cleaning", 17: "ironing", 24: "rope_jumping",
    }

    # IMU column indices (acceleration only for compactness)
    # Hand: cols 4-6, Chest: cols 21-23, Ankle: cols 38-40
    imu_cols = [4, 5, 6, 21, 22, 23, 38, 39, 40]
    hr_col = 2
    activity_col = 1

    if subjects is None:
        subjects = list(range(1, 10))

    for subj_id in subjects:
        fpath = data_path / f"subject10{subj_id}.dat"
        if not fpath.exists():
            fpath = data_path / f"subject{subj_id:02d}.dat"
        if not fpath.exists():
            # Try common naming variants
            candidates = list(data_path.glob(f"*subject*{subj_id}*"))
            if candidates:
                fpath = candidates[0]
            else:
                warnings.warn(f"Subject {subj_id} file not found in {data_path}, skipping")
                continue

        # Load data (space-separated, NaN for missing)
        try:
            raw = np.loadtxt(str(fpath), dtype=np.float64)
        except Exception as e:
            warnings.warn(f"Failed to load {fpath}: {e}")
            continue

        # Replace NaN with 0
        raw = np.nan_to_num(raw, nan=0.0)

        # Get unique activities in this subject's data
        subj_activities = np.unique(raw[:, activity_col].astype(int))

        for act_id in subj_activities:
            if act_id == 0:
                continue  # transient / no activity
            if activities and act_id not in activities:
                continue

            # Extract segments for this activity
            mask = raw[:, activity_col].astype(int) == act_id
            act_data = raw[mask]

            if len(act_data) < window_size:
                continue

            # Build state vectors: IMU features + optional heart rate
            state_cols = list(imu_cols)
            if include_hr:
                state_cols = [hr_col] + state_cols

            states = act_data[:, state_cols].astype(np.float32)

            # Normalise per-column to [0, 1]
            col_min = states.min(axis=0)
            col_max = states.max(axis=0)
            col_range = col_max - col_min
            col_range[col_range < 1e-8] = 1.0
            states = (states - col_min) / col_range

            # Window into trajectory segments
            for start in range(0, len(states) - window_size + 1, stride):
                window_states = states[start:start + window_size]

                # Derive pseudo-actions from acceleration magnitude changes
                acc_mag = np.sqrt(np.sum(window_states[:, -9:-6] ** 2, axis=1))
                actions = np.clip(np.diff(acc_mag, prepend=acc_mag[0]), -1, 1)

                trajectories.append({
                    "states": window_states,
                    "actions": actions.astype(np.float32),
                    "worker_id": f"subject_{subj_id}",
                    "metadata": {
                        "dataset": "pamap2",
                        "subject_id": subj_id,
                        "activity_id": act_id,
                        "activity_name": activity_names.get(act_id, f"unknown_{act_id}"),
                        "window_start": start,
                        "n_steps": window_size,
                    },
                })

    return trajectories


def load_dataset(
    source: str,
    path: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Unified loader entry point.

    Args:
        source: Dataset name ('robomimic', 'd4rl_adroit', 'pamap2')
        path: Path to dataset files (required for robomimic, pamap2)
        **kwargs: Dataset-specific arguments

    Returns:
        List of trajectory dicts in HC-MARL standard format
    """
    loaders = {
        "robomimic": load_robomimic_demos,
        "d4rl_adroit": load_d4rl_demos,
        "d4rl": load_d4rl_demos,
        "pamap2": load_pamap2,
    }

    if source not in loaders:
        raise ValueError(
            f"Unknown dataset: {source}. Supported: {list(loaders.keys())}"
        )

    loader = loaders[source]

    if source in ("robomimic", "pamap2"):
        if path is None:
            raise ValueError(f"path required for {source} dataset")
        return loader(path=path, **kwargs)
    else:
        return loader(**kwargs)
