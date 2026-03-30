"""
Validate MMICRL on real PAMAP2 data.

Loads the PAMAP2 Physical Activity Monitoring dataset (Reiss & Stricker, ISWC 2012),
converts it to HC-MARL demonstration format, and runs the full MMICRL pipeline:
  1. Load real IMU + heart rate data from 9 subjects
  2. Window into fixed-length trajectory segments per subject
  3. Feed into MMICRL.fit() to discover latent worker types
  4. Verify distinct per-type safety thresholds are learned

Data source: HuggingFace mirror of UCI PAMAP2 (monster-monash/PAMAP2)
Reference: Reiss & Stricker. "Introducing a New Benchmarked Dataset for
           Activity Recognition." ISWC 2012.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
from collections import defaultdict

from hcmarl.mmicrl import DemonstrationCollector, MMICRL


def load_pamap2_hf(data_dir: str, window_size: int = 50, stride: int = 25):
    """
    Load PAMAP2 from HuggingFace-format CSV + subject_id CSV.

    Returns a DemonstrationCollector populated with real demonstrations.
    """
    csv_path = os.path.join(data_dir, "features_downsampled.csv")
    subj_path = os.path.join(data_dir, "PAMAP2_subject_id.csv")
    label_path = os.path.join(data_dir, "PAMAP2_y.csv")

    # Load sensor data (40 IMU features per sample)
    print("Loading PAMAP2 sensor data...")
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([[float(x) for x in row] for row in reader], dtype=np.float32)

    # Load subject IDs
    with open(subj_path) as f:
        subj_ids = np.array([int(line.strip()) for line in f if line.strip()])

    # Load activity labels
    with open(label_path) as f:
        labels = np.array([int(line.strip()) for line in f if line.strip()])

    n = min(len(data), len(subj_ids), len(labels))
    data = data[:n]
    subj_ids = subj_ids[:n]
    labels = labels[:n]

    print(f"  Loaded {n} samples, {len(np.unique(subj_ids))} subjects, "
          f"{len(np.unique(labels))} activities")

    # Map IMU data to 3CC-r-like fatigue state: [MR, MA, MF] per body part.
    #
    # The 3CC-r model has: MR (resting) → MA (active) → MF (fatigued)
    # We simulate this from acceleration data:
    #   - Acceleration magnitude = "neural drive" (exertion level)
    #   - High exertion drives MR→MA→MF (fatigue accumulation)
    #   - Low exertion drives MF→MR (recovery)
    #
    # This captures real inter-subject differences: some subjects
    # generate higher accelerations (more vigorous movers) and
    # fatigue faster than gentler movers doing the same activity.
    #
    # Body parts: hand (shoulder proxy), chest (trunk), ankle (lower body)

    # Column indices for 6g accelerometer
    body_parts = {
        'hand': [5, 6, 7],     # handAcc6_1,2,3
        'chest': [18, 19, 20], # chestAcc6_1,2,3
        'ankle': [31, 32, 33], # ankleAcc6_1,2,3
    }
    hr_col = 0

    print(f"  Simulating 3CC-r fatigue dynamics from IMU acceleration data...")

    # Compute acceleration magnitude per body part (proxy for neural drive)
    acc_mags = {}
    for part, cols in body_parts.items():
        acc = data[:, cols]
        mag = np.sqrt(np.sum(acc ** 2, axis=1))
        # Normalise to [0, 1] where 0 = rest, 1 = max exertion
        mag_min, mag_max = mag.min(), mag.max()
        if mag_max - mag_min > 1e-8:
            mag = (mag - mag_min) / (mag_max - mag_min)
        acc_mags[part] = mag

    # Normalise heart rate to [0, 1]
    hr = data[:, hr_col]
    hr_min, hr_max = hr.min(), hr.max()
    if hr_max - hr_min > 1e-8:
        hr = (hr - hr_min) / (hr_max - hr_min)

    # Build state vectors: for each sample, compute [MR, MA, MF] per body part + HR
    # State = [hand_MR, hand_MA, hand_MF, chest_MR, chest_MA, chest_MF,
    #          ankle_MR, ankle_MA, ankle_MF, HR]
    n_features = 10
    features = np.zeros((n, n_features), dtype=np.float32)

    for part_idx, part in enumerate(['hand', 'chest', 'ankle']):
        mag = acc_mags[part]
        # Simulate fatigue dynamics per-subject
        for subj in np.unique(subj_ids):
            mask = subj_ids == subj
            subj_mag = mag[mask]

            # Simplified 3CC-r simulation
            mr = np.ones(len(subj_mag), dtype=np.float32)
            ma = np.zeros(len(subj_mag), dtype=np.float32)
            mf = np.zeros(len(subj_mag), dtype=np.float32)

            # Parameters (from 3CC-r model, simplified)
            r = 15.0   # recovery rate (grip, from Looft et al. 2018)
            F = 0.0069 # fatigue rate
            R = 0.0024 # recovery rate

            for t in range(1, len(subj_mag)):
                u = subj_mag[t]  # neural drive from acceleration
                dt = 1.0  # time step

                # 3CC-r ODEs (simplified Euler)
                dmr = -u * mr[t-1] * r + R * mf[t-1]
                dma = u * mr[t-1] * r - F * ma[t-1]
                dmf = F * ma[t-1] - R * mf[t-1]

                mr[t] = np.clip(mr[t-1] + dmr * dt * 0.01, 0, 1)
                ma[t] = np.clip(ma[t-1] + dma * dt * 0.01, 0, 1)
                mf[t] = np.clip(mf[t-1] + dmf * dt * 0.01, 0, 1)

                # Renormalise to maintain MR + MA + MF ≈ 1
                total = mr[t] + ma[t] + mf[t]
                if total > 1e-8:
                    mr[t] /= total
                    ma[t] /= total
                    mf[t] /= total

            features[mask, part_idx*3] = mr
            features[mask, part_idx*3 + 1] = ma
            features[mask, part_idx*3 + 2] = mf

    features[:, 9] = hr[:n]

    print(f"  Using {n_features} features: [MR,MA,MF]×3_body_parts + HR")

    # Group by subject and activity, then window into trajectories
    collector = DemonstrationCollector(n_muscles=3)

    subjects = np.unique(subj_ids)
    activities = np.unique(labels)

    for subj in subjects:
        subj_mask = subj_ids == subj
        for act in activities:
            act_mask = labels == act
            combined_mask = subj_mask & act_mask
            segment = features[combined_mask]

            if len(segment) < window_size:
                continue

            # Window into trajectory segments
            for start in range(0, len(segment) - window_size + 1, stride):
                window = segment[start:start + window_size]

                # Create (state, action) pairs
                trajectory = []
                for t in range(len(window)):
                    state = window[t]  # 10-dim: [hand3, chest3, ankle3, hr]
                    # Derive action from acceleration magnitude change
                    acc_mag = np.sqrt(np.sum(state[:3] ** 2))
                    if t > 0:
                        prev_mag = np.sqrt(np.sum(window[t-1, :3] ** 2))
                        action = 0 if acc_mag > prev_mag else 3  # work vs rest
                    else:
                        action = 0
                    trajectory.append((state.copy(), action))

                collector.demonstrations.append(trajectory)
                collector.worker_ids.append(f"subject_{subj}")

    return collector


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "pamap2")

    print("=" * 70)
    print("MMICRL Validation on Real PAMAP2 Data")
    print("=" * 70)

    # Step 1: Load real data
    collector = load_pamap2_hf(data_dir, window_size=50, stride=25)
    n_demos = len(collector.demonstrations)
    unique_workers = set(collector.worker_ids)
    print(f"\n  Total trajectories: {n_demos}")
    print(f"  Unique subjects: {sorted(unique_workers)}")

    assert n_demos > 0, "No demonstrations loaded!"
    assert len(unique_workers) >= 2, "Need at least 2 unique subjects"

    # Step 2: Run MMICRL
    print("\n" + "-" * 70)
    print("Running MMICRL fit()...")
    print("-" * 70)

    mmicrl = MMICRL(n_types=3, n_muscles=3, lambda1=1.0, lambda2=1.0)
    result = mmicrl.fit(collector)

    print(f"\n  Discovered {result['n_types_discovered']} types")
    print(f"  Mutual Information I(tau;z) = {result['mutual_information']:.4f}")
    print(f"  Objective value = {result['objective_value']:.4f}")
    print(f"  Type proportions: {result['type_proportions']}")

    # Step 3: Verify results
    print("\n" + "-" * 70)
    print("Per-type learned thresholds:")
    print("-" * 70)

    theta_per_type = result['theta_per_type']
    for k in sorted(theta_per_type.keys()):
        thetas = theta_per_type[k]
        print(f"  Type {k}: {thetas}")

    # Verification checks
    print("\n" + "-" * 70)
    print("Verification:")
    print("-" * 70)

    # Check 1: MI should be positive (types are informative)
    mi = result['mutual_information']
    assert mi > 0, f"MI should be positive, got {mi}"
    print(f"  [PASS] MI > 0: {mi:.4f}")

    # Check 2: All types should have non-empty clusters
    props = result['type_proportions']
    for k, p in enumerate(props):
        assert p > 0.01, f"Type {k} has proportion {p} < 1%"
    print(f"  [PASS] All types non-degenerate: {[f'{p:.2%}' for p in props]}")

    # Check 3: Types should have DISTINCT thresholds (not all identical)
    shoulder_thetas = [theta_per_type[k]['shoulder'] for k in range(3)]
    theta_range = max(shoulder_thetas) - min(shoulder_thetas)
    assert theta_range > 0.01, f"Shoulder thresholds too similar: {shoulder_thetas}"
    print(f"  [PASS] Distinct shoulder thresholds: {[f'{t:.3f}' for t in shoulder_thetas]} "
          f"(range={theta_range:.3f})")

    elbow_thetas = [theta_per_type[k]['elbow'] for k in range(3)]
    grip_thetas = [theta_per_type[k]['grip'] for k in range(3)]
    print(f"  [INFO] Elbow thresholds: {[f'{t:.3f}' for t in elbow_thetas]}")
    print(f"  [INFO] Grip thresholds: {[f'{t:.3f}' for t in grip_thetas]}")

    # Check 4: Thresholds should be in valid range [0.1, 0.95]
    for k in range(3):
        for muscle, theta in theta_per_type[k].items():
            assert 0.1 <= theta <= 0.95, f"Type {k} {muscle} theta={theta} out of range"
    print(f"  [PASS] All thresholds in [0.1, 0.95]")

    # Check 5: Cross-check subject distribution across types
    assignments = result.get('type_assignments', mmicrl.type_assignments)
    if assignments is not None:
        print(f"\n  Subject -> Type mapping (sample):")
        subj_type_counts = defaultdict(lambda: defaultdict(int))
        for i, (worker_id, type_k) in enumerate(zip(collector.worker_ids, assignments)):
            subj_type_counts[worker_id][int(type_k)] += 1

        for subj in sorted(subj_type_counts.keys()):
            counts = subj_type_counts[subj]
            total = sum(counts.values())
            dist = {k: f"{v/total:.0%}" for k, v in sorted(counts.items())}
            dominant = max(counts, key=counts.get)
            print(f"    {subj}: dominant=Type {dominant} | distribution={dist}")

    print("\n" + "=" * 70)
    print("MMICRL PAMAP2 VALIDATION: ALL CHECKS PASSED")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = main()
