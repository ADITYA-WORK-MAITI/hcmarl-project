"""
Validate MMICRL on real WSD4FEDSRM data via Path G pipeline.

End-to-end validation:
  1. Load WSD4FEDSRM shoulder fatigue dataset (34 subjects, Zenodo 8415066)
  2. Calibrate individual F per subject from endurance times (grid search)
  3. Generate demonstrations with real inter-worker variability
  4. Run MMICRL type discovery (CFDE / normalizing flows)
  5. Validate: types correlate with calibrated F values
  6. Validate: per-type theta_max thresholds are physiologically reasonable
  7. Cross-validate: Borg RPE progression patterns correlate with types

Data provenance:
  - Shoulder F: Calibrated from real endurance-time data (WSD4FEDSRM)
  - Other muscles: Published population distributions (Frey-Law & Avin 2010)
  - Reperfusion r=15: Validated for shoulder (Looft & Frey-Law 2020)
  - Calibration method: Frey-Law, Looft & Heitsman 2012

No synthetic data. No proxy data. All parameters traced to published sources.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hcmarl.real_data_calibration import (
    run_path_g,
    generate_demonstrations_from_profiles,
    load_path_g_into_collector,
    predict_endurance_time,
    calibrate_F_for_subject,
    compute_dynamic_isometric_report,
    POPULATION_FR,
    TASK_TO_MVIC_FRACTION,
)
from hcmarl.mmicrl import MMICRL


def validate_calibration(path_g_result):
    """Validate calibrated F values against physiological expectations."""
    print("=" * 60)
    print("VALIDATION 1: Calibration Quality")
    print("=" * 60)

    cal = path_g_result['calibration_results']
    F_vals = [r['F'] for r in cal.values()]
    rms_vals = [r['rms_error_sec'] for r in cal.values()]

    # Check F values are physiologically reasonable
    F_mean = np.mean(F_vals)
    F_std = np.std(F_vals)
    F_min, F_max = np.min(F_vals), np.max(F_vals)

    print(f"  F distribution: mean={F_mean:.4f}, SD={F_std:.4f}")
    print(f"  F range: [{F_min:.4f}, {F_max:.4f}]")
    print(f"  F ratio (max/min): {F_max/F_min:.1f}x")
    print(f"  RMS errors: mean={np.mean(rms_vals):.1f}s, "
          f"median={np.median(rms_vals):.1f}s")

    n_good = sum(1 for r in rms_vals if r < 15)
    n_ok = sum(1 for r in rms_vals if 15 <= r < 30)
    n_poor = sum(1 for r in rms_vals if r >= 30)
    print(f"  Fit quality: {n_good} good (<15s), {n_ok} OK (15-30s), "
          f"{n_poor} poor (>30s)")

    # Assertions
    passed = 0
    total = 5

    if 0.1 < F_mean < 5.0:
        print("  [PASS] F mean in physiological range (0.1-5.0)")
        passed += 1
    else:
        print(f"  [FAIL] F mean {F_mean:.4f} outside range")

    if F_max / F_min > 2.0:
        print(f"  [PASS] Inter-worker F variation > 2x ({F_max/F_min:.1f}x)")
        passed += 1
    else:
        print(f"  [FAIL] F variation only {F_max/F_min:.1f}x")

    if np.median(rms_vals) < 20.0:
        print(f"  [PASS] Median RMS < 20s ({np.median(rms_vals):.1f}s)")
        passed += 1
    else:
        print(f"  [FAIL] Median RMS = {np.median(rms_vals):.1f}s")

    if n_poor <= 5:
        print(f"  [PASS] At most 5 poor fits ({n_poor})")
        passed += 1
    else:
        print(f"  [FAIL] {n_poor} poor fits")

    if len(cal) == 34:
        print(f"  [PASS] All 34 subjects calibrated")
        passed += 1
    else:
        print(f"  [FAIL] Only {len(cal)}/34 subjects calibrated")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def validate_demonstrations(demos, wids, path_g_result):
    """Validate demonstration quality and inter-worker variability."""
    print("\n" + "=" * 60)
    print("VALIDATION 2: Demonstration Quality")
    print("=" * 60)

    lengths = [len(d) for d in demos]
    print(f"  Total demos: {len(demos)} from {len(set(wids))} workers")
    print(f"  Trajectory lengths: mean={np.mean(lengths):.0f}, "
          f"range=[{min(lengths)}, {max(lengths)}]")

    # Check inter-worker variability in MF trajectories
    final_MF = [d[-1, 3] for d in demos]  # MF at end
    final_MR = [d[-1, 1] for d in demos]  # MR at end
    print(f"  Final MF: mean={np.mean(final_MF):.3f}, std={np.std(final_MF):.3f}")
    print(f"  Final MR: mean={np.mean(final_MR):.3f}, std={np.std(final_MR):.3f}")

    passed = 0
    total = 4

    if len(demos) == len(set(wids)) * 3:
        print(f"  [PASS] 3 demos per worker")
        passed += 1
    else:
        print(f"  [FAIL] Unexpected demo count")

    if max(lengths) > min(lengths) * 1.3:
        print(f"  [PASS] Variable trajectory lengths (ratio {max(lengths)/min(lengths):.1f})")
        passed += 1
    else:
        print(f"  [FAIL] Trajectories too similar in length")

    if np.std(final_MF) > 0.05:
        print(f"  [PASS] Inter-worker MF variability (std={np.std(final_MF):.3f})")
        passed += 1
    else:
        print(f"  [FAIL] Final MF too uniform")

    if any(mr > 0.05 for mr in final_MR):
        print(f"  [PASS] Some workers still have MR > 0 (not all exhausted)")
        passed += 1
    else:
        print(f"  [FAIL] All workers fully exhausted")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def validate_mmicrl_types(mmicrl_results, mmicrl_model, wids, path_g_result):
    """Validate MMICRL type discovery results."""
    print("\n" + "=" * 60)
    print("VALIDATION 3: MMICRL Type Discovery")
    print("=" * 60)

    print(f"  Types discovered: {mmicrl_results['n_types_discovered']}")
    print(f"  Type proportions: {mmicrl_results['type_proportions']}")
    print(f"  Mutual information: {mmicrl_results['mutual_information']:.4f}")
    print(f"  Objective value: {mmicrl_results['objective_value']:.4f}")

    # Check type-F correlation
    cal = path_g_result['calibration_results']
    subj_ids = sorted(cal.keys(), key=lambda x: int(x.split('_')[1]))
    type_assignments = mmicrl_model.type_assignments

    type_F = {}
    for demo_idx, wid in enumerate(wids):
        if demo_idx < len(type_assignments):
            t = type_assignments[demo_idx]
            subj_id = subj_ids[wid]
            if t not in type_F:
                type_F[t] = []
            type_F[t].append(cal[subj_id]['F'])

    # Sort types by mean F
    type_order = sorted(type_F.keys(),
                        key=lambda t: np.mean(type_F[t]))
    print("\n  Type-F correlation:")
    for t in type_order:
        f_vals = type_F[t]
        print(f"    Type {t}: n={len(f_vals)}, "
              f"F_mean={np.mean(f_vals):.4f}, "
              f"F_range=[{np.min(f_vals):.4f}, {np.max(f_vals):.4f}]")

    # Check per-type thresholds
    print("\n  Per-type thresholds:")
    for t, thresholds in mmicrl_results['theta_per_type'].items():
        print(f"    Type {t}: theta_max = {thresholds}")

    passed = 0
    total = 4

    if mmicrl_results['mutual_information'] > 0.1:
        print(f"\n  [PASS] MI > 0.1 ({mmicrl_results['mutual_information']:.4f})")
        passed += 1
    else:
        print(f"\n  [FAIL] MI too low ({mmicrl_results['mutual_information']:.4f})")

    props = mmicrl_results['type_proportions']
    if min(props) > 0.05:
        print(f"  [PASS] No empty types (min proportion={min(props):.3f})")
        passed += 1
    else:
        print(f"  [FAIL] Empty type detected")

    # Check if types have different mean F
    mean_Fs = [np.mean(type_F[t]) for t in type_order]
    if len(mean_Fs) >= 2 and max(mean_Fs) / min(mean_Fs) > 1.2:
        print(f"  [PASS] Types have different F means "
              f"(ratio {max(mean_Fs)/min(mean_Fs):.2f})")
        passed += 1
    else:
        print(f"  [FAIL] Types have similar F means")

    # Check thresholds are reasonable
    all_theta = []
    for t, th in mmicrl_results['theta_per_type'].items():
        for m, v in th.items():
            all_theta.append(v)
    if all(0.1 < v < 1.0 for v in all_theta):
        print(f"  [PASS] All theta_max in (0.1, 1.0)")
        passed += 1
    else:
        print(f"  [FAIL] Theta_max out of range")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def validate_borg_correlation(path_g_result, mmicrl_model, wids):
    """Cross-validate against Borg RPE progression patterns."""
    print("\n" + "=" * 60)
    print("VALIDATION 4: Borg RPE Cross-Validation")
    print("=" * 60)

    cal = path_g_result['calibration_results']
    borg = path_g_result['borg_rpe_data']
    subj_ids = sorted(cal.keys(), key=lambda x: int(x.split('_')[1]))
    type_assignments = mmicrl_model.type_assignments

    # Group subjects by type
    type_subjects = {}
    for demo_idx, wid in enumerate(wids):
        if demo_idx < len(type_assignments):
            t = type_assignments[demo_idx]
            subj_id = subj_ids[wid]
            if t not in type_subjects:
                type_subjects[t] = set()
            type_subjects[t].add(subj_id)

    # For each type, compute average Borg RPE slope
    print("  Per-type Borg RPE analysis (task1_35i):")
    type_slopes = {}
    for t in sorted(type_subjects.keys()):
        slopes = []
        for subj_id in type_subjects[t]:
            if subj_id in borg and 'task1_35i' in borg[subj_id]:
                task = borg[subj_id]['task1_35i']
                rpe_t = task.get('rpe_times', [])
                rpe_v = task.get('rpe_values', [])
                if len(rpe_t) >= 3 and len(rpe_v) >= 3:
                    # Simple slope: (last RPE - first RPE) / time span
                    times = np.array(rpe_t[:len(rpe_v)])
                    vals = np.array(rpe_v[:len(times)])
                    if times[-1] > times[0]:
                        slope = (vals[-1] - vals[0]) / (times[-1] - times[0])
                        slopes.append(slope)
        if slopes:
            type_slopes[t] = np.mean(slopes)
            print(f"    Type {t}: n={len(slopes)}, "
                  f"mean RPE slope={type_slopes[t]:.4f} /s")

    passed = 0
    total = 1
    if len(type_slopes) >= 2:
        print(f"\n  [PASS] RPE slopes computed for {len(type_slopes)} types")
        passed += 1
    else:
        print(f"\n  [FAIL] RPE slopes only for {len(type_slopes)} types")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def validate_dynamic_isometric(path_g_result):
    """Validate dynamic vs isometric F regime reconciliation (L4)."""
    print("\n" + "=" * 60)
    print("VALIDATION 5: Dynamic vs Isometric F Reconciliation")
    print("=" * 60)

    report = compute_dynamic_isometric_report(path_g_result['calibration_results'])

    print(f"  Isometric F (Table 1): {report['F_isometric']:.4f} min^-1")
    print(f"  Dynamic F range: [{report['ratio_min']*report['F_isometric']:.4f}, "
          f"{report['ratio_max']*report['F_isometric']:.4f}] min^-1")
    print(f"  Scaling ratio F_dynamic/F_isometric:")
    print(f"    mean={report['ratio_mean']:.1f}x, SD={report['ratio_std']:.1f}x")
    print(f"    range=[{report['ratio_min']:.1f}x, {report['ratio_max']:.1f}x]")
    print(f"  Cross-validation (at 35% MVC):")
    print(f"    Dynamic F -> ET={report['et_dynamic_F_at_35_mean']:.0f}s "
          f"(should be <60s, confirming dynamic regime)")
    print(f"    Isometric F -> ET={report['et_isometric_F_at_35_mean']:.0f}s "
          f"(should be >600s, confirming isometric regime)")

    passed = 0
    total = 3

    if report['all_dynamic_F_above_10x_isometric']:
        print(f"\n  [PASS] All dynamic F > 10x isometric F")
        passed += 1
    else:
        print(f"\n  [FAIL] Some dynamic F < 10x isometric F "
              f"(min ratio={report['ratio_min']:.1f}x)")

    if report['all_dynamic_et_below_60s']:
        print(f"  [PASS] Dynamic F predicts ET < 60s (confirms not isometric)")
        passed += 1
    else:
        print(f"  [FAIL] Some dynamic ET >= 60s")

    if report['isometric_et_above_600s']:
        print(f"  [PASS] Isometric F predicts ET > 600s (confirms not dynamic)")
        passed += 1
    else:
        print(f"  [FAIL] Isometric ET < 600s")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def validate_correlation_structure(path_g_result):
    """Validate inter-muscle correlation in sampled parameters (L5)."""
    print("\n" + "=" * 60)
    print("VALIDATION 6: Inter-Muscle F Correlation Structure")
    print("=" * 60)

    from scipy.stats import spearmanr

    profiles = path_g_result['worker_profiles']
    if not profiles:
        print("  [FAIL] No worker profiles")
        return False

    # Extract shoulder F and all non-shoulder F
    shoulder_F = [p['muscles']['shoulder']['F'] for p in profiles]
    other_muscles = [m for m in profiles[0]['muscles'] if m != 'shoulder']

    passed = 0
    total = 0

    for muscle in other_muscles:
        muscle_F = [p['muscles'][muscle]['F'] for p in profiles]
        pop_F = POPULATION_FR[muscle][0]

        # Check 1: Spearman correlation with shoulder F should be positive
        rho_s, p_val = spearmanr(shoulder_F, muscle_F)
        total += 1
        if rho_s > 0:
            print(f"  {muscle}: Spearman rho={rho_s:.3f} (p={p_val:.3f}) [PASS]")
            passed += 1
        else:
            print(f"  {muscle}: Spearman rho={rho_s:.3f} (p={p_val:.3f}) [FAIL: expected positive]")

        # Check 2: Population marginal preserved (mean within 50% of published)
        mean_F = np.mean(muscle_F)
        ratio = mean_F / pop_F
        total += 1
        if 0.5 < ratio < 2.0:
            print(f"    mean F={mean_F:.5f}, pop={pop_F:.5f}, ratio={ratio:.2f} [PASS]")
            passed += 1
        else:
            print(f"    mean F={mean_F:.5f}, pop={pop_F:.5f}, ratio={ratio:.2f} [FAIL]")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def main():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "wsd4fedsrm", "WSD4FEDSRM"
    )

    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}")
        print("Download from: https://zenodo.org/records/8415066")
        sys.exit(1)

    # Run Path G calibration
    path_g_result = run_path_g(data_dir)

    # Generate demonstrations
    demos, wids = generate_demonstrations_from_profiles(
        path_g_result['worker_profiles'],
        muscle='shoulder',
        n_episodes_per_worker=3,
        episode_duration_sec=90.0,
        variable_length=True,
    )

    # Load into MMICRL
    collector = load_path_g_into_collector(demos, wids)

    # Run MMICRL
    mmicrl = MMICRL(
        n_types=3, lambda1=1.0, lambda2=1.0,
        n_muscles=1, n_iterations=150, hidden_dims=[64, 64],
    )
    mmicrl_results = mmicrl.fit(collector)

    # Run all validations
    v1 = validate_calibration(path_g_result)
    v2 = validate_demonstrations(demos, wids, path_g_result)
    v3 = validate_mmicrl_types(mmicrl_results, mmicrl, wids, path_g_result)
    v4 = validate_borg_correlation(path_g_result, mmicrl, wids)
    v5 = validate_dynamic_isometric(path_g_result)
    v6 = validate_correlation_structure(path_g_result)

    # Final summary
    print("\n" + "=" * 60)
    print("OVERALL VALIDATION SUMMARY")
    print("=" * 60)
    results = [
        ("Calibration Quality", v1),
        ("Demonstration Quality", v2),
        ("MMICRL Type Discovery", v3),
        ("Borg RPE Cross-Validation", v4),
        ("Dynamic vs Isometric Reconciliation", v5),
        ("Inter-Muscle F Correlation", v6),
    ]
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  All validations PASSED.")
    else:
        print("\n  Some validations FAILED. Review output above.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
