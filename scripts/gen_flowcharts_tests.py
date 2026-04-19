"""
Generate code flowcharts for tests/*.py (12 files).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from flowchart_framework import FlowchartBuilder, HDR, CLS, FUNC, CONST, PROP, UTIL, EXT_TEST

OUT = "diagrams/code_flowcharts"
os.makedirs(OUT, exist_ok=True)


# =====================================================================
# tests/test_three_cc_r.py
# =====================================================================
def gen_test_three_cc_r():
    fb = FlowchartBuilder("tests/test_three_cc_r.py",
        "51 tests: MuscleParams, ThreeCCrState, ODE, conservation, steady state",
        lines="462", imports_desc="pytest, numpy, hcmarl.three_cc_r.*")

    fb.make_node("cls_params", "TestMuscleParams (10 tests)", CLS, [
        ("test_shoulder_raw_values", "F=0.0146, R=0.00058, r=15 from Table 1"),
        ("test_ankle/knee/elbow", "verify raw F, R, r per muscle"),
        ("test_C_max, delta_max", "derived quantities: F*R/(F+R), R/(F+R)"),
        ("test_Rr, theta_min_max", "R*r, F/(F+R*r)"),
        ("test_all_muscles_len", "len(ALL_MUSCLES) == 6"),
    ])

    fb.make_node("cls_state", "TestThreeCCrState (6 tests)", CLS, [
        ("test_fresh", "MR=1, MA=MF=0"),
        ("test_conservation_check", "|MR+MA+MF - 1| < tol"),
        ("test_as_array_from_array", "roundtrip ndarray <-> dataclass"),
        ("test_invalid_sum_raises", "MR+MA+MF != 1 -> ValueError"),
    ])

    fb.make_node("cls_ode", "TestThreeCCrODE (15 tests)", CLS, [
        ("test_R_eff_working", "TL>0 -> R"),
        ("test_R_eff_resting", "TL=0 -> R*r (reperfusion)"),
        ("test_baseline_neural_drive", "kp*max(TL-MA,0)"),
        ("test_ode_rhs_at_rest", "C=0, TL=0 -> recovery-only ODE"),
        ("test_euler_step_conservation", "MR+MA+MF=1 after step"),
        ("test_simulate_rk45", "solve_ivp with rtol=1e-8"),
    ])

    fb.make_node("cls_steady", "TestSteadyState (8 tests)", CLS, [
        ("test_steady_state_work", "MA=delta_max, MF=1-delta_max (Eqs 7-8)"),
        ("test_below_Cmax_no_fatigue", "TL < C_max => no exhaustion"),
        ("test_above_Cmax_exhaustion", "TL > C_max => exhaustion"),
    ])

    fb.make_node("cls_simulate", "TestSimulate (12 tests)", CLS, [
        ("test_work_phase_fatigue_grows", "MF increases under load"),
        ("test_rest_phase_recovery", "MF decreases during rest"),
        ("test_long_simulation_stability", "no NaN after 1000 min"),
    ])

    fb.edge("file_hdr", "cls_params", "", style="bold")
    fb.edge("file_hdr", "cls_state", "", style="bold")
    fb.edge("file_hdr", "cls_ode", "", style="bold")
    fb.edge("file_hdr", "cls_steady", "", style="bold")
    fb.edge("file_hdr", "cls_simulate", "", style="bold")

    fb.dangling_in("cls_params", "SHOULDER, ANKLE, KNEE, ELBOW, TRUNK, GRIP", "three_cc_r.py")
    fb.dangling_in("cls_ode", "ThreeCCr, ThreeCCrState", "three_cc_r.py")
    fb.dangling_in("cls_steady", "MuscleParams.C_max, delta_max", "three_cc_r.py")

    fb.add_legend_entry("SHOULDER, ANKLE, KNEE, ELBOW, TRUNK, GRIP", "MuscleParams instances from Table 1", "SHOULDER.F=0.0146")
    fb.add_legend_entry("ThreeCCr, ThreeCCrState", "ODE solver class + state dataclass", "ThreeCCr(SHOULDER)")
    fb.add_legend_entry("MuscleParams.C_max, delta_max", "derived properties used in steady-state tests", "C_max=0.000558, delta_max=0.038")
    fb.render(OUT)


# =====================================================================
# tests/test_ecbf.py
# =====================================================================
def gen_test_ecbf():
    fb = FlowchartBuilder("tests/test_ecbf.py",
        "33 tests: ECBFParams validation, barrier functions, QP solver",
        lines="294", imports_desc="pytest, numpy, hcmarl.ecbf_filter.*, three_cc_r.*")

    fb.make_node("cls_params_e", "TestECBFParams (5 tests)", CLS, [
        ("test_valid_params", "theta_max=0.70, alpha1=0.05 ok"),
        ("test_theta_max_below", "Assumption 5.5 violation -> ValueError"),
        ("test_negative_alpha", "alpha1 < 0 -> ValueError"),
    ])

    fb.make_node("cls_barrier", "TestBarrierFunctions (10 tests)", CLS, [
        ("test_h1_work", "h1(MF, theta_max) >= 0 when safe (Eq 19)"),
        ("test_h2_rest", "h2(MR) >= 0 when resting (Eq 20)"),
        ("test_psi1_psi2", "Eq 21-22: second-order CBFs"),
    ])

    fb.make_node("cls_qp", "TestQPSolver (12 tests)", CLS, [
        ("test_safe_state_unclipped", "C_safe = C_nominal when safe"),
        ("test_unsafe_clips_down", "C_safe < C_nominal near theta_max"),
        ("test_rest_forces_zero", "TL=0 -> C_safe=0"),
        ("test_multi_muscle", "per-muscle independent QP"),
    ])

    fb.make_node("cls_diag", "TestDiagnostics (6 tests)", CLS, [
        ("test_diagnostics_fields", "ECBFDiagnostics dataclass"),
        ("test_intervention_count", "tracks clips per muscle"),
    ])

    fb.edge("file_hdr", "cls_params_e", "", style="bold")
    fb.edge("file_hdr", "cls_barrier", "", style="bold")
    fb.edge("file_hdr", "cls_qp", "", style="bold")
    fb.edge("file_hdr", "cls_diag", "", style="bold")

    fb.dangling_in("cls_params_e", "ECBFParams, ECBFFilter", "ecbf_filter.py")
    fb.dangling_in("cls_barrier", "SHOULDER, ThreeCCrState", "three_cc_r.py")

    fb.add_legend_entry("ECBFParams, ECBFFilter", "safety filter classes under test", "ECBFFilter(SHOULDER, params)")
    fb.add_legend_entry("SHOULDER, ThreeCCrState", "muscle params + state for barrier computation", "ThreeCCrState(MR=0.7,MA=0.2,MF=0.1)")
    fb.render(OUT)


# =====================================================================
# tests/test_nswf.py
# =====================================================================
def gen_test_nswf():
    fb = FlowchartBuilder("tests/test_nswf.py",
        "24 tests: NSWFParams, disagreement utility (P1-P4), allocation, Jain index",
        lines="275", imports_desc="pytest, numpy, hcmarl.nswf_allocator.*")

    fb.make_node("cls_nswf_p", "TestNSWFParams (3 tests)", CLS, [
        ("test_valid_params", "kappa=1.0, epsilon=1e-3"),
        ("test_negative_kappa", "ValueError"),
        ("test_zero_epsilon", "ValueError"),
    ])

    fb.make_node("cls_disagree", "TestDisagreementUtility (8 tests)", CLS, [
        ("P1", "D_i(0) = 0"),
        ("P2", "monotonically increasing"),
        ("P3", "convex on (0,1)"),
        ("P4", "lim MF->1 = infinity"),
    ])

    fb.make_node("cls_alloc", "TestAllocation (8 tests)", CLS, [
        ("test_all_rest_when_exhausted", "MF > theta_max -> rest"),
        ("test_fair_allocation", "balanced across workers"),
        ("test_NSWF_objective", "product of surpluses (Eq 31)"),
    ])

    fb.make_node("cls_jain", "TestJainIndex (5 tests)", CLS, [
        ("test_equal_distribution", "Jain = 1.0"),
        ("test_one_gets_all", "Jain = 1/N"),
    ])

    fb.edge("file_hdr", "cls_nswf_p", "", style="bold")
    fb.edge("file_hdr", "cls_disagree", "", style="bold")
    fb.edge("file_hdr", "cls_alloc", "", style="bold")
    fb.edge("file_hdr", "cls_jain", "", style="bold")

    fb.dangling_in("cls_nswf_p", "NSWFAllocator, NSWFParams, AllocationResult", "nswf_allocator.py")

    fb.add_legend_entry("NSWFAllocator, NSWFParams, AllocationResult", "allocator classes under test", "NSWFAllocator(NSWFParams(kappa=1.0))")
    fb.render(OUT)


# =====================================================================
# tests/test_pipeline.py
# =====================================================================
def gen_test_pipeline():
    fb = FlowchartBuilder("tests/test_pipeline.py",
        "17 tests: TaskProfile, WorkerState, full pipeline step (Sec 7.3)",
        lines="269", imports_desc="pytest, numpy, hcmarl.pipeline.*, three_cc_r.*")

    fb.make_node("cls_tp", "TestTaskProfile (3 tests)", CLS, [
        ("test_valid_profile", "demands dict -> get_load('shoulder')"),
        ("test_missing_muscle", "returns 0.0"),
        ("test_invalid_load", "ValueError for TL > 1"),
    ])

    fb.make_node("cls_ws", "TestWorkerState (4 tests)", CLS, [
        ("test_fresh_worker", "MR=1 for all muscles"),
        ("test_max_fatigue", "tracks max MF across muscles"),
    ])

    fb.make_node("cls_pipe", "TestPipelineStep (10 tests)", CLS, [
        ("test_pipeline_init", "7-step pipeline construction"),
        ("test_step_physics", "ODE + ECBF + NSWF in sequence"),
        ("test_ecbf_intervention_count", "tracks per-muscle clips"),
        ("test_end_to_end_episode", "480-step rollout"),
    ])

    fb.edge("file_hdr", "cls_tp", "", style="bold")
    fb.edge("file_hdr", "cls_ws", "", style="bold")
    fb.edge("file_hdr", "cls_pipe", "", style="bold")
    fb.edge("cls_tp", "cls_pipe", "TaskProfile")
    fb.edge("cls_ws", "cls_pipe", "WorkerState")

    fb.dangling_in("cls_tp", "TaskProfile", "pipeline.py")
    fb.dangling_in("cls_ws", "WorkerState", "pipeline.py")
    fb.dangling_in("cls_pipe", "HCMARLPipeline, ECBFParams, NSWFParams", "pipeline.py, ecbf_filter.py, nswf_allocator.py")
    fb.dangling_in("cls_pipe", "SHOULDER, ELBOW, GRIP", "three_cc_r.py")

    fb.add_legend_entry("TaskProfile", "task demand profile object passed to pipeline tests", "TaskProfile(id=1, demands={'shoulder':0.4})")
    fb.add_legend_entry("WorkerState", "worker state object passed to pipeline tests", "WorkerState.fresh(0, ['shoulder'])")
    fb.add_legend_entry("HCMARLPipeline, ECBFParams, NSWFParams", "pipeline + config objects under test", "HCMARLPipeline(num_workers=4)")
    fb.add_legend_entry("SHOULDER, ELBOW, GRIP", "MuscleParams used to build pipeline", "SHOULDER.F=0.0146")
    fb.render(OUT)


# =====================================================================
# tests/test_phase3.py
# =====================================================================
def gen_test_phase3():
    fb = FlowchartBuilder("tests/test_phase3.py",
        "15 tests: MMICRL, CFDE, DemonstrationCollector, OnlineAdapter",
        lines="271", imports_desc="numpy, torch, hcmarl.mmicrl.*")

    fb.make_node("cls_collector", "DemonstrationCollector tests (3)", CLS, [
        ("test_synthetic_demos", "generate 60 demos, len=60"),
        ("test_features_shape", "(20, 5) trajectory features"),
    ])

    fb.make_node("cls_mmicrl", "MMICRL.fit tests (5)", CLS, [
        ("test_fit", "mutual_information >= 0, 3 types"),
        ("test_lambda_equality", "lambda1=lambda2 -> obj = lambda*I"),
        ("test_type_proportions", "sum = 1.0"),
        ("test_theta_per_type", "3 types, each with muscle thresholds"),
    ])

    fb.make_node("cls_cfde", "CFDE tests (4)", CLS, [
        ("test_cfde_forward", "forward pass returns (z, log_det)"),
        ("test_cfde_sample", "n_samples random draws"),
        ("test_cfde_log_prob", "log p(x) computation"),
    ])

    fb.make_node("cls_online", "OnlineAdapter tests (3)", CLS, [
        ("test_online_update", "Bayesian posterior update"),
        ("test_validate_mmicrl", "end-to-end validation fn"),
    ])

    fb.edge("file_hdr", "cls_collector", "", style="bold")
    fb.edge("file_hdr", "cls_mmicrl", "", style="bold")
    fb.edge("file_hdr", "cls_cfde", "", style="bold")
    fb.edge("file_hdr", "cls_online", "", style="bold")
    fb.edge("cls_collector", "cls_mmicrl", "demonstrations")
    fb.edge("cls_cfde", "cls_mmicrl", "CFDE")

    fb.dangling_in("cls_mmicrl", "MMICRL, DemonstrationCollector", "mmicrl.py")
    fb.dangling_in("cls_cfde", "CFDE, _MADE", "mmicrl.py")
    fb.dangling_in("cls_online", "OnlineAdapter, validate_mmicrl", "mmicrl.py")

    fb.add_legend_entry("demonstrations", "list of (state, action) trajectories from collector", "[(s0,a0), (s1,a1), ...] x 60")
    fb.add_legend_entry("CFDE", "Conditional Flow Density Estimator model object", "CFDE(n_features=5, n_types=3)")
    fb.add_legend_entry("MMICRL, DemonstrationCollector", "MMICRL class + demo collector under test", "MMICRL(n_types=3)")
    fb.add_legend_entry("CFDE, _MADE", "flow model + autoregressive network under test", "CFDE(n_features=5)")
    fb.add_legend_entry("OnlineAdapter, validate_mmicrl", "online Bayesian update + validation function", "OnlineAdapter(mmicrl)")
    fb.render(OUT)


# =====================================================================
# tests/test_real_data_calibration.py
# =====================================================================
def gen_test_real_data_calib():
    fb = FlowchartBuilder("tests/test_real_data_calibration.py",
        "53 tests: Path G calibration, endurance prediction, population sampling",
        lines="541", imports_desc="pytest, numpy, hcmarl.real_data_calibration.*")

    fb.make_node("cls_endurance", "TestEndurance (8 tests)", CLS, [
        ("predict_endurance_time", "ODE-based exhaustion time"),
        ("test_low_load_long", "TL=0.10 -> high endurance"),
        ("test_high_load_short", "TL=0.80 -> quick exhaustion"),
        ("test_rohmert_agreement", "matches power-model reference"),
    ])

    fb.make_node("cls_calib", "TestCalibration (10 tests)", CLS, [
        ("calibrate_F_for_subject", "1D grid search for optimal F"),
        ("test_shoulder_F_range", "F in [0.005, 0.05]"),
        ("test_endurance_population", "multi-subject aggregate"),
    ])

    fb.make_node("cls_sample", "TestPopulationSampling (12 tests)", CLS, [
        ("sample_FR_from_population", "draw (F, R) pairs"),
        ("sample_correlated_FR", "inter-muscle correlated draws"),
        ("test_population_CV", "CV_F matches Table 1"),
        ("test_correlation_structure", "positive inter-muscle rho"),
    ])

    fb.make_node("cls_demo", "TestDemoGeneration (10 tests)", CLS, [
        ("generate_demonstrations", "trajectory from sampled params"),
        ("load_path_g_into_collector", "MMICRL format conversion"),
    ])

    fb.make_node("cls_wsd", "TestWSD4FEDSRM (13 tests, skip if no data)", CLS, [
        ("Needs", "data/wsd4fedsrm/WSD4FEDSRM/ (~1.6 GB)"),
        ("test_load_borg_data", "borg_data.csv parse"),
        ("test_full_calibration_pipeline", "all 34 subjects"),
    ])

    fb.edge("file_hdr", "cls_endurance", "", style="bold")
    fb.edge("file_hdr", "cls_calib", "", style="bold")
    fb.edge("file_hdr", "cls_sample", "", style="bold")
    fb.edge("file_hdr", "cls_demo", "", style="bold")
    fb.edge("file_hdr", "cls_wsd", "", style="bold")
    fb.edge("cls_endurance", "cls_calib", "predict_endurance_time")
    fb.edge("cls_calib", "cls_sample", "F_opt")
    fb.edge("cls_sample", "cls_demo", "(F_i, R_i) pairs")
    fb.edge("cls_demo", "cls_wsd", "demo_trajectories")

    fb.dangling_in("cls_endurance", "predict_endurance_time, SHOULDER", "real_data_calibration.py, three_cc_r.py")
    fb.dangling_in("cls_sample", "POPULATION_FR, POPULATION_CV_F", "real_data_calibration.py")
    fb.dangling_in("cls_wsd", "WSD4FEDSRM dataset", "data/wsd4fedsrm/")
    fb.dangling_out("cls_demo", "DemonstrationCollector", "mmicrl.py")

    fb.add_legend_entry("predict_endurance_time", "function computing ODE-based exhaustion time", "predict_endurance_time(F=0.0146, TL=0.35) -> 105s")
    fb.add_legend_entry("F_opt", "calibrated fatigue rate from grid search", "1.24 min^-1")
    fb.add_legend_entry("(F_i, R_i) pairs", "sampled fatigue/recovery rate pairs per subject", "[(0.013, 0.0006), ...]")
    fb.add_legend_entry("demo_trajectories", "generated demo trajectories for MMICRL input", "list of (state, action) sequences")
    fb.add_legend_entry("predict_endurance_time, SHOULDER", "endurance function + shoulder MuscleParams", "predict_endurance_time(SHOULDER.F, ...)")
    fb.add_legend_entry("POPULATION_FR, POPULATION_CV_F", "population-level F,R means and coefficients of variation", "F_pop=0.0146, CV_F=0.30")
    fb.add_legend_entry("WSD4FEDSRM dataset", "1.6 GB shoulder fatigue dataset (34 subjects)", "data/wsd4fedsrm/WSD4FEDSRM/")
    fb.add_legend_entry("DemonstrationCollector", "collector object with loaded demos for MMICRL", "DemonstrationCollector(n_muscles=3)")
    fb.render(OUT)


# =====================================================================
# tests/test_phase2.py
# =====================================================================
def gen_test_phase2():
    fb = FlowchartBuilder("tests/test_phase2.py",
        "12 tests: SingleWorkerWarehouseEnv, MultiAgentEnv, baselines",
        lines="186", imports_desc="numpy, hcmarl.warehouse_env, baselines")

    fb.make_node("cls_single", "SingleWorker tests (4)", CLS, [
        ("test_reset", "obs.shape == (10,), MR=1"),
        ("test_step", "heavy_lift action -> reward, info"),
        ("test_conservation", "MR+MA+MF=1 after 30 steps"),
        ("test_rest_recovers", "MF decreases during rest"),
    ])

    fb.make_node("cls_multi", "MultiAgentEnv tests (4)", CLS, [
        ("test_multi_reset", "4 workers, obs shape"),
        ("test_multi_step", "actions dict -> rewards"),
        ("test_multi_conservation", "per-worker MR+MA+MF=1"),
    ])

    fb.make_node("cls_baseline", "Baselines tests (4)", CLS, [
        ("test_create_all", "10 baselines instantiated"),
        ("test_get_actions", "each returns valid actions"),
        ("test_round_robin_cycles", "deterministic task cycling"),
    ])

    fb.edge("file_hdr", "cls_single", "", style="bold")
    fb.edge("file_hdr", "cls_multi", "", style="bold")
    fb.edge("file_hdr", "cls_baseline", "", style="bold")

    fb.dangling_in("cls_single", "SingleWorkerWarehouseEnv", "warehouse_env.py")
    fb.dangling_in("cls_multi", "WarehouseMultiAgentEnv", "warehouse_env.py")
    fb.dangling_in("cls_baseline", "create_all_baselines", "baselines/__init__.py")

    fb.add_legend_entry("SingleWorkerWarehouseEnv", "single-worker Gymnasium env under test", "SingleWorkerWarehouseEnv()")
    fb.add_legend_entry("WarehouseMultiAgentEnv", "multi-agent env under test (4 workers)", "WarehouseMultiAgentEnv(n_workers=4)")
    fb.add_legend_entry("create_all_baselines", "factory returning all 10 baseline instances", "create_all_baselines(obs_dim=10, n_actions=4)")
    fb.render(OUT)


# =====================================================================
# tests/test_all_methods.py
# =====================================================================
def gen_test_all_methods():
    fb = FlowchartBuilder("tests/test_all_methods.py",
        "6 tests: instantiate & get_actions for all 6 agent types",
        lines="52", imports_desc="numpy, torch, all agent/baseline modules")

    fb.make_node("method_tests", "6 instantiation tests", CLS, [
        ("test_random_policy", "WarehousePettingZoo + random actions"),
        ("test_mappo_instantiation", "MAPPO(obs=19, gs=73, act=6, n=4)"),
        ("test_ippo_instantiation", "IPPO(obs=19, act=6, n=4)"),
        ("test_mappo_lag_instantiation", "MAPPOLagrangian, lam > 0"),
        ("test_omnisafe_wrapper", "OmniSafeWrapper('PPOLag')"),
        ("test_safepo_wrapper", "SafePOWrapper(obs=19, act=6, n=4)"),
    ])

    fb.edge("file_hdr", "method_tests", "", style="bold")

    fb.dangling_in("method_tests", "MAPPO, IPPO, MAPPOLagrangian", "agents/*.py")
    fb.dangling_in("method_tests", "OmniSafeWrapper, SafePOWrapper", "baselines/*.py")
    fb.dangling_in("method_tests", "WarehousePettingZoo", "envs/pettingzoo_wrapper.py")

    fb.add_legend_entry("MAPPO, IPPO, MAPPOLagrangian", "RL agent classes under test", "MAPPO(obs_dim=19, n_actions=6)")
    fb.add_legend_entry("OmniSafeWrapper, SafePOWrapper", "baseline wrapper classes under test", "OmniSafeWrapper('PPOLag', obs_dim=19)")
    fb.add_legend_entry("WarehousePettingZoo", "PettingZoo parallel env for random policy test", "WarehousePettingZoo(n_workers=2)")
    fb.render(OUT)


# =====================================================================
# tests/test_pettingzoo.py
# =====================================================================
def gen_test_pettingzoo():
    fb = FlowchartBuilder("tests/test_pettingzoo.py",
        "4 tests: PettingZoo reset, step, global_obs, episode completion",
        lines="36", imports_desc="numpy, hcmarl.envs.pettingzoo_wrapper")

    fb.make_node("pz_tests", "4 PettingZoo tests", CLS, [
        ("test_reset", "3 workers -> obs len=3"),
        ("test_parallel_step", "4 workers -> obs,r len=4"),
        ("test_global_obs", "global_obs.shape = (4*18+1,)"),
        ("test_episode_completes", "5 steps -> all terminated"),
    ])

    fb.edge("file_hdr", "pz_tests", "", style="bold")
    fb.dangling_in("pz_tests", "WarehousePettingZoo", "envs/pettingzoo_wrapper.py")

    fb.add_legend_entry("WarehousePettingZoo", "PettingZoo ParallelEnv wrapper under test", "WarehousePettingZoo(n_workers=3)")
    fb.render(OUT)


# =====================================================================
# tests/test_warehouse_env.py
# =====================================================================
def gen_test_warehouse_env():
    fb = FlowchartBuilder("tests/test_warehouse_env.py",
        "4 tests: env reset, step, conservation, cost signal",
        lines="33", imports_desc="numpy, hcmarl.warehouse_env")

    fb.make_node("we_tests", "4 WarehouseEnv tests", CLS, [
        ("test_env_reset", "obs.shape[0] > 0"),
        ("test_env_step", "action=0 -> float reward"),
        ("test_conservation", "MR+MA+MF=1 after 30 steps"),
        ("test_cost_signal", "100 heavy lifts -> cost info"),
    ])

    fb.edge("file_hdr", "we_tests", "", style="bold")
    fb.dangling_in("we_tests", "SingleWorkerWarehouseEnv", "warehouse_env.py")

    fb.add_legend_entry("SingleWorkerWarehouseEnv", "single-worker Gymnasium env under test", "SingleWorkerWarehouseEnv()")
    fb.render(OUT)


# =====================================================================
# tests/test_hcmarl_agent.py
# =====================================================================
def gen_test_hcmarl_agent():
    fb = FlowchartBuilder("tests/test_hcmarl_agent.py",
        "3 tests: agent init, get_actions, save/load",
        lines="35", imports_desc="numpy, torch, hcmarl.agents.hcmarl_agent")

    fb.make_node("agent_tests", "3 HCMARLAgent tests", CLS, [
        ("test_agent_init", "n_agents=4"),
        ("test_agent_get_actions", "4 workers -> 4 actions in [0,6)"),
        ("test_agent_save_load", "save + load roundtrip"),
    ])

    fb.edge("file_hdr", "agent_tests", "", style="bold")
    fb.dangling_in("agent_tests", "HCMARLAgent", "agents/hcmarl_agent.py")

    fb.add_legend_entry("HCMARLAgent", "HC-MARL agent wrapping MAPPO+ECBF+NSWF", "HCMARLAgent(obs_dim=19, n_agents=4)")
    fb.render(OUT)


# =====================================================================
# tests/test_env_integration.py
# =====================================================================
def gen_test_env_integration():
    fb = FlowchartBuilder("tests/test_env_integration.py",
        "1 test: full episode with PettingZoo + HCMARLLogger",
        lines="27", imports_desc="numpy, pettingzoo_wrapper, logger")

    fb.make_node("integ_test", "test_full_episode_with_logger", CLS, [
        ("Setup", "WarehousePettingZoo(n=3, steps=20)"),
        ("Loop", "20 random steps, accumulate reward"),
        ("Logger", "HCMARLLogger.compute_episode_metrics()"),
        ("Assert", "len(metrics) == 9"),
    ])

    fb.edge("file_hdr", "integ_test", "", style="bold")
    fb.dangling_in("integ_test", "WarehousePettingZoo", "envs/pettingzoo_wrapper.py")
    fb.dangling_in("integ_test", "HCMARLLogger", "logger.py")

    fb.add_legend_entry("WarehousePettingZoo", "PettingZoo env used in integration test", "WarehousePettingZoo(n_workers=3, max_steps=20)")
    fb.add_legend_entry("HCMARLLogger", "logger computing 9 HC-MARL metrics", "HCMARLLogger(log_dir='/tmp/test')")
    fb.render(OUT)


# =====================================================================
# tests/__init__.py
# =====================================================================
def gen_tests_init():
    fb = FlowchartBuilder("tests/__init__.py",
        "Empty init for pytest discovery", lines="1")
    fb.make_node("empty", "# (empty file)", CONST, [
        ("Purpose", "allows pytest to discover test modules"),
    ])
    fb.edge("file_hdr", "empty", "", style="bold")
    fb.render(OUT, stem_override="tests_init")


if __name__ == "__main__":
    generators = [
        gen_test_three_cc_r, gen_test_ecbf, gen_test_nswf, gen_test_pipeline,
        gen_test_phase3, gen_test_real_data_calib, gen_test_phase2,
        gen_test_all_methods, gen_test_pettingzoo, gen_test_warehouse_env,
        gen_test_hcmarl_agent, gen_test_env_integration, gen_tests_init,
    ]
    for fn in generators:
        try:
            fn()
        except Exception as e:
            print(f"FAIL: {fn.__name__}: {e}")
