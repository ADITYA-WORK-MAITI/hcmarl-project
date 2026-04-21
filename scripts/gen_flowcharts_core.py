"""
Generate code flowcharts for ALL hcmarl/ core modules.
Produces one PNG per file in diagrams/code_flowcharts/.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from flowchart_framework import FlowchartBuilder, HDR, CLS, FUNC, CONST, PROP, UTIL

OUT = "diagrams/code_flowcharts"
os.makedirs(OUT, exist_ok=True)


# =====================================================================
# 1. hcmarl/three_cc_r.py
# =====================================================================
def gen_three_cc_r():
    fb = FlowchartBuilder(
        "hcmarl/three_cc_r.py",
        "3CC-r fatigue ODE model (Liu/Xia/Looft)",
        equations="Eqs 1-8, 25, 35 (Sections 3, 7.2)",
        lines="387",
        imports_desc="numpy, scipy.integrate.solve_ivp, dataclasses",
    )
    # --- Nodes ---
    fb.make_node("MuscleParams", "@dataclass MuscleParams (frozen)", CLS, [
        ("name: str", "'shoulder'"),
        ("F: float", "fatigue rate [min^-1]"),
        ("R: float", "recovery rate [min^-1]"),
        ("r: float", "reperfusion multiplier"),
    ])
    fb.make_node("MuscleParams_props", "MuscleParams @properties", PROP, [
        ("C_max", "F*R/(F+R) [Eq 6]"),
        ("delta_max", "R/(F+R) [Eq 6]"),
        ("Rr", "R*r reperfusion rate"),
        ("theta_min_max", "F/(F+R*r) [Eq 25]"),
        ("Rr_over_F", "Rr/F overshoot ratio"),
    ])
    fb.make_node("constants", "Module Constants (Table 1)", CONST, [
        ("SHOULDER", "F=0.01820 R=0.00168 r=15"),
        ("ANKLE", "F=0.00589 R=0.00058 r=15"),
        ("KNEE", "F=0.01500 R=0.00149 r=15"),
        ("ELBOW", "F=0.00912 R=0.00094 r=15"),
        ("TRUNK", "F=0.00755 R=0.00075 r=15"),
        ("GRIP", "F=0.00980 R=0.00064 r=30"),
        ("ALL_MUSCLES", "list[MuscleParams] len=6"),
        ("MUSCLE_REGISTRY", "dict[str, MuscleParams]"),
    ])
    fb.make_node("get_muscle", "get_muscle(name: str)", FUNC, [
        ("Input", "name e.g. 'shoulder'"),
        ("Lookup", "MUSCLE_REGISTRY[name.lower()]"),
        ("Return", "MuscleParams"),
    ])
    fb.make_node("ThreeCCrState", "@dataclass ThreeCCrState", CLS, [
        ("MR: float", "resting fraction [0,1]"),
        ("MA: float", "active fraction [0,1]"),
        ("MF: float", "fatigued fraction [0,1]"),
        ("__post_init__", "validates MR+MA+MF=1 (Eq 1)"),
    ])
    fb.make_node("State_methods", "ThreeCCrState methods", FUNC, [
        ("as_array()", "-> np.ndarray [MR,MA,MF]"),
        ("from_array(arr)", "@classmethod -> ThreeCCrState"),
        ("fresh()", "@classmethod -> MR=1,MA=0,MF=0"),
    ])
    fb.make_node("ThreeCCr_init", "class ThreeCCr.__init__", CLS, [
        ("params: MuscleParams", "calibrated muscle group"),
        ("kp: float", "10.0 (proportional gain, Eq 35)"),
    ])
    fb.make_node("R_eff", "ThreeCCr.R_eff(target_load)", FUNC, [
        ("Input", "target_load: float [0,1]"),
        ("Logic", "TL>0 -> R | TL=0 -> R*r"),
        ("Return", "float Reff [Eq 5]"),
    ])
    fb.make_node("baseline_neural_drive", "ThreeCCr.baseline_neural_drive(TL,MA)", FUNC, [
        ("Input", "target_load, MA"),
        ("Logic", "kp*max(TL-MA,0) [Eq 35]"),
        ("Return", "float C(t)"),
    ])
    fb.make_node("ode_rhs", "ThreeCCr.ode_rhs(state,C,TL)", FUNC, [
        ("Input", "state=[MR,MA,MF], C, TL"),
        ("dMA/dt", "C - F*MA [Eq 2]"),
        ("dMF/dt", "F*MA - Reff*MF [Eq 3]"),
        ("dMR/dt", "Reff*MF - C [Eq 4]"),
        ("Return", "np.ndarray [dMR,dMA,dMF]"),
    ])
    fb.make_node("step_euler", "ThreeCCr.step_euler(state,C,TL,dt)", FUNC, [
        ("Input", "ThreeCCrState, C, TL, dt=1.0"),
        ("Method", "x_new = x + dt * ode_rhs"),
        ("Clamp", "clip [0,1] renormalize sum=1"),
        ("Return", "ThreeCCrState (new)"),
    ])
    fb.make_node("simulate", "ThreeCCr.simulate(state0,TL,dur,dt_eval)", FUNC, [
        ("Input", "state0, TL, duration, dt_eval=0.1"),
        ("Method", "scipy solve_ivp RK45"),
        ("C_override", "optional fixed C"),
        ("Return", "dict: t,MR,MA,MF,C arrays"),
    ])
    fb.make_node("verify_conservation", "ThreeCCr.verify_conservation(state)", FUNC, [
        ("Check", "|MR+MA+MF - 1| < tol"),
        ("Return", "bool"),
    ])
    fb.make_node("steady_state", "ThreeCCr.steady_state_work()", FUNC, [
        ("Logic", "MR=0, MA=delta_max, MF=1-delta_max"),
        ("Equation", "Theorem 3.4 (Eqs 7-8)"),
        ("Return", "ThreeCCrState"),
    ])

    # --- Internal edges ---
    fb.edge("file_hdr", "MuscleParams", "", style="bold")
    fb.edge("file_hdr", "constants", "", style="bold")
    fb.edge("file_hdr", "ThreeCCrState", "", style="bold")
    fb.edge("file_hdr", "ThreeCCr_init", "", style="bold")
    fb.edge("MuscleParams", "MuscleParams_props", "F, R, r")
    fb.edge("MuscleParams", "constants", "F, R, r")
    fb.edge("constants", "get_muscle", "MUSCLE_REGISTRY")
    fb.edge("ThreeCCrState", "State_methods", "MR, MA, MF")
    fb.edge("ThreeCCr_init", "R_eff", "self.params", style="dashed")
    fb.edge("ThreeCCr_init", "baseline_neural_drive", "self.kp", style="dashed")
    fb.edge("R_eff", "ode_rhs", "Reff")
    fb.edge("baseline_neural_drive", "ode_rhs", "C(t)", color="#DC2626")
    fb.edge("ode_rhs", "step_euler", "dx/dt")
    fb.edge("ode_rhs", "simulate", "ode_rhs_fn")
    fb.edge("ThreeCCrState", "step_euler", "state.as_array()")
    fb.edge("ThreeCCrState", "simulate", "state0.as_array()")
    fb.edge("step_euler", "ThreeCCrState", "from_array(x_new)", style="dotted", constraint="false")
    fb.edge("MuscleParams_props", "steady_state", "delta_max")

    # --- Dangling IN (imports from stdlib) ---
    fb.dangling_in("MuscleParams", "dataclass decorator", "dataclasses")
    fb.dangling_in("simulate", "solve_ivp (RK45)", "scipy.integrate")
    fb.dangling_in("ode_rhs", "np.array", "numpy")

    # --- Dangling OUT (exported to other files) ---
    fb.dangling_out("MuscleParams", "MuscleParams, ThreeCCrState", "ecbf_filter.py")
    fb.dangling_out("MuscleParams", "MuscleParams, ThreeCCr, SHOULDER", "pipeline.py")
    fb.dangling_out("get_muscle", "get_muscle()", "pipeline.py")
    fb.dangling_out("MuscleParams", "MuscleParams, ThreeCCr, SHOULDER", "real_data_calibration.py")
    fb.dangling_out("get_muscle", "get_muscle()", "pettingzoo_wrapper.py")
    fb.dangling_out("constants", "SHOULDER, ELBOW, GRIP", "tests/", is_test=True)

    # --- Ranks ---
    fb.set_rank("r1", ["MuscleParams", "constants"])
    fb.set_rank("r2", ["MuscleParams_props", "get_muscle", "ThreeCCrState"])
    fb.set_rank("r3", ["State_methods", "ThreeCCr_init"])
    fb.set_rank("r4", ["R_eff", "baseline_neural_drive"])
    fb.set_rank("r5", ["ode_rhs"])
    fb.set_rank("r6", ["step_euler", "simulate", "verify_conservation", "steady_state"])

    # --- Legend ---
    fb.add_legend_entry("F, R, r", "fatigue rate, recovery rate, reperfusion mult", "0.01820, 0.00168, 15")
    fb.add_legend_entry("MR, MA, MF", "motor unit fractions (resting/active/fatigued)", "[0.7, 0.2, 0.1]")
    fb.add_legend_entry("MUSCLE_REGISTRY", "dict[str, MuscleParams] all 6 muscles", "{'shoulder': MuscleParams(...)}")
    fb.add_legend_entry("self.params", "MuscleParams instance stored on ThreeCCr", "MuscleParams('shoulder', F=0.01820, ...)")
    fb.add_legend_entry("self.kp", "proportional gain scalar stored on ThreeCCr", "10.0")
    fb.add_legend_entry("Reff", "effective recovery: R or R*r", "0.0087")
    fb.add_legend_entry("C(t)", "neural drive (recruitment rate) [min^-1]", "0.05")
    fb.add_legend_entry("dx/dt", "ODE derivatives [dMR,dMA,dMF]", "[-0.01, 0.008, 0.002]")
    fb.add_legend_entry("ode_rhs_fn", "callable ODE right-hand side passed to solve_ivp", "ThreeCCr.ode_rhs")
    fb.add_legend_entry("state.as_array()", "ThreeCCrState serialised as np.ndarray", "[0.7, 0.2, 0.1]")
    fb.add_legend_entry("state0.as_array()", "initial state as np.ndarray for solve_ivp", "[1.0, 0.0, 0.0]")
    fb.add_legend_entry("from_array(x_new)", "np.ndarray deserialised back to ThreeCCrState", "ThreeCCrState(0.69,0.21,0.10)")
    fb.add_legend_entry("delta_max", "max sustainable duty cycle R/(F+R)", "0.038")
    fb.add_legend_entry("theta_min_max", "rest-phase safety threshold F/(F+Rr)", "0.627")
    fb.add_legend_entry("TL", "target load (fraction of MVC)", "0.35")

    fb.render(OUT)


# =====================================================================
# 2. hcmarl/ecbf_filter.py
# =====================================================================
def gen_ecbf_filter():
    fb = FlowchartBuilder(
        "hcmarl/ecbf_filter.py",
        "Dual-barrier ECBF safety filter (CBF-QP)",
        equations="Eqs 12-23, 28-30 (Section 5)",
        lines="429",
        imports_desc="cvxpy, numpy, hcmarl.three_cc_r",
    )
    fb.make_node("ECBFParams", "@dataclass ECBFParams", CLS, [
        ("theta_max: float", "max fatigue threshold [Eq 12]"),
        ("alpha1: float", "0.05 ECBF gain (psi_1) [Eq 16]"),
        ("alpha2: float", "0.05 ECBF enforcement [Eq 17]"),
        ("alpha3: float", "0.1 resting-floor CBF [Eq 23]"),
        ("validate(muscle)", "checks Eq 26: theta_max >= F/(F+Rr)"),
    ])
    fb.make_node("ECBFDiag", "@dataclass ECBFDiagnostics", CLS, [
        ("C_nominal, C_filtered", "input vs output drive"),
        ("h, h2", "barrier values [Eqs 12, 21]"),
        ("psi_0, psi_1, h_dot", "composite barriers [Eqs 15-16, 13]"),
        ("C_upper_ecbf, C_upper_cbf", "analytical bounds [Eqs 19, 23]"),
        ("qp_status, was_clipped", "CVXPY status, bool"),
    ])
    fb.make_node("ECBFFilter_init", "class ECBFFilter.__init__", CLS, [
        ("muscle: MuscleParams", "from three_cc_r"),
        ("ecbf_params: ECBFParams", "design parameters"),
        ("Stores", "_F, _R, _Rr, _theta_max, alphas"),
    ])
    fb.make_node("barriers", "Barrier computations", FUNC, [
        ("h(MF)", "Theta_max - MF [Eq 12]"),
        ("h2(MR)", "MR [Eq 21]"),
        ("h_dot(MA,MF,Reff)", "-F*MA + Reff*MF [Eq 13]"),
        ("h_ddot(MA,MF,C,Reff)", "-F*C + F^2*MA + ... [Eq 14]"),
        ("psi_0(MF)", "= h(x) [Eq 15]"),
        ("psi_1(MA,MF,Reff)", "h_dot + alpha1*h [Eq 16]"),
        ("h2_dot(MF,C,Reff)", "Reff*MF - C [Eq 22]"),
    ])
    fb.make_node("bounds", "Analytical upper bounds", FUNC, [
        ("ecbf_upper_bound()", "C <= ... / F [Eq 19]"),
        ("cbf_upper_bound()", "Reff*MF + alpha3*MR [Eq 23]"),
    ])
    fb.make_node("filter_qp", "ECBFFilter.filter() — CBF-QP", FUNC, [
        ("Input", "state, C_nominal, target_load"),
        ("QP var", "C_var (cvxpy Variable)"),
        ("Objective", "min ||C - C_nom||^2 [Eq 20]"),
        ("Constr 1", "ECBF (Eq 18) fatigue ceiling"),
        ("Constr 2", "CBF (Eq 23) resting floor"),
        ("Constr 3", "C >= 0"),
        ("Solver", "OSQP (fallback: analytical)"),
        ("Return", "(C_filtered, ECBFDiagnostics)"),
    ])
    fb.make_node("filter_analytical", "filter_analytical()", FUNC, [
        ("Input", "state, C_nominal, TL"),
        ("Logic", "min(C_nom, ub_ecbf, ub_cbf)"),
        ("Return", "C_safe: float"),
    ])
    fb.make_node("rest_analysis", "Rest-phase analysis (Sec 5.4)", FUNC, [
        ("rest_phase_safe(state)", "h>=0 and h2>=0 [Thm 5.7]"),
        ("psi1_jump_at_rest(MF)", "R*(r-1)*MF [Eq 28]"),
        ("min_rest_duration_bound(MA)", "(1/F)*ln(...) [Eq 30]"),
    ])

    # --- Internal edges ---
    fb.edge("file_hdr", "ECBFParams", "", style="bold")
    fb.edge("file_hdr", "ECBFDiag", "", style="bold")
    fb.edge("file_hdr", "ECBFFilter_init", "", style="bold")
    fb.edge("ECBFParams", "ECBFFilter_init", "ecbf_params")
    fb.edge("ECBFFilter_init", "barriers", "self._F, _R, alphas", style="dashed")
    fb.edge("barriers", "bounds", "h, h_dot, psi_1")
    fb.edge("barriers", "filter_qp", "psi_1, h, h2")
    fb.edge("bounds", "filter_qp", "C_upper_ecbf, C_upper_cbf")
    fb.edge("bounds", "filter_analytical", "ub_ecbf, ub_cbf")
    fb.edge("filter_qp", "ECBFDiag", "C_filtered, h, psi_1, was_clipped", style="dashed")
    fb.edge("ECBFFilter_init", "rest_analysis", "self.muscle", style="dashed")

    # --- Dangling IN ---
    fb.dangling_in("ECBFFilter_init", "MuscleParams, ThreeCCrState", "three_cc_r.py")
    fb.dangling_in("filter_qp", "cp.Variable, cp.Problem", "cvxpy")
    fb.dangling_in("filter_qp", "C_nominal (from RL/baseline)", "pipeline.py")

    # --- Dangling OUT ---
    fb.dangling_out("filter_qp", "C_filtered, ECBFDiagnostics", "pipeline.py")
    fb.dangling_out("ECBFParams", "ECBFParams", "pipeline.py")
    fb.dangling_out("filter_qp", "C_safe", "warehouse_env.py")
    fb.dangling_out("rest_analysis", "rest_phase_safe()", "tests/", is_test=True)

    # --- Ranks ---
    fb.set_rank("r1", ["ECBFParams", "ECBFDiag"])
    fb.set_rank("r2", ["ECBFFilter_init"])
    fb.set_rank("r3", ["barriers"])
    fb.set_rank("r4", ["bounds"])
    fb.set_rank("r5", ["filter_qp", "filter_analytical"])
    fb.set_rank("r6", ["rest_analysis"])

    # --- Legend ---
    fb.add_legend_entry("ecbf_params", "ECBFParams instance (alpha1,alpha2,alpha3,theta_max)", "ECBFParams(theta_max=0.7, alpha1=0.05, ...)")
    fb.add_legend_entry("self._F, _R, alphas", "scalar muscle + gain params stored on ECBFFilter", "_F=0.01820, _R=0.00168, alpha1=0.05")
    fb.add_legend_entry("h, h_dot, psi_1", "barrier value, time-derivative, composite barrier", "0.35, -0.01, 0.02")
    fb.add_legend_entry("psi_1, h, h2", "composite barrier + fatigue + resting barrier values", "0.02, 0.35, 0.65")
    fb.add_legend_entry("C_upper_ecbf, C_upper_cbf", "analytical upper bounds on C from Eqs 19, 23", "0.10, 0.14")
    fb.add_legend_entry("ub_ecbf, ub_cbf", "same upper bounds forwarded to analytical fallback", "0.10, 0.14")
    fb.add_legend_entry("C_filtered, h, psi_1, was_clipped", "QP output fields written into ECBFDiagnostics", "0.08, 0.35, 0.02, True")
    fb.add_legend_entry("self.muscle", "MuscleParams instance stored on ECBFFilter", "MuscleParams('shoulder', ...)")
    fb.add_legend_entry("C_nominal", "proposed neural drive from RL/baseline", "0.15")
    fb.add_legend_entry("h(x)", "fatigue barrier: Theta_max - MF", "0.35")
    fb.add_legend_entry("h2(x)", "resting barrier: MR", "0.65")
    fb.add_legend_entry("Reff", "effective recovery rate (R or R*r)", "0.0087")
    fb.add_legend_entry("alpha1,2,3", "CBF class-K gain parameters", "0.05, 0.05, 0.1")
    fb.add_legend_entry("theta_max", "max allowable fatigue fraction", "0.70")
    fb.add_legend_entry("was_clipped", "whether QP modified C_nom", "True/False")

    fb.render(OUT)


# =====================================================================
# 3. hcmarl/nswf_allocator.py
# =====================================================================
def gen_nswf_allocator():
    fb = FlowchartBuilder(
        "hcmarl/nswf_allocator.py",
        "Nash Social Welfare task allocator",
        equations="Eqs 31-33, Prop 6.3 (Section 6)",
        lines="361",
        imports_desc="numpy, dataclasses, hcmarl.utils.safe_log",
    )
    fb.make_node("NSWFParams", "@dataclass NSWFParams", CLS, [
        ("kappa: float", "1.0 disagreement scaling [Eq 32]"),
        ("epsilon: float", "1e-3 rest surplus [Eq 31]"),
    ])
    fb.make_node("AllocResult", "@dataclass AllocationResult", CLS, [
        ("assignments", "dict[int,int] worker->task"),
        ("objective_value", "NSWF objective [Eq 33]"),
        ("surpluses", "dict[int,float] per-worker"),
        ("disagreement_utilities", "dict[int,float] D_i"),
    ])
    fb.make_node("NSWFAlloc_init", "class NSWFAllocator.__init__", CLS, [
        ("params", "NSWFParams (kappa, epsilon)"),
    ])
    fb.make_node("disagreement", "disagreement_utility(MF)", FUNC, [
        ("Formula", "kappa * MF^2 / (1-MF) [Eq 32]"),
        ("P1", "D(0) = 0"),
        ("P2", "D'(MF) > 0 monotone"),
        ("P3", "D -> inf as MF -> 1"),
        ("Return", "float D_i"),
    ])
    fb.make_node("rest_util", "rest_utility(MF)", FUNC, [
        ("Formula", "D_i(MF) + epsilon [Eq 31]"),
        ("Return", "float"),
    ])
    fb.make_node("surplus_fn", "surplus(utility, MF)", FUNC, [
        ("Formula", "U(i,j) - D_i(MF)"),
        ("Return", "float (positive = rational)"),
    ])
    fb.make_node("allocate", "allocate(utility_matrix, fatigue_levels)", FUNC, [
        ("Input", "U: (N,M), MF: (N,)"),
        ("Route", "N,M<=8 -> exact | else -> greedy"),
        ("Return", "AllocationResult"),
    ])
    fb.make_node("solve_exact", "_solve_exact(N,M,surplus_matrix,D,eps)", FUNC, [
        ("Method", "recursive enumeration"),
        ("Objective", "max sum_i ln(surplus_i) [Eq 33]"),
        ("Constraint", "one task per worker, one worker per task"),
    ])
    fb.make_node("solve_greedy", "_solve_greedy(N,M,surplus_matrix,D,eps)", FUNC, [
        ("Method", "sort by gain descending"),
        ("Gain", "ln(surplus) - ln(epsilon)"),
        ("Unassigned", "default to rest (task 0)"),
    ])

    fb.edge("file_hdr", "NSWFParams", "", style="bold")
    fb.edge("file_hdr", "AllocResult", "", style="bold")
    fb.edge("file_hdr", "NSWFAlloc_init", "", style="bold")
    fb.edge("NSWFParams", "NSWFAlloc_init", "params")
    fb.edge("NSWFAlloc_init", "disagreement", "self.params.kappa", style="dashed")
    fb.edge("NSWFAlloc_init", "allocate", "self.params", style="dashed")
    fb.edge("disagreement", "rest_util", "D_i")
    fb.edge("disagreement", "surplus_fn", "D_i")
    fb.edge("surplus_fn", "allocate", "surplus_matrix")
    fb.edge("allocate", "solve_exact", "N,M<=8")
    fb.edge("allocate", "solve_greedy", "N,M>8")
    fb.edge("solve_exact", "AllocResult", "assignments, surpluses, D_i")
    fb.edge("solve_greedy", "AllocResult", "assignments, surpluses, D_i")

    fb.dangling_in("surplus_fn", "safe_log()", "hcmarl/utils.py")
    fb.dangling_in("allocate", "utility_matrix (N,M)", "pipeline.py")
    fb.dangling_in("allocate", "fatigue_levels (N,)", "pipeline.py")
    fb.dangling_out("AllocResult", "AllocationResult", "pipeline.py")
    fb.dangling_out("disagreement", "D_i values", "warehouse_env.py")
    fb.dangling_out("AllocResult", "assignments, surpluses", "tests/", is_test=True)

    fb.set_rank("r1", ["NSWFParams", "AllocResult"])
    fb.set_rank("r2", ["NSWFAlloc_init"])
    fb.set_rank("r3", ["disagreement", "rest_util", "surplus_fn"])
    fb.set_rank("r4", ["allocate"])
    fb.set_rank("r5", ["solve_exact", "solve_greedy"])

    fb.add_legend_entry("params", "NSWFParams instance (kappa, epsilon)", "NSWFParams(kappa=1.0, epsilon=0.001)")
    fb.add_legend_entry("self.params.kappa", "disagreement scaling scalar from stored params", "1.0")
    fb.add_legend_entry("self.params", "NSWFParams stored on NSWFAllocator", "NSWFParams(kappa=1.0, epsilon=0.001)")
    fb.add_legend_entry("D_i", "disagreement utility kappa*MF^2/(1-MF)", "0.143")
    fb.add_legend_entry("surplus_matrix", "S[i,j] = U[i,j] - D_i per worker-task pair", "np.array([[0.86, 0.56], ...])")
    fb.add_legend_entry("N,M<=8", "routing condition: use exact solver", "N=4, M=3")
    fb.add_legend_entry("N,M>8", "routing condition: use greedy solver", "N=10, M=9")
    fb.add_legend_entry("assignments, surpluses, D_i", "AllocationResult fields written by solver", "{0:1,...}, {0:0.86,...}, {0:0.14,...}")
    fb.add_legend_entry("utility_matrix", "U(i,j) productivity per worker-task pair", "np.ones((4,3))")
    fb.add_legend_entry("fatigue_levels", "MF_i per worker", "[0.1, 0.3, 0.05, 0.6]")
    fb.add_legend_entry("surplus", "U(i,j) - D_i (must be >0)", "0.857")
    fb.add_legend_entry("kappa", "disagreement scaling constant", "1.0")
    fb.add_legend_entry("epsilon", "rest-task surplus (small >0)", "0.001")
    fb.add_legend_entry("assignments", "worker -> task mapping (0=rest)", "{0:1, 1:2, 2:0, 3:3}")

    fb.render(OUT)


# =====================================================================
# 4. hcmarl/pipeline.py
# =====================================================================
def gen_pipeline():
    fb = FlowchartBuilder(
        "hcmarl/pipeline.py",
        "End-to-end HC-MARL control pipeline (Sec 7.3)",
        equations="Steps 1-7 of Section 7.3",
        lines="412",
        imports_desc="numpy, three_cc_r, ecbf_filter, nswf_allocator, utils",
    )
    fb.make_node("TaskProfile", "@dataclass TaskProfile", CLS, [
        ("task_id: int", "1-indexed productive task"),
        ("name: str", "'heavy_lift'"),
        ("demands: dict", "muscle->TL e.g. {'shoulder':0.45}"),
        ("get_load(muscle)", "returns TL for muscle [Def 7.1]"),
    ])
    fb.make_node("WorkerState", "@dataclass WorkerState", CLS, [
        ("worker_id: int", "0-indexed"),
        ("muscle_states: dict", "muscle->ThreeCCrState"),
        ("current_task: int", "None/0 = resting"),
        ("max_fatigue()", "max MF across muscles"),
        ("fresh(id, muscles)", "@classmethod"),
    ])
    fb.make_node("Pipeline_init", "class HCMARLPipeline.__init__", CLS, [
        ("num_workers: int", "N workers"),
        ("muscle_names: list", "tracked muscles"),
        ("task_profiles: list", "available tasks"),
        ("Builds", "ThreeCCr per muscle"),
        ("Builds", "ECBFFilter per muscle"),
        ("Builds", "NSWFAllocator"),
        ("Creates", "WorkerState per worker"),
    ])
    fb.make_node("step1", "Step 1: _observe_states()", FUNC, [
        ("Return", "list[WorkerState]"),
    ])
    fb.make_node("step2", "Step 2: _allocate_tasks(U)", FUNC, [
        ("Input", "utility_matrix (N,M)"),
        ("Calls", "NSWFAllocator.allocate()"),
        ("Return", "AllocationResult"),
    ])
    fb.make_node("steps3_6", "Steps 3-6: _update_worker(worker, task_id)", FUNC, [
        ("Step 3", "Load translation: TL = task.get_load(m)"),
        ("Step 4", "Neural drive: C_nom = baseline_neural_drive(TL,MA)"),
        ("Step 5", "Safety: C_safe = ecbf.filter(state,C_nom,TL)"),
        ("Step 6", "State: new_state = step_euler(state,C_safe,TL,dt)"),
        ("Return", "dict diagnostics per muscle"),
    ])
    fb.make_node("step_main", "step(utility_matrix) — full round", FUNC, [
        ("Calls", "Steps 1-6 for all workers"),
        ("Step 7", "time += dt, step_count++"),
        ("Appends", "history"),
        ("Return", "dict: step, time, allocation, workers"),
    ])
    fb.make_node("from_config", "from_config(config_path) @classmethod", FUNC, [
        ("Input", "YAML config path"),
        ("Parses", "num_workers, muscles, tasks, ECBF, NSWF"),
        ("Return", "HCMARLPipeline instance"),
    ])
    fb.make_node("summary_fn", "summary()", FUNC, [
        ("Return", "human-readable state string"),
    ])

    fb.edge("file_hdr", "TaskProfile", "", style="bold")
    fb.edge("file_hdr", "WorkerState", "", style="bold")
    fb.edge("file_hdr", "Pipeline_init", "", style="bold")
    fb.edge("TaskProfile", "Pipeline_init", "task_profiles")
    fb.edge("WorkerState", "Pipeline_init", "WorkerState.fresh()")
    fb.edge("Pipeline_init", "step1", "self.workers", style="dashed")
    fb.edge("step1", "step2", "workers -> fatigue_levels")
    fb.edge("step2", "steps3_6", "AllocationResult.assignments")
    fb.edge("steps3_6", "step_main", "per-worker diagnostics")
    fb.edge("Pipeline_init", "from_config", "cls()", style="dashed")
    fb.edge("Pipeline_init", "summary_fn", "self.workers", style="dashed")

    fb.dangling_in("Pipeline_init", "ThreeCCr, get_muscle, MuscleParams", "three_cc_r.py")
    fb.dangling_in("Pipeline_init", "ECBFFilter, ECBFParams", "ecbf_filter.py")
    fb.dangling_in("Pipeline_init", "NSWFAllocator, NSWFParams", "nswf_allocator.py")
    fb.dangling_in("from_config", "load_yaml()", "utils.py")
    fb.dangling_in("step_main", "utility_matrix", "scripts/train.py")
    fb.dangling_out("step_main", "step_result dict", "scripts/train.py")
    fb.dangling_out("step_main", "history list", "scripts/evaluate.py")
    fb.dangling_out("Pipeline_init", "HCMARLPipeline", "tests/", is_test=True)

    fb.set_rank("r1", ["TaskProfile", "WorkerState"])
    fb.set_rank("r2", ["Pipeline_init"])
    fb.set_rank("r3", ["step1", "from_config"])
    fb.set_rank("r4", ["step2"])
    fb.set_rank("r5", ["steps3_6"])
    fb.set_rank("r6", ["step_main", "summary_fn"])

    fb.add_legend_entry("task_profiles", "list[TaskProfile] defining tasks and muscle demands", "[TaskProfile(1,'heavy_lift',...), ...]")
    fb.add_legend_entry("WorkerState.fresh()", "@classmethod call to create initial WorkerState (MR=1,MA=0,MF=0)", "WorkerState.fresh(0, ['shoulder'])")
    fb.add_legend_entry("self.workers", "list[WorkerState] stored on pipeline instance", "[WorkerState(0,...), WorkerState(1,...)]")
    fb.add_legend_entry("workers -> fatigue_levels", "MF per worker extracted for allocator", "[0.1, 0.3, 0.05, 0.6]")
    fb.add_legend_entry("AllocationResult.assignments", "dict[worker_id, task_id] from NSWF solver", "{0:1, 1:2, 2:0, 3:3}")
    fb.add_legend_entry("per-worker diagnostics", "dict of MR,MA,MF,TL,C_nom,C_safe,h,h2 per worker", "{'worker_0': {...}, ...}")
    fb.add_legend_entry("cls()", "HCMARLPipeline constructor called from from_config", "HCMARLPipeline(4, ['shoulder'], ...)")
    fb.add_legend_entry("utility_matrix", "U(i,j) shape (N,M)", "np.ones((4,3))")
    fb.add_legend_entry("AllocationResult", "assignments + surpluses + D_i", "{0:1,1:0,...}")
    fb.add_legend_entry("TL", "target load per muscle from task profile", "0.45")
    fb.add_legend_entry("C_nom", "baseline neural drive (pre-safety)", "0.15")
    fb.add_legend_entry("C_safe", "ECBF-filtered neural drive", "0.08")
    fb.add_legend_entry("ThreeCCrState", "muscle state [MR,MA,MF]", "[0.7,0.2,0.1]")
    fb.add_legend_entry("diagnostics", "per-muscle MR,MA,MF,TL,C_nom,C_safe,h,h2,psi_1", "dict")
    fb.add_legend_entry("dt", "integration timestep [minutes]", "1.0")

    fb.render(OUT)


# =====================================================================
# 5. hcmarl/mmicrl.py
# =====================================================================
def gen_mmicrl():
    fb = FlowchartBuilder(
        "hcmarl/mmicrl.py",
        "Multi-Modal Inverse Constrained RL (CFDE flows)",
        equations="Eqs 9-11 (Section 4), Qiao et al. NeurIPS 2023",
        lines="~1100",
        imports_desc="torch, torch.nn, numpy, math",
    )
    fb.make_node("autoregressive_mask", "_get_autoregressive_mask()", FUNC, [
        ("Input", "in/out features, mask_type"),
        ("Logic", "Germain et al. 2015 MADE mask"),
        ("Return", "float Tensor (binary mask)"),
    ])
    fb.make_node("MaskedLinear", "_MaskedLinear (nn.Module)", CLS, [
        ("linear", "weight * mask"),
        ("cond_linear", "optional conditioning input z"),
        ("forward(x, cond)", "masked linear + conditioning"),
    ])
    fb.make_node("MADE", "_MADE (nn.Module)", CLS, [
        ("joiner", "MaskedLinear (input layer)"),
        ("trunk", "ReLU->MaskedLinear->ReLU->MaskedLinear"),
        ("forward(x,cond,mode)", "direct: (u, log_det) | inverse: (x, log_det)"),
        ("Output", "mu, log_alpha per dimension"),
    ])
    fb.make_node("BatchNormFlow", "_BatchNormFlow", CLS, [
        ("Params", "log_gamma, beta, running stats"),
        ("forward()", "batch norm as flow layer"),
    ])
    fb.make_node("Reverse", "_Reverse", CLS, [
        ("Logic", "reverse permutation of dims"),
    ])
    fb.make_node("FlowSequential", "_FlowSequential", CLS, [
        ("forward()", "chain MADE+BN+Reverse, accumulate log_det"),
        ("log_probs(x,cond)", "Gaussian base + log_jacob"),
    ])
    fb.make_node("CFDE", "class CFDE (nn.Module)", CLS, [
        ("flow", "FlowSequential of MADE+BN+Reverse layers"),
        ("log_prior", "learnable type prior log p(z)"),
        ("log_prob(x,z)", "compute log p(x|z)"),
        ("log_prob_all_types(x)", "log p(x|z_k) for all K"),
        ("posterior(x)", "p(z|x) via Bayes rule"),
        ("assign_types(x)", "argmax_z p(z|x)"),
        ("trajectory_log_posterior()", "p(z|tau) aggregated"),
        ("train_density()", "EM: E-step assign, M-step flow"),
    ])
    fb.make_node("ConstraintNet", "class ConstraintNetwork", CLS, [
        ("net", "Linear->ReLU->Linear->ReLU->Linear->Sigmoid"),
        ("forward(s)", "c_theta(s) -> [0,1]"),
        ("train_on_demos()", "BCE on safe/constrained labels"),
        ("Extracts", "theta_max per muscle at c_theta=0.5"),
    ])
    fb.make_node("DemoCollector", "class DemonstrationCollector", CLS, [
        ("demonstrations", "list of trajectories [(s,a),...]"),
        ("collect_from_env()", "record from policy in env"),
        ("generate_synthetic_demos()", "3 types: cautious/moderate/aggressive"),
        ("get_trajectory_features()", "summary: mean_MF, max_MF, rest_frac..."),
        ("get_step_data()", "per-step (s,a) + traj_indices"),
    ])
    fb.make_node("MMICRL_cls", "class MMICRL", CLS, [
        ("n_types: int", "3 latent worker types"),
        ("lambda1, lambda2", "1.0, 1.0 (Eq 9-11)"),
        ("cfde", "CFDE model (trained)"),
        ("theta_max_per_type", "learned thresholds"),
    ])
    fb.make_node("discover_types", "MMICRL._discover_types_cfde()", FUNC, [
        ("Input", "step_features, traj_indices"),
        ("Init", "K-means on traj summaries"),
        ("EM Loop", "M-step: train flow | E-step: reassign"),
        ("Guard", "reject if any type < 5% demos"),
        ("Return", "traj_assignments (n_demos,)"),
    ])
    fb.make_node("compute_mi", "MMICRL._compute_mutual_information()", FUNC, [
        ("Formula", "I(tau;z) = H(z) - H(z|tau)"),
        ("Soft", "from CFDE trajectory posteriors"),
        ("Fallback", "hard assignment entropy"),
        ("Return", "float MI value"),
    ])
    fb.make_node("learn_constraints", "MMICRL._learn_constraints()", FUNC, [
        ("Per type k", "collect states, train ConstraintNet"),
        ("Boundary", "c_theta(s) = 0.5 crossing"),
        ("Return", "dict[type_k, dict[muscle, theta_max]]"),
    ])
    fb.make_node("fit", "MMICRL.fit(collector)", FUNC, [
        ("Step 1", "get_step_data() from collector"),
        ("Step 2", "_discover_types_cfde()"),
        ("Step 3", "_compute_mutual_information()"),
        ("Step 4", "_learn_constraints()"),
        ("Objective", "lambda2*MI + (l1-l2)*H[pi] [Eq 10]"),
        ("Return", "results dict"),
    ])
    fb.make_node("get_threshold", "get_threshold_for_worker()", FUNC, [
        ("Input", "worker trajectory features"),
        ("Uses", "CFDE posterior for type assignment"),
        ("Return", "dict[muscle, theta_max]"),
    ])

    # --- Edges ---
    fb.edge("file_hdr", "CFDE", "", style="bold")
    fb.edge("file_hdr", "ConstraintNet", "", style="bold")
    fb.edge("file_hdr", "DemoCollector", "", style="bold")
    fb.edge("file_hdr", "MMICRL_cls", "", style="bold")
    fb.edge("autoregressive_mask", "MaskedLinear", "mask tensor")
    fb.edge("MaskedLinear", "MADE", "layers")
    fb.edge("MADE", "FlowSequential", "modules")
    fb.edge("BatchNormFlow", "FlowSequential", "modules")
    fb.edge("Reverse", "FlowSequential", "modules")
    fb.edge("FlowSequential", "CFDE", "self.flow")
    fb.edge("CFDE", "MMICRL_cls", "self.cfde")
    fb.edge("ConstraintNet", "learn_constraints", "c_net per type")
    fb.edge("DemoCollector", "fit", "collector.get_step_data()")
    fb.edge("MMICRL_cls", "fit", "self.cfde, self.n_types", style="dashed")
    fb.edge("fit", "discover_types", "step_features, traj_indices")
    fb.edge("discover_types", "compute_mi", "assignments")
    fb.edge("fit", "learn_constraints", "demonstrations, assignments")
    fb.edge("fit", "get_threshold", "trained CFDE", style="dashed")

    # --- Dangling IN ---
    fb.dangling_in("MADE", "nn.Module, nn.Linear", "torch.nn")
    fb.dangling_in("DemoCollector", "env, policy", "warehouse_env.py")

    # --- Dangling OUT ---
    fb.dangling_out("get_threshold", "theta_max per muscle per worker", "ecbf_filter.py")
    fb.dangling_out("fit", "results: MI, type_proportions, theta_per_type", "scripts/train.py")
    fb.dangling_out("CFDE", "CFDE model", "tests/", is_test=True)
    fb.dangling_out("DemoCollector", "demonstrations", "tests/", is_test=True)

    fb.set_rank("r1", ["autoregressive_mask", "MaskedLinear"])
    fb.set_rank("r2", ["MADE", "BatchNormFlow", "Reverse"])
    fb.set_rank("r3", ["FlowSequential"])
    fb.set_rank("r4", ["CFDE", "ConstraintNet", "DemoCollector"])
    fb.set_rank("r5", ["MMICRL_cls"])
    fb.set_rank("r6", ["discover_types", "compute_mi", "learn_constraints"])
    fb.set_rank("r7", ["fit", "get_threshold"])

    fb.add_legend_entry("mask tensor", "binary autoregressive mask applied to weight matrix", "torch.Tensor shape (out, in) of 0/1")
    fb.add_legend_entry("layers", "list of MaskedLinear modules assembled into MADE", "[MaskedLinear(14,256), MaskedLinear(256,28)]")
    fb.add_legend_entry("modules", "nn.Module instances added to FlowSequential chain", "[MADE(...), BatchNormFlow(...), Reverse(...)]")
    fb.add_legend_entry("self.flow", "FlowSequential instance stored on CFDE", "FlowSequential([MADE, BN, Reverse, ...])")
    fb.add_legend_entry("self.cfde", "trained CFDE model stored on MMICRL instance", "CFDE(n_blocks=5, hidden=256, K=3)")
    fb.add_legend_entry("c_net per type", "ConstraintNetwork instance trained per type k", "ConstraintNetwork(input_dim=14)")
    fb.add_legend_entry("collector.get_step_data()", "returns (step_features, traj_indices) arrays", "(np.ndarray(12000,14), np.ndarray(12000,))")
    fb.add_legend_entry("self.cfde, self.n_types", "CFDE model and type count forwarded to fit()", "CFDE(...), 3")
    fb.add_legend_entry("step_features, traj_indices", "per-step feature array + trajectory-id mapping", "(12000, 14) float32, [0,0,...,199]")
    fb.add_legend_entry("demonstrations, assignments", "raw demo trajectories + type labels for constraint learning", "list[traj], [0,2,1,...]")
    fb.add_legend_entry("trained CFDE", "fitted CFDE forwarded to get_threshold for posterior inference", "CFDE after M-step convergence")
    fb.add_legend_entry("assignments", "trajectory -> type_k mapping", "[0,2,1,0,1,...]")
    fb.add_legend_entry("z_onehot", "one-hot type code", "[0,1,0] for type 1")
    fb.add_legend_entry("log_prob", "log p(x|z) from flow", "-3.42")
    fb.add_legend_entry("posterior", "p(z|x) or p(z|tau)", "[0.1, 0.8, 0.1]")
    fb.add_legend_entry("MI I(tau;z)", "mutual information bits", "0.95")
    fb.add_legend_entry("theta_per_type", "learned thresholds per type per muscle", "{0:{'shoulder':0.55}}")
    fb.add_legend_entry("lambda1, lambda2", "MMICRL weighting (Eq 9)", "1.0, 1.0")

    fb.render(OUT)


# =====================================================================
# 6. hcmarl/real_data_calibration.py
# =====================================================================
def gen_real_data_calibration():
    fb = FlowchartBuilder(
        "hcmarl/real_data_calibration.py",
        "Path G: WSD4FEDSRM calibration + correlated sampling",
        equations="Eqs 2-4, 35; Frey-Law et al. 2012 method",
        lines="~700",
        imports_desc="numpy, csv, os, hcmarl.three_cc_r",
    )
    fb.make_node("predict_et", "predict_endurance_time(F,R,r,TL,...)", FUNC, [
        ("Input", "F, R, r, target_load, duty_cycle"),
        ("Method", "inline 3CC-r ODE + Eq 35 controller"),
        ("Exhaustion", "MR < 1e-4 or MA < TL*0.5"),
        ("Return", "endurance time [seconds]"),
    ])
    fb.make_node("calibrate_F", "calibrate_F_for_subject(obs_ET,...)", FUNC, [
        ("Input", "dict {TL: observed_ET_seconds}"),
        ("Stage 1", "coarse log-spaced grid (100 pts)"),
        ("Stage 2", "fine linear search around best"),
        ("Metric", "RMS error across load levels"),
        ("Return", "(F_opt, rms_error)"),
    ])
    fb.make_node("dyn_iso_report", "compute_dynamic_isometric_report()", FUNC, [
        ("Input", "calibration_results dict"),
        ("Computes", "F_dynamic / F_isometric ratio per subject"),
        ("Cross-val", "ET at 35% with dynamic vs isometric F"),
        ("Return", "report dict: ratios, means, validations"),
    ])
    fb.make_node("load_wsd", "load_wsd4fedsrm(data_dir)", FUNC, [
        ("Reads", "demographic.csv, MVIC_force_data.csv, borg_data.csv"),
        ("Extracts", "per-subject: MVIC, endurance times, RPE, demographics"),
        ("Return", "dict[subject_id, subject_data]"),
    ])
    fb.make_node("task_mvic", "TASK_TO_MVIC_FRACTION (const)", CONST, [
        ("task1_35i", "0.35 (30-40% IR midpoint)"),
        ("task2_45i", "0.45"),
        ("task3_55i", "0.55"),
        ("+ 3 ER tasks", "same %MVIC levels"),
    ])
    fb.make_node("pop_model", "ENDURANCE_POWER_MODEL (const)", CONST, [
        ("Formula", "ET = b0 * (%MVC)^b1 [seconds]"),
        ("Source", "Frey-Law & Avin 2010 Table 2"),
        ("6 muscles", "shoulder, ankle, knee, elbow, grip, trunk"),
    ])
    fb.make_node("pop_FR", "POPULATION_FR, CV_F, CV_R (const)", CONST, [
        ("Source", "Frey-Law et al. 2012 Table 1"),
        ("CV_F", "0.30 default, 0.36 elbow (Liu 2002)"),
        ("CV_R", "0.40 (Liu 2002)"),
    ])
    fb.make_node("sample_FR", "sample_FR_from_population(muscle,n)", FUNC, [
        ("Method", "log-normal sampling, positivity guaranteed"),
        ("Return", "list[(F_i, R_i)]"),
    ])
    fb.make_node("sample_corr", "sample_correlated_FR(muscle,shoulder_Fs)", FUNC, [
        ("Model", "z_i from shoulder F, rho=0.5 correlation"),
        ("Formula", "log(F_m) = log(F_pop) + sigma*(rho*z + sqrt(1-rho^2)*eps)"),
        ("Return", "list[(F_i, R_i)] conditioned on shoulder"),
    ])
    fb.make_node("run_path_g", "run_path_g(data_dir) — main entry", FUNC, [
        ("Step 1", "load_wsd4fedsrm()"),
        ("Step 2", "calibrate_F per subject"),
        ("Step 3", "sample_correlated_FR for non-shoulder"),
        ("Step 4", "dynamic_isometric_report"),
        ("Step 5", "generate_calibrated_demos()"),
        ("Return", "full calibration results"),
    ])
    fb.make_node("gen_demos", "generate_calibrated_demos()", FUNC, [
        ("Input", "calibrated (F_i,R_i) per subject per muscle"),
        ("Method", "simulate 3CC-r with realistic task sequences"),
        ("Return", "list of demo trajectories for MMICRL"),
    ])

    fb.edge("file_hdr", "predict_et", "", style="bold")
    fb.edge("file_hdr", "calibrate_F", "", style="bold")
    fb.edge("file_hdr", "load_wsd", "", style="bold")
    fb.edge("file_hdr", "run_path_g", "", style="bold")
    fb.edge("predict_et", "calibrate_F", "predicted ET per F")
    fb.edge("calibrate_F", "dyn_iso_report", "per-subject F_opt")
    fb.edge("load_wsd", "run_path_g", "subject data")
    fb.edge("task_mvic", "run_path_g", "TL fractions")
    fb.edge("calibrate_F", "run_path_g", "F_opt per subject")
    fb.edge("sample_corr", "run_path_g", "non-shoulder (F,R)")
    fb.edge("pop_FR", "sample_FR", "population means")
    fb.edge("pop_FR", "sample_corr", "F_pop, R_pop, CV")
    fb.edge("pop_model", "dyn_iso_report", "isometric ET reference")
    fb.edge("run_path_g", "gen_demos", "calibrated params")

    fb.dangling_in("predict_et", "3CC-r ODE structure", "three_cc_r.py (SHOULDER)")
    fb.dangling_in("load_wsd", "WSD4FEDSRM/ CSV files", "data/WSD4FEDSRM/")
    fb.dangling_out("gen_demos", "demo trajectories", "mmicrl.py (DemoCollector)")
    fb.dangling_out("run_path_g", "calibration results", "scripts/train.py")
    fb.dangling_out("sample_corr", "per-subject (F_i,R_i)", "tests/", is_test=True)

    fb.set_rank("r1", ["predict_et", "load_wsd"])
    fb.set_rank("r2", ["calibrate_F", "task_mvic"])
    fb.set_rank("r3", ["pop_model", "pop_FR"])
    fb.set_rank("r4", ["sample_FR", "sample_corr", "dyn_iso_report"])
    fb.set_rank("r5", ["run_path_g"])
    fb.set_rank("r6", ["gen_demos"])

    fb.add_legend_entry("predicted ET per F", "endurance time [sec] computed by predict_et for candidate F", "87.4")
    fb.add_legend_entry("per-subject F_opt", "best-fit F and rms_error per subject from calibrate_F", "(1.24, 4.7)")
    fb.add_legend_entry("subject data", "dict[subject_id, {MVIC, ET, RPE, demographics}]", "{'S01': {'MVIC': 142.3, ...}}")
    fb.add_legend_entry("TL fractions", "target-load fractions from TASK_TO_MVIC_FRACTION", "{task1_35i: 0.35, task2_45i: 0.45, ...}")
    fb.add_legend_entry("F_opt per subject", "array of calibrated F values across subjects", "[1.24, 0.98, 1.41, ...]")
    fb.add_legend_entry("non-shoulder (F,R)", "correlated (F_i, R_i) samples for non-shoulder muscles", "[(0.015, 0.0006), ...]")
    fb.add_legend_entry("population means", "POPULATION_FR mean F, mean R per muscle", "{'shoulder': (0.01820, 0.00168)}")
    fb.add_legend_entry("F_pop, R_pop, CV", "population mean + coefficient of variation for log-normal sampling", "0.01820, 0.00168, 0.30")
    fb.add_legend_entry("isometric ET reference", "ET predicted from isometric power model for cross-validation", "105.0")
    fb.add_legend_entry("calibrated params", "per-subject per-muscle (F_i,R_i) passed to demo generator", "{'S01': {'shoulder': (1.24, ...)}, ...}")
    fb.add_legend_entry("F, R", "fatigue/recovery rate [min^-1]", "F=1.2, R=0.02")
    fb.add_legend_entry("observed_ET", "measured endurance time [seconds]", "{0.35: 105, 0.45: 76}")
    fb.add_legend_entry("F_opt", "calibrated fatigue rate for subject", "1.24")
    fb.add_legend_entry("rms_error", "RMS calibration error [seconds]", "4.7")
    fb.add_legend_entry("scaling_ratio", "F_dynamic / F_isometric", "85x")
    fb.add_legend_entry("rho", "inter-muscle correlation for F", "0.5")
    fb.add_legend_entry("z_i", "standardised shoulder F deviate", "1.3")
    fb.add_legend_entry("b0, b1", "power-model coefficients (Frey-Law & Avin)", "891.6, -1.83")

    fb.render(OUT)


# =====================================================================
# 7. hcmarl/warehouse_env.py
# =====================================================================
def gen_warehouse_env():
    fb = FlowchartBuilder(
        "hcmarl/warehouse_env.py",
        "Warehouse environment (single + multi-agent)",
        equations="Eqs 2-4, 19, 23, 32-33, 35",
        lines="452",
        imports_desc="numpy, gymnasium, hcmarl.envs.reward_functions",
    )
    fb.make_node("SingleEnv_init", "class SingleWorkerWarehouseEnv.__init__", CLS, [
        ("muscle_groups", "dict: shoulder/elbow/grip -> F,R,r"),
        ("tasks", "dict: heavy_lift/light_sort/carry/rest -> TLs"),
        ("theta_max", "dict: per-muscle safety thresholds"),
        ("ecbf_mode", "'on'|'off'"),
        ("obs_space", "Box(n_muscles*3 + 1)"),
        ("action_space", "Discrete(n_tasks)"),
    ])
    fb.make_node("single_obs", "_get_obs()", FUNC, [
        ("Builds", "[MR,MA,MF] per muscle + step/max_steps"),
        ("Return", "np.float32 array"),
    ])
    fb.make_node("single_reward", "_compute_reward(task_name)", FUNC, [
        ("Calls", "nswf_reward(productivity, fatigue, theta_max)"),
        ("Return", "float"),
    ])
    fb.make_node("integrate_3ccr", "_integrate_3cc_r(task_name)", FUNC, [
        ("Per muscle", "C_nom = kp*max(TL-MA,0) [Eq 35]"),
        ("ECBF on", "ecbf_bound [Eq 19] + cbf_bound [Eq 23]"),
        ("ODE", "Euler step Eqs 2-4 + clamp + renorm"),
        ("Return", "(ecbf_interventions, ecbf_clip_total)"),
    ])
    fb.make_node("single_reset", "reset()", FUNC, [
        ("Sets", "all muscles MR=1,MA=0,MF=0"),
        ("Return", "(obs, info)"),
    ])
    fb.make_node("single_step", "step(action)", FUNC, [
        ("1", "task_name from action index"),
        ("2", "_integrate_3cc_r(task_name)"),
        ("3", "obs, reward, terminated, info"),
        ("info", "task, fatigue, violations, cost, ecbf"),
    ])
    fb.make_node("MultiEnv_init", "class WarehouseMultiAgentEnv.__init__", CLS, [
        ("n_workers", "4 agents"),
        ("possible_agents", "['worker_0',...,'worker_3']"),
        ("observation_spaces", "per-agent Box"),
        ("action_spaces", "per-agent Discrete"),
        ("global_obs_space", "N*muscles*3 + 1"),
    ])
    fb.make_node("multi_step", "WarehouseMultiAgentEnv.step(actions)", FUNC, [
        ("Input", "dict[agent, action_int]"),
        ("Per agent", "_integrate_worker(idx, task_name)"),
        ("Return", "(obs, rewards, terms, truncs, infos)"),
    ])
    fb.make_node("make_helpers", "make_single_env() / make_multi_env()", FUNC, [
        ("Return", "env instance"),
    ])

    fb.edge("file_hdr", "SingleEnv_init", "", style="bold")
    fb.edge("file_hdr", "MultiEnv_init", "", style="bold")
    fb.edge("SingleEnv_init", "single_obs", "self.state", style="dashed")
    fb.edge("SingleEnv_init", "integrate_3ccr", "self.muscle_groups,theta_max", style="dashed")
    fb.edge("single_obs", "single_step", "obs")
    fb.edge("single_reward", "single_step", "reward")
    fb.edge("integrate_3ccr", "single_step", "ecbf_interventions")
    fb.edge("single_reset", "single_step", "self.state", style="dashed")
    fb.edge("MultiEnv_init", "multi_step", "self.states per worker", style="dashed")

    fb.dangling_in("single_reward", "nswf_reward(), safety_cost()", "envs/reward_functions.py")
    fb.dangling_in("SingleEnv_init", "gym.Env, spaces", "gymnasium")
    fb.dangling_out("single_step", "obs, reward, done, info", "agents/mappo.py")
    fb.dangling_out("multi_step", "obs, rewards, terms, infos", "agents/hcmarl_agent.py")
    fb.dangling_out("make_helpers", "env instances", "scripts/train.py")
    fb.dangling_out("SingleEnv_init", "env", "tests/", is_test=True)

    fb.set_rank("r1", ["SingleEnv_init", "MultiEnv_init"])
    fb.set_rank("r2", ["single_obs", "single_reward", "integrate_3ccr"])
    fb.set_rank("r3", ["single_reset", "single_step", "multi_step"])
    fb.set_rank("r4", ["make_helpers"])

    fb.add_legend_entry("self.state", "dict muscle->ThreeCCrState stored on env, reset to fresh on reset()", "{'shoulder': ThreeCCrState(1,0,0)}")
    fb.add_legend_entry("self.muscle_groups,theta_max", "muscle parameter dicts forwarded to integrator", "{'shoulder':{F:0.01820,...}}, {'shoulder':0.7}")
    fb.add_legend_entry("obs", "flattened [MR,MA,MF]*muscles + step_norm", "[0.7,0.2,0.1,...,0.5]")
    fb.add_legend_entry("reward", "NSWF reward (Eq 33 style)", "0.85")
    fb.add_legend_entry("ecbf_interventions", "count of muscles where ECBF clipped C", "1")
    fb.add_legend_entry("self.states per worker", "list[dict muscle->ThreeCCrState] per agent in multi-env", "[{'shoulder': ThreeCCrState(...)}, ...]")
    fb.add_legend_entry("action", "task index (0=heavy,1=light,2=carry,3=rest)", "2")
    fb.add_legend_entry("fatigue", "dict muscle->MF", "{'shoulder':0.15}")
    fb.add_legend_entry("violations", "count of MF > theta_max", "0")
    fb.add_legend_entry("C_nominal", "kp*max(TL-MA,0)", "1.5")
    fb.add_legend_entry("ecbf_bound", "ECBF upper bound on C [Eq 19]", "0.8")

    fb.render(OUT)


# =====================================================================
# 8. hcmarl/utils.py
# =====================================================================
def gen_utils():
    fb = FlowchartBuilder(
        "hcmarl/utils.py",
        "Shared utility functions",
        lines="166",
        imports_desc="logging, os, pathlib, numpy, yaml",
    )
    fb.make_node("get_logger", "get_logger(name, level=INFO)", FUNC, [
        ("Input", "module name e.g. __name__"),
        ("Creates", "StreamHandler with timestamp format"),
        ("Return", "logging.Logger"),
    ])
    fb.make_node("load_yaml", "load_yaml(path)", FUNC, [
        ("Input", "str or Path to .yaml"),
        ("Uses", "yaml.safe_load()"),
        ("Return", "dict[str, Any]"),
    ])
    fb.make_node("resolve_root", "resolve_project_root()", FUNC, [
        ("Logic", "walk up from __file__ until setup.py/.git"),
        ("Return", "Path to project root"),
    ])
    fb.make_node("clip_norm", "clip_and_normalise(x)", FUNC, [
        ("Input", "np.ndarray"),
        ("Logic", "clip [0,1] then x/sum(x)"),
        ("Purpose", "enforce MR+MA+MF=1 (Eq 1)"),
        ("Return", "np.ndarray"),
    ])
    fb.make_node("safe_log_fn", "safe_log(x, floor=1e-20)", FUNC, [
        ("Logic", "log(max(x, floor))"),
        ("Used by", "NSWF objective ln(surplus)"),
        ("Return", "float"),
    ])
    fb.make_node("is_pd", "is_positive_definite(matrix)", FUNC, [
        ("Logic", "all eigenvalues > 0"),
        ("Return", "bool"),
    ])
    fb.make_node("seed_fn", "seed_everything(seed)", FUNC, [
        ("Sets", "np.random, PYTHONHASHSEED"),
        ("Optional", "torch.manual_seed if available"),
    ])

    fb.edge("file_hdr", "get_logger", "", style="bold")
    fb.edge("file_hdr", "load_yaml", "", style="bold")
    fb.edge("file_hdr", "clip_norm", "", style="bold")
    fb.edge("file_hdr", "safe_log_fn", "", style="bold")
    fb.edge("file_hdr", "seed_fn", "", style="bold")
    fb.edge("file_hdr", "resolve_root", "", style="bold")
    fb.edge("file_hdr", "is_pd", "", style="bold")

    fb.dangling_in("get_logger", "logging module", "logging")
    fb.dangling_in("load_yaml", "yaml.safe_load", "pyyaml")
    fb.dangling_out("get_logger", "logger", "pipeline.py, logger.py")
    fb.dangling_out("load_yaml", "config dict", "pipeline.py")
    fb.dangling_out("safe_log_fn", "safe_log", "nswf_allocator.py")
    fb.dangling_out("clip_norm", "clip_and_normalise", "three_cc_r.py")
    fb.dangling_out("seed_fn", "seed_everything", "scripts/train.py")

    fb.add_legend_entry("name", "logger name (usually __name__)", "'hcmarl.pipeline'")
    fb.add_legend_entry("path", "YAML config file path", "'config/default_config.yaml'")
    fb.add_legend_entry("x", "state array to clip+normalise", "[0.7, 0.2, 0.12]")
    fb.add_legend_entry("safe_log result", "log with floor guard", "-6.9")
    fb.add_legend_entry("seed", "random seed integer", "42")

    fb.render(OUT)


# =====================================================================
# 9. hcmarl/logger.py
# =====================================================================
def gen_logger():
    fb = FlowchartBuilder(
        "hcmarl/logger.py",
        "HC-MARL training logger (W&B + CSV)",
        lines="92",
        imports_desc="numpy, csv, os, collections.defaultdict",
    )
    fb.make_node("Logger_init", "class HCMARLLogger.__init__", CLS, [
        ("log_dir", "'logs/' (creates if needed)"),
        ("use_wandb", "bool (optional W&B)"),
        ("csv_path", "logs/training_log.csv"),
        ("history", "defaultdict(list)"),
    ])
    fb.make_node("METRICS", "METRIC_NAMES (class var)", CONST, [
        ("9 metrics", "violation_rate, cumulative_cost, safety_rate"),
        ("", "tasks_completed, cumulative_reward, jain_fairness"),
        ("", "peak_fatigue, forced_rest_rate, constraint_recovery_time"),
    ])
    fb.make_node("log_step", "log_step(metrics)", FUNC, [
        ("Input", "dict[str, float]"),
        ("Appends", "to self.history"),
        ("W&B", "wandb.log if enabled"),
    ])
    fb.make_node("log_episode", "log_episode(metrics)", FUNC, [
        ("Input", "dict[str, float]"),
        ("Writes", "CSV row (DictWriter)"),
        ("W&B", "wandb.log if enabled"),
    ])
    fb.make_node("compute_ep", "compute_episode_metrics(episode_data)", FUNC, [
        ("Input", "raw episode dict (violations, rewards, etc)"),
        ("Computes", "all 9 metrics from raw data"),
        ("Jain", "(sum^2)/(n*sum_sq) fairness index"),
        ("Return", "dict[str, float]"),
    ])
    fb.make_node("close_fn", "close()", FUNC, [
        ("Action", "wandb.finish() if enabled"),
    ])

    fb.edge("file_hdr", "Logger_init", "", style="bold")
    fb.edge("file_hdr", "METRICS", "", style="bold")
    fb.edge("Logger_init", "log_step", "self.history", style="dashed")
    fb.edge("Logger_init", "log_episode", "self.csv_path", style="dashed")
    fb.edge("compute_ep", "log_episode", "metrics dict")
    fb.edge("Logger_init", "close_fn", "self.wandb", style="dashed")

    fb.dangling_in("Logger_init", "wandb (optional)", "wandb")
    fb.dangling_in("compute_ep", "episode_data from training loop", "scripts/train.py")
    fb.dangling_out("log_episode", "training_log.csv", "logs/")
    fb.dangling_out("log_step", "W&B dashboard", "wandb.ai")

    fb.add_legend_entry("self.history", "defaultdict(list) accumulating per-step metric scalars", "defaultdict({'violation_rate': [0.02, 0.01, ...]})")
    fb.add_legend_entry("self.csv_path", "path string to training_log.csv written by log_episode", "'logs/training_log.csv'")
    fb.add_legend_entry("metrics dict", "dict[str, float] of 9 computed episode metrics", "{'violation_rate':0.02, 'jain_fairness':0.92, ...}")
    fb.add_legend_entry("self.wandb", "wandb run object (or None) stored on HCMARLLogger", "wandb.run or None")
    fb.add_legend_entry("metrics", "dict of 9 training metrics", "{'violation_rate':0.02,...}")
    fb.add_legend_entry("violation_rate", "violations / (steps*workers*muscles)", "0.015")
    fb.add_legend_entry("jain_fairness", "Jain's fairness index [0,1]", "0.92")
    fb.add_legend_entry("peak_fatigue", "max MF observed in episode", "0.68")
    fb.add_legend_entry("episode_data", "raw aggregated episode stats", "dict")

    fb.render(OUT)


# =====================================================================
# Run all core module generators
# =====================================================================
if __name__ == "__main__":
    generators = [
        gen_three_cc_r,
        gen_ecbf_filter,
        gen_nswf_allocator,
        gen_pipeline,
        gen_mmicrl,
        gen_real_data_calibration,
        gen_warehouse_env,
        gen_utils,
        gen_logger,
    ]
    for gen_fn in generators:
        try:
            gen_fn()
        except Exception as e:
            print(f"FAIL: {gen_fn.__name__}: {e}")
