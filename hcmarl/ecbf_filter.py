"""
ECBF Safety Filter: Exponential Control Barrier Functions.

Implements the dual-barrier CBF-QP from:
    - Nguyen & Sreenath (2016): Exponential CBFs for high relative degree
    - Xiao & Belta (2019): Generalised high-relative-degree CBFs
    - Ames et al. (2017): CBF-based QP for safety-critical control
    - Prajna & Jadbabaie (2004): Barrier certificate foundations

Mathematical reference: HC-MARL Framework v12, Sections 5.1--5.4.
All equation numbers below refer to that document.

The filter enforces two safety constraints simultaneously:
    1. Fatigue ceiling: h(x) = Theta_max - MF >= 0  (relative degree 2)
    2. Resting floor:  h2(x) = MR >= 0               (relative degree 1)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import cvxpy as cp
import numpy as np

from hcmarl.three_cc_r import MuscleParams, ThreeCCrState


@dataclasses.dataclass
class ECBFParams:
    """Design parameters for the ECBF safety filter.

    Attributes:
        theta_max: Maximum allowable fatigue fraction (Eq 12).
            Must satisfy theta_max >= F/(F + R*r) for rest-phase safety
            (Assumption 5.5, Eq 26).
        alpha1: Positive class-K gain for ECBF composite barrier psi_1 (Eq 16).
        alpha2: Positive class-K gain for ECBF enforcement condition (Eq 17).
        alpha3: Positive class-K gain for resting-floor CBF (Eq 23).
    """

    theta_max: float
    alpha1: float = 0.05
    alpha2: float = 0.05
    alpha3: float = 0.1

    def validate(self, muscle: MuscleParams) -> None:
        """Check design requirements.

        Raises:
            ValueError: If theta_max violates Assumption 5.5 (Eq 26) or
                if any gain is non-positive.
        """
        if self.theta_max <= 0.0 or self.theta_max >= 1.0:
            raise ValueError(
                f"theta_max must be in (0, 1), got {self.theta_max}"
            )
        if self.theta_max < muscle.theta_min_max:
            raise ValueError(
                f"theta_max = {self.theta_max:.4f} violates Assumption 5.5 "
                f"(Eq 26): must be >= theta_min_max = {muscle.theta_min_max:.4f} "
                f"for {muscle.name}."
            )
        for attr in ("alpha1", "alpha2", "alpha3"):
            val = getattr(self, attr)
            if val <= 0.0:
                raise ValueError(f"{attr} must be positive, got {val}")


@dataclasses.dataclass
class ECBFDiagnostics:
    """Diagnostic information from one QP solve.

    Useful for logging and debugging.
    """

    C_nominal: float      # Input nominal neural drive
    C_filtered: float     # Output safe neural drive
    h: float              # Fatigue barrier h(x) = Theta_max - MF  (Eq 12)
    h2: float             # Resting barrier h2(x) = MR             (Eq 21)
    psi_0: float          # = h(x)                                  (Eq 15)
    psi_1: float          # = h_dot + alpha1 * h                    (Eq 16)
    h_dot: float          # = -F*MA + Reff*MF                       (Eq 13)
    C_upper_ecbf: float   # Upper bound from ECBF constraint (Eq 19)
    C_upper_cbf: float    # Upper bound from resting CBF (Eq 23)
    qp_status: str        # CVXPY solver status
    was_clipped: bool     # Whether C_nominal was modified
    infeasible: bool = False  # S-3: True when QP infeasible (mandatory rest)


class ECBFFilter:
    """Dual-barrier ECBF safety filter.

    Solves the CBF-QP (Eq 20):
        C*(t) = argmin_{C} ||C - C_nom||^2
        subject to:
            ECBF constraint (Eq 18)  -- fatigue ceiling, rel. degree 2
            CBF  constraint (Eq 23)  -- resting floor, rel. degree 1
            C >= 0

    Both constraints are linear in C, so the QP is convex (Remark 5.3,
    Boyd & Vandenberghe 2004) and efficiently solvable via OSQP
    (Stellato et al. 2020).

    Args:
        muscle: Calibrated muscle parameters.
        ecbf_params: ECBF design parameters (gains and threshold).
    """

    def __init__(
        self,
        muscle: MuscleParams,
        ecbf_params: ECBFParams,
    ) -> None:
        self.muscle = muscle
        self.params = ecbf_params
        self.params.validate(muscle)

        # Pre-extract for readability
        self._F = muscle.F
        self._R = muscle.R
        self._Rr = muscle.Rr
        self._theta_max = ecbf_params.theta_max
        self._alpha1 = ecbf_params.alpha1
        self._alpha2 = ecbf_params.alpha2
        self._alpha3 = ecbf_params.alpha3

    # -----------------------------------------------------------------
    # Barrier function computations
    # -----------------------------------------------------------------

    def h(self, MF: float) -> float:
        """Fatigue ceiling barrier (Eq 12): h(x) = Theta_max - MF."""
        return self._theta_max - MF

    def h2(self, MR: float) -> float:
        """Resting floor barrier (Eq 21): h2(x) = MR = 1 - MA - MF."""
        return MR

    def h_dot(self, MA: float, MF: float, R_eff: float) -> float:
        """First derivative of fatigue barrier (Eq 13).

        h_dot = -dMF/dt = -(F*MA - Reff*MF) = -F*MA + Reff*MF

        Note: The control input C does NOT appear. This confirms
        relative degree >= 2 for the fatigue barrier.
        """
        return -self._F * MA + R_eff * MF

    def h_ddot(
        self, MA: float, MF: float, C: float, R_eff: float
    ) -> float:
        """Second derivative of fatigue barrier (Eq 14).

        h_ddot = -F*C + F^2*MA + Reff*F*MA - Reff^2*MF

        The input C appears with coefficient -F. Since F > 0, the
        system has relative degree 2 w.r.t. h.
        """
        F = self._F
        return -F * C + F**2 * MA + R_eff * F * MA - R_eff**2 * MF

    def psi_0(self, MF: float) -> float:
        """Composite barrier degree 0 (Eq 15): psi_0 = h(x)."""
        return self.h(MF)

    def psi_1(self, MA: float, MF: float, R_eff: float) -> float:
        """Composite barrier degree 1 (Eq 16): psi_1 = h_dot + alpha1 * h."""
        return self.h_dot(MA, MF, R_eff) + self._alpha1 * self.h(MF)

    def h2_dot(self, MF: float, C: float, R_eff: float) -> float:
        """First derivative of resting barrier (Eq 22).

        h2_dot = dMR/dt = Reff*MF - C

        The input C appears with coefficient -1. Relative degree 1.
        """
        return R_eff * MF - C

    # -----------------------------------------------------------------
    # Constraint bounds (analytical, for diagnostics and fallback)
    # -----------------------------------------------------------------

    def ecbf_upper_bound(
        self, MA: float, MF: float, R_eff: float
    ) -> float:
        """Analytical upper bound on C from ECBF constraint (Eq 19).

        Derived by solving the ECBF condition (Eq 18) for C.
        Since the coefficient of C is -F < 0, the inequality flips
        when isolating C:

        C <= (1/F) * [F^2*MA + Reff*F*MA - Reff^2*MF
                      + alpha1*(-F*MA + Reff*MF)
                      + alpha2*(-F*MA + Reff*MF + alpha1*(Theta_max - MF))]
        """
        F = self._F
        a1 = self._alpha1
        a2 = self._alpha2
        tm = self._theta_max

        # State-dependent terms from h_ddot (without the -F*C term)
        state_terms = F**2 * MA + R_eff * F * MA - R_eff**2 * MF

        # alpha1 * h_dot
        a1_term = a1 * (-F * MA + R_eff * MF)

        # alpha2 * psi_1
        psi1_val = (-F * MA + R_eff * MF) + a1 * (tm - MF)
        a2_term = a2 * psi1_val

        return (state_terms + a1_term + a2_term) / F

    def cbf_upper_bound(
        self, MA: float, MF: float, R_eff: float
    ) -> float:
        """Analytical upper bound on C from resting-floor CBF (Eq 23).

        C <= Reff*MF + alpha3*(1 - MA - MF)
        """
        MR = 1.0 - MA - MF
        return R_eff * MF + self._alpha3 * MR

    # -----------------------------------------------------------------
    # QP solver (Eq 20)
    # -----------------------------------------------------------------

    def filter(
        self,
        state: ThreeCCrState,
        C_nominal: float,
        target_load: float,
        solver: str = "OSQP",
    ) -> tuple[float, ECBFDiagnostics]:
        """Solve the CBF-QP to compute the safe neural drive.

        CBF-QP (Eq 20):
            C* = argmin_{C >= 0} ||C - C_nom||^2
            subject to:
                ECBF constraint (Eq 18)  [fatigue ceiling]
                CBF  constraint (Eq 23)  [resting floor]

        Args:
            state: Current physiological state [MR, MA, MF].
            C_nominal: Nominal neural drive from RL policy or baseline.
            target_load: TL(t), used to determine Reff.
            solver: CVXPY solver name (default OSQP).

        Returns:
            Tuple of (C_filtered, diagnostics).
        """
        MR, MA, MF = state.MR, state.MA, state.MF
        F = self._F

        # Determine Reff (Eq 5)
        R_eff = self.muscle.R if target_load > 0.0 else self.muscle.Rr

        # Compute barrier values for diagnostics
        h_val = self.h(MF)
        h2_val = self.h2(MR)
        h_dot_val = self.h_dot(MA, MF, R_eff)
        psi0_val = self.psi_0(MF)
        psi1_val = self.psi_1(MA, MF, R_eff)
        C_ub_ecbf = self.ecbf_upper_bound(MA, MF, R_eff)
        C_ub_cbf = self.cbf_upper_bound(MA, MF, R_eff)

        # ----- Build QP -----
        C_var = cp.Variable(1)

        # Objective: minimise ||C - C_nom||^2
        objective = cp.Minimize(cp.sum_squares(C_var - C_nominal))

        constraints = []

        # Constraint 1: ECBF (Eq 18)
        # psi_1_dot + alpha2 * psi_1 >= 0
        # Expanding psi_1_dot = h_ddot + alpha1 * h_dot:
        #   (-F*C + F^2*MA + Reff*F*MA - Reff^2*MF)
        #   + alpha1 * (-F*MA + Reff*MF)
        #   + alpha2 * psi_1
        #   >= 0
        #
        # Rearranging: -F*C + [everything else] >= 0
        # => -F*C >= -[everything else]
        #
        # In CVXPY form: -F * C_var + state_and_gain_terms >= 0

        state_terms = F**2 * MA + R_eff * F * MA - R_eff**2 * MF
        a1_h_dot = self._alpha1 * (-F * MA + R_eff * MF)
        a2_psi1 = self._alpha2 * psi1_val

        constraints.append(
            -F * C_var + state_terms + a1_h_dot + a2_psi1 >= 0
        )

        # Constraint 2: Resting-floor CBF (Eq 23)
        # h2_dot >= -alpha3 * h2
        # Reff*MF - C >= -alpha3 * MR
        # => C <= Reff*MF + alpha3*MR
        constraints.append(
            C_var <= R_eff * MF + self._alpha3 * MR
        )

        # Constraint 3: Non-negative neural drive
        constraints.append(C_var >= 0)

        # ----- Solve -----
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=solver, warm_start=True)
        except cp.SolverError:
            # Fallback: use analytical bounds directly
            C_safe = max(0.0, min(C_nominal, C_ub_ecbf, C_ub_cbf))
            diag = ECBFDiagnostics(
                C_nominal=C_nominal,
                C_filtered=C_safe,
                h=h_val,
                h2=h2_val,
                psi_0=psi0_val,
                psi_1=psi1_val,
                h_dot=h_dot_val,
                C_upper_ecbf=C_ub_ecbf,
                C_upper_cbf=C_ub_cbf,
                qp_status="solver_error_fallback",
                was_clipped=(abs(C_safe - C_nominal) > 1e-9),
                infeasible=(C_ub_ecbf < 0.0 and C_ub_cbf < 0.0),
            )
            return C_safe, diag

        if problem.status in ("optimal", "optimal_inaccurate"):
            C_filtered = float(C_var.value[0])
            # Final safety clamp (numerical noise)
            C_filtered = max(0.0, C_filtered)
            qp_status = problem.status
            qp_infeasible = False
        else:
            # Infeasible or unbounded: force rest (Remark 5.13)
            C_filtered = 0.0
            qp_status = problem.status
            qp_infeasible = True

        was_clipped = abs(C_filtered - C_nominal) > 1e-9

        # S-3: infeasible flag tracks mandatory rest separately from
        # was_clipped, which only compares C_nominal to C_filtered.
        # When C_nominal==0 and QP is infeasible, was_clipped=False
        # but infeasible=True — mandatory rest is still logged.
        diag = ECBFDiagnostics(
            C_nominal=C_nominal,
            C_filtered=C_filtered,
            h=h_val,
            h2=h2_val,
            psi_0=psi0_val,
            psi_1=psi1_val,
            h_dot=h_dot_val,
            C_upper_ecbf=C_ub_ecbf,
            C_upper_cbf=C_ub_cbf,
            qp_status=qp_status,
            was_clipped=was_clipped,
            infeasible=qp_infeasible,
        )

        return C_filtered, diag

    # -----------------------------------------------------------------
    # Analytical filter (no QP, for lightweight use)
    # -----------------------------------------------------------------

    def filter_analytical(
        self,
        state: ThreeCCrState,
        C_nominal: float,
        target_load: float,
    ) -> tuple[float, bool]:
        """Compute safe neural drive using analytical bounds only.

        Applies both upper bounds (Eqs 19, 23) and the non-negativity
        constraint without solving a QP. Equivalent to the QP solution
        when C_nominal is scalar (1-D QP).

        S-2: Returns an infeasibility flag to distinguish mandatory rest
        (both upper bounds < 0, no feasible C > 0) from voluntary rest
        (C_nominal was already 0). This enables paper-quality diagnostics
        when using filter_analytical in the env hot loop for speed.

        Args:
            state: Current state.
            C_nominal: Nominal neural drive.
            target_load: TL(t).

        Returns:
            Tuple of (C_safe, infeasible). infeasible=True when both
            ECBF and CBF upper bounds are negative (no feasible C > 0).
        """
        MA, MF = state.MA, state.MF
        R_eff = self.muscle.R if target_load > 0.0 else self.muscle.Rr

        ub_ecbf = self.ecbf_upper_bound(MA, MF, R_eff)
        ub_cbf = self.cbf_upper_bound(MA, MF, R_eff)

        # S-2: Detect infeasibility — both bounds negative means
        # no positive C satisfies the safety constraints.
        infeasible = (ub_ecbf < 0.0) and (ub_cbf < 0.0)

        C_safe = min(C_nominal, ub_ecbf, ub_cbf)
        C_safe = max(0.0, C_safe)
        return C_safe, infeasible

    # -----------------------------------------------------------------
    # Rest-phase analysis (Section 5.4)
    # -----------------------------------------------------------------

    def rest_phase_safe(self, state: ThreeCCrState) -> bool:
        """Check if rest-phase safety holds (Theorem 5.7).

        Under Assumption 5.5 (theta_max >= F/(F+Rr)), if h(x) >= 0
        and h2(x) >= 0, then the rest-phase dynamics preserve h >= 0.

        Returns:
            True if the state is in the safe set for autonomous rest.
        """
        return self.h(state.MF) >= -1e-9 and self.h2(state.MR) >= -1e-9

    def psi1_jump_at_rest(self, MF: float) -> float:
        """Positive jump in psi_1 at work-to-rest transition (Eq 28).

        psi_1^rest - psi_1^work = R*(r-1)*MF > 0
        """
        return self.muscle.R * (self.muscle.r - 1) * MF

    def min_rest_duration_bound(self, MA_at_stop: float) -> float:
        """Conservative upper bound on minimum rest duration (Eq 30).

        Delta_t_bar = (1/F) * ln(F*MA(ts) / (min(R, alpha1)*Theta_max))

        Returns 0 if the expression is non-positive (no rest needed).
        """
        F = self._F
        R = self.muscle.R
        a1 = self._alpha1
        tm = self._theta_max

        beta_min = min(R, a1) * tm
        if beta_min <= 0.0 or F * MA_at_stop <= beta_min:
            return 0.0

        return (1.0 / F) * np.log(F * MA_at_stop / beta_min)
