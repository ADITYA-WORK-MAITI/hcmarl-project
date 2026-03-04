"""Unit tests for hcmarl.three_cc_r.

Verifies all equations from Framework v12, Sections 3.1--3.5 and 7.2.
"""

import math

import numpy as np
import pytest

from hcmarl.three_cc_r import (
    ALL_MUSCLES,
    ANKLE,
    ELBOW,
    GRIP,
    KNEE,
    MUSCLE_REGISTRY,
    SHOULDER,
    TRUNK,
    MuscleParams,
    ThreeCCr,
    ThreeCCrState,
    get_muscle,
)


# =====================================================================
# 1. MuscleParams derived quantities
# =====================================================================

class TestMuscleParams:
    """Test calibrated parameters and derived quantities (Table 1)."""

    def test_shoulder_raw_values(self):
        """Verify shoulder F, R, r from Table 1."""
        assert SHOULDER.F == 0.0146
        assert SHOULDER.R == 0.00058
        assert SHOULDER.r == 15

    def test_ankle_raw_values(self):
        assert ANKLE.F == 0.00589
        assert ANKLE.R == 0.0182
        assert ANKLE.r == 15

    def test_knee_raw_values(self):
        assert KNEE.F == 0.0150
        assert KNEE.R == 0.00175
        assert KNEE.r == 15

    def test_elbow_raw_values(self):
        assert ELBOW.F == 0.00912
        assert ELBOW.R == 0.00094
        assert ELBOW.r == 15

    def test_trunk_raw_values(self):
        assert TRUNK.F == 0.00657
        assert TRUNK.R == 0.00354
        assert TRUNK.r == 15

    def test_grip_raw_values(self):
        assert GRIP.F == 0.00794
        assert GRIP.R == 0.00109
        assert GRIP.r == 30  # Grip uses r=30, not 15

    def test_shoulder_delta_max(self):
        """Verify delta_max = R/(F+R) = 3.8% for shoulder (Table 1)."""
        expected = 0.00058 / (0.0146 + 0.00058)
        assert abs(SHOULDER.delta_max - expected) < 1e-10
        assert abs(SHOULDER.delta_max - 0.0382) < 0.001

    def test_ankle_delta_max(self):
        """Verify delta_max = 75.5% for ankle (Table 1)."""
        expected = 0.0182 / (0.00589 + 0.0182)
        assert abs(ANKLE.delta_max - expected) < 1e-10
        assert abs(ANKLE.delta_max - 0.755) < 0.001

    def test_shoulder_C_max(self):
        """Verify C_max = F*R/(F+R) for shoulder (Eq 6)."""
        expected = 0.0146 * 0.00058 / (0.0146 + 0.00058)
        assert abs(SHOULDER.C_max - expected) < 1e-12

    def test_C_max_dimensional_check(self):
        """C_max = F*R/(F+R) has units [min^-1] (Remark 3.5)."""
        # Verify numerically that C_max = delta_max * F
        for m in ALL_MUSCLES:
            assert abs(m.C_max - m.delta_max * m.F) < 1e-12

    def test_shoulder_theta_min_max(self):
        """Verify theta_min_max = F/(F+Rr) = 62.7% for shoulder (Eq 25)."""
        expected = 0.0146 / (0.0146 + 0.00058 * 15)
        assert abs(SHOULDER.theta_min_max - expected) < 1e-10
        assert abs(SHOULDER.theta_min_max - 0.627) < 0.001

    def test_ankle_theta_min_max(self):
        """Verify theta_min_max = 2.1% for ankle (Table 2)."""
        expected = 0.00589 / (0.00589 + 0.0182 * 15)
        assert abs(ANKLE.theta_min_max - expected) < 1e-10
        assert abs(ANKLE.theta_min_max - 0.021) < 0.001

    def test_grip_theta_min_max(self):
        """Verify theta_min_max = 19.5% for grip (Table 2, r=30)."""
        expected = 0.00794 / (0.00794 + 0.00109 * 30)
        assert abs(GRIP.theta_min_max - expected) < 1e-10
        assert abs(GRIP.theta_min_max - 0.195) < 0.002

    def test_shoulder_Rr_over_F(self):
        """Shoulder Rr/F = 0.596 < 1 => overshoot possible (Table 2)."""
        expected = (0.00058 * 15) / 0.0146
        assert abs(SHOULDER.Rr_over_F - expected) < 1e-10
        assert SHOULDER.Rr_over_F < 1.0  # Overshoot possible

    def test_ankle_Rr_over_F(self):
        """Ankle Rr/F = 46.35 >> 1 => no overshoot (Table 2)."""
        assert ANKLE.Rr_over_F > 1.0

    def test_fatigue_resistance_ranking(self):
        """Verify ranking: ankle > trunk > grip > knee > elbow > shoulder.

        Source: Frey-Law & Avin (2010), Rohmert (1960).
        """
        deltas = {m.name: m.delta_max for m in ALL_MUSCLES}
        assert deltas["ankle"] > deltas["trunk"]
        assert deltas["trunk"] > deltas["grip"]
        assert deltas["grip"] > deltas["knee"]
        assert deltas["knee"] > deltas["elbow"]
        assert deltas["elbow"] > deltas["shoulder"]

    def test_six_muscles_registered(self):
        assert len(ALL_MUSCLES) == 6
        assert len(MUSCLE_REGISTRY) == 6

    def test_get_muscle_valid(self):
        m = get_muscle("shoulder")
        assert m is SHOULDER

    def test_get_muscle_case_insensitive(self):
        m = get_muscle("SHOULDER")
        assert m is SHOULDER

    def test_get_muscle_invalid(self):
        with pytest.raises(KeyError):
            get_muscle("bicep")


# =====================================================================
# 2. ThreeCCrState
# =====================================================================

class TestThreeCCrState:
    """Test state vector construction and validation."""

    def test_fresh_state(self):
        s = ThreeCCrState.fresh()
        assert s.MR == 1.0
        assert s.MA == 0.0
        assert s.MF == 0.0

    def test_conservation_holds(self):
        s = ThreeCCrState(MR=0.5, MA=0.3, MF=0.2)
        assert abs(s.MR + s.MA + s.MF - 1.0) < 1e-10

    def test_conservation_violated_raises(self):
        with pytest.raises(ValueError, match="Conservation violated"):
            ThreeCCrState(MR=0.5, MA=0.5, MF=0.5)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            ThreeCCrState(MR=-0.1, MA=0.6, MF=0.5)

    def test_as_array_roundtrip(self):
        s = ThreeCCrState(MR=0.6, MA=0.1, MF=0.3)
        arr = s.as_array()
        s2 = ThreeCCrState.from_array(arr)
        assert abs(s2.MR - 0.6) < 1e-10
        assert abs(s2.MA - 0.1) < 1e-10
        assert abs(s2.MF - 0.3) < 1e-10


# =====================================================================
# 3. ThreeCCr ODE system
# =====================================================================

class TestThreeCCrODE:
    """Test the ODE right-hand side and related methods."""

    def setup_method(self):
        self.model = ThreeCCr(params=SHOULDER, kp=10.0)

    # -- Reperfusion switch (Eq 5) --

    def test_R_eff_work_phase(self):
        """During work (TL > 0): Reff = R (Eq 5)."""
        assert self.model.R_eff(0.3) == SHOULDER.R

    def test_R_eff_rest_phase(self):
        """During rest (TL = 0): Reff = R*r (Eq 5)."""
        assert self.model.R_eff(0.0) == SHOULDER.Rr

    # -- Baseline neural drive controller (Eq 35) --

    def test_baseline_drive_rest(self):
        """C = 0 when resting (Eq 35)."""
        assert self.model.baseline_neural_drive(0.0, 0.5) == 0.0

    def test_baseline_drive_tracking(self):
        """C = kp * (TL - MA) when TL > MA (Eq 35)."""
        C = self.model.baseline_neural_drive(0.3, 0.1)
        assert abs(C - 10.0 * (0.3 - 0.1)) < 1e-10

    def test_baseline_drive_non_negative(self):
        """C = kp * max(TL - MA, 0) -- non-negative (Eq 35)."""
        C = self.model.baseline_neural_drive(0.1, 0.3)
        assert C == 0.0

    # -- ODE RHS structure (Eqs 2--4) --

    def test_ode_rhs_conservation(self):
        """Sum of derivatives = 0 (Theorem 3.2).

        dMR/dt + dMA/dt + dMF/dt = (Reff*MF - C) + (C - F*MA) + (F*MA - Reff*MF) = 0
        """
        x = np.array([0.5, 0.3, 0.2])
        for C_val in [0.0, 0.01, 0.05]:
            for TL in [0.0, 0.3]:
                dx = self.model.ode_rhs(x, C_val, TL)
                assert abs(dx.sum()) < 1e-12, (
                    f"Conservation violated: sum(dx) = {dx.sum()}"
                )

    def test_ode_rhs_fresh_no_drive(self):
        """Fresh state with no drive: all derivatives = 0."""
        x = np.array([1.0, 0.0, 0.0])
        dx = self.model.ode_rhs(x, C=0.0, target_load=0.0)
        np.testing.assert_allclose(dx, [0.0, 0.0, 0.0], atol=1e-15)

    def test_ode_rhs_dMA_formula(self):
        """dMA/dt = C - F*MA (Eq 2)."""
        x = np.array([0.5, 0.3, 0.2])
        C = 0.02
        dx = self.model.ode_rhs(x, C, target_load=0.3)
        expected_dMA = C - SHOULDER.F * 0.3
        assert abs(dx[1] - expected_dMA) < 1e-12

    def test_ode_rhs_dMF_formula_work(self):
        """dMF/dt = F*MA - R*MF during work (Eq 3, Reff=R)."""
        x = np.array([0.5, 0.3, 0.2])
        dx = self.model.ode_rhs(x, C=0.02, target_load=0.3)
        expected_dMF = SHOULDER.F * 0.3 - SHOULDER.R * 0.2
        assert abs(dx[2] - expected_dMF) < 1e-12

    def test_ode_rhs_dMF_formula_rest(self):
        """dMF/dt = F*MA - R*r*MF during rest (Eq 3, Reff=Rr)."""
        x = np.array([0.5, 0.3, 0.2])
        dx = self.model.ode_rhs(x, C=0.0, target_load=0.0)
        expected_dMF = SHOULDER.F * 0.3 - SHOULDER.Rr * 0.2
        assert abs(dx[2] - expected_dMF) < 1e-12

    def test_ode_rhs_dMR_formula(self):
        """dMR/dt = Reff*MF - C (Eq 4)."""
        x = np.array([0.5, 0.3, 0.2])
        C = 0.02
        dx = self.model.ode_rhs(x, C, target_load=0.3)
        expected_dMR = SHOULDER.R * 0.2 - C
        assert abs(dx[0] - expected_dMR) < 1e-12

    # -- Steady-state verification (Theorem 3.4) --

    def test_steady_state_conservation(self):
        """Steady state at sustainability limit: MR = 0 (Theorem 3.4)."""
        ss = self.model.steady_state_work()
        assert abs(ss.MR) < 1e-10
        assert abs(ss.MR + ss.MA + ss.MF - 1.0) < 1e-10

    def test_steady_state_derivatives_zero(self):
        """At steady state with C=C_max: all derivatives = 0."""
        ss = self.model.steady_state_work()
        x = ss.as_array()
        C = SHOULDER.C_max
        dx = self.model.ode_rhs(x, C, target_load=0.5)
        np.testing.assert_allclose(dx, [0.0, 0.0, 0.0], atol=1e-10)

    def test_steady_state_MA_equals_delta_max(self):
        """MA at steady state = delta_max = R/(F+R) (Eq 7)."""
        ss = self.model.steady_state_work()
        assert abs(ss.MA - SHOULDER.delta_max) < 1e-10

    def test_steady_state_MF(self):
        """MF at steady state = C_max/R = F/(F+R) (Eq 8)."""
        ss = self.model.steady_state_work()
        expected_MF = SHOULDER.C_max / SHOULDER.R
        assert abs(ss.MF - expected_MF) < 1e-10


# =====================================================================
# 4. Euler step
# =====================================================================

class TestEulerStep:
    """Test single-step integration."""

    def test_euler_conservation(self):
        """Euler step preserves conservation law."""
        model = ThreeCCr(params=SHOULDER)
        s = ThreeCCrState(MR=0.7, MA=0.2, MF=0.1)
        s_new = model.step_euler(s, C=0.01, target_load=0.2, dt=1.0)
        assert abs(s_new.MR + s_new.MA + s_new.MF - 1.0) < 1e-10

    def test_euler_rest_recovery(self):
        """During rest, MF should decrease (for Rr/F > 1 muscles)."""
        model = ThreeCCr(params=ANKLE)  # Rr/F = 46.35
        s = ThreeCCrState(MR=0.5, MA=0.0, MF=0.5)
        s_new = model.step_euler(s, C=0.0, target_load=0.0, dt=1.0)
        assert s_new.MF < s.MF


# =====================================================================
# 5. Full simulation (solve_ivp)
# =====================================================================

class TestSimulation:
    """Test ODE integration via scipy."""

    def test_simulation_conservation_throughout(self):
        """MR + MA + MF = 1 at every evaluation point."""
        model = ThreeCCr(params=SHOULDER)
        result = model.simulate(
            state0=ThreeCCrState.fresh(),
            target_load=0.3,
            duration=10.0,
            dt_eval=0.5,
        )
        totals = result["MR"] + result["MA"] + result["MF"]
        np.testing.assert_allclose(totals, 1.0, atol=1e-6)

    def test_simulation_non_negative(self):
        """All compartments stay non-negative."""
        model = ThreeCCr(params=SHOULDER)
        result = model.simulate(
            state0=ThreeCCrState.fresh(),
            target_load=0.3,
            duration=10.0,
        )
        assert np.all(result["MR"] >= -1e-9)
        assert np.all(result["MA"] >= -1e-9)
        assert np.all(result["MF"] >= -1e-9)

    def test_simulation_fatigue_increases_under_load(self):
        """Under constant load, MF increases from zero."""
        model = ThreeCCr(params=SHOULDER)
        result = model.simulate(
            state0=ThreeCCrState.fresh(),
            target_load=0.3,
            duration=30.0,
        )
        assert result["MF"][-1] > result["MF"][0]

    def test_rest_recovery(self):
        """After fatiguing, rest causes MF to decrease."""
        model = ThreeCCr(params=ANKLE)
        fatigued = ThreeCCrState(MR=0.3, MA=0.0, MF=0.7)
        result = model.simulate(
            state0=fatigued,
            target_load=0.0,
            duration=30.0,
        )
        assert result["MF"][-1] < result["MF"][0]
