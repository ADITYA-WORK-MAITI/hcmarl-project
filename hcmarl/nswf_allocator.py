"""
NSWF Task Allocator: Nash Social Welfare Function.

Implements the cooperative task allocation from:
    - Nash (1950): The Bargaining Problem
    - Nash (1953): Two-Person Cooperative Games
    - Kaneko & Nakamura (1979): Nash Social Welfare Function for N players
    - Binmore, Shaked & Sutton (1989): Outside Option principle
    - Navon et al. (2022): Log-transform for gradient-based computation

Mathematical reference: HC-MARL Framework v12, Section 6.
All equation numbers below refer to that document.

The allocator solves:
    max_{a_ij} sum_i ln(U(i, j*(i)) - D_i(MF_i))
subject to assignment constraints (Def 6.1).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from hcmarl.utils import safe_log


@dataclasses.dataclass
class NSWFParams:
    """Parameters for the NSWF allocator.

    Attributes:
        kappa: Positive scaling constant for disagreement utility (Eq 32).
        epsilon: Small positive constant for rest-task surplus (Eq 31).
    """

    kappa: float = 1.0
    epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")


@dataclasses.dataclass
class AllocationResult:
    """Result of the NSWF task allocation.

    Attributes:
        assignments: Dict mapping worker index i -> task index j.
            j = 0 means rest (task 0).
        objective_value: Value of the NSWF objective (Eq 33).
        surpluses: Dict mapping worker i -> U(i, j*(i)) - D_i.
        disagreement_utilities: Dict mapping worker i -> D_i(MF_i).
    """

    assignments: dict[int, int]
    objective_value: float
    surpluses: dict[int, float]
    disagreement_utilities: dict[int, float]


class NSWFAllocator:
    """Nash Social Welfare task allocator.

    Solves the centralised allocation problem (Eq 33):
        max_{a_ij} sum_i ln(U(i, j*(i)) - D_i(MF_i))

    Subject to (Def 6.1):
        (i)   Each worker receives exactly one assignment.
        (ii)  Each productive task assigned to at most one worker.
        (iii) Multiple workers may rest simultaneously (task 0).

    Implementation uses brute-force enumeration for small N, M (sufficient
    for warehouse scenarios with N <= 12 workers and M <= 12 tasks).
    For larger instances, the log-NSWF can be solved via the Hungarian
    algorithm on transformed costs.

    Args:
        params: NSWF configuration parameters.
    """

    def __init__(self, params: Optional[NSWFParams] = None) -> None:
        self.params = params or NSWFParams()

    # -----------------------------------------------------------------
    # Disagreement utility (Eq 32)
    # -----------------------------------------------------------------

    def disagreement_utility(self, MF: float) -> float:
        """Divergent disagreement utility (Eq 32).

        D_i(MF) = kappa * MF^2 / (1 - MF)

        Properties (Proposition 6.3):
            (P1) D_i(0) = 0                    -- no premium when fresh
            (P2) D_i'(MF) > 0 for MF in (0,1)  -- monotonically increasing
            (P3) D_i(MF) -> +inf as MF -> 1     -- infinite cost at burnout

        Low-fatigue regime (Remark 6.4):
            For MF << 1: D_i ~ kappa * MF^2  (smooth quadratic)

        Args:
            MF: Fatigued fraction for worker i, in [0, 1).

        Returns:
            Disagreement utility value.

        Raises:
            ValueError: If MF is not in [0, 1).
        """
        if MF < 0.0 or MF >= 1.0:
            if MF >= 1.0:
                return float("inf")
            raise ValueError(f"MF must be in [0, 1), got {MF}")

        return self.params.kappa * (MF ** 2) / (1.0 - MF)

    def disagreement_derivative(self, MF: float) -> float:
        """Derivative of disagreement utility (proof of P2 in Prop 6.3).

        D_i'(MF) = kappa * MF * (2 - MF) / (1 - MF)^2

        Args:
            MF: Fatigued fraction in [0, 1).

        Returns:
            D_i'(MF), always positive for MF in (0, 1).
        """
        if MF <= 0.0:
            return 0.0
        if MF >= 1.0:
            return float("inf")
        return self.params.kappa * MF * (2.0 - MF) / (1.0 - MF) ** 2

    # -----------------------------------------------------------------
    # Rest-task utility (Eq 31)
    # -----------------------------------------------------------------

    def rest_utility(self, MF: float) -> float:
        """Utility of the rest option for worker i (Eq 31).

        U(i, 0) = D_i(MF_i) + epsilon

        This ensures the surplus of resting is exactly epsilon > 0,
        keeping the log argument well-defined.

        Args:
            MF: Fatigued fraction.

        Returns:
            Rest utility.
        """
        return self.disagreement_utility(MF) + self.params.epsilon

    # -----------------------------------------------------------------
    # Surplus computation
    # -----------------------------------------------------------------

    def surplus(self, utility: float, MF: float) -> float:
        """Compute worker surplus: U(i, j) - D_i(MF_i).

        For the NSWF objective (Eq 33), this must be positive for
        every assignment.

        Args:
            utility: U(i, j) -- productivity utility of worker i doing task j.
            MF: Worker i's fatigue level.

        Returns:
            Surplus value. Positive means the assignment is individually
            rational. Negative means the worker prefers rest.
        """
        return utility - self.disagreement_utility(MF)

    # -----------------------------------------------------------------
    # Allocation solver (Eq 33)
    # -----------------------------------------------------------------

    def allocate(
        self,
        utility_matrix: np.ndarray,
        fatigue_levels: np.ndarray,
    ) -> AllocationResult:
        """Solve the NSWF allocation problem (Eq 33).

        Args:
            utility_matrix: Array of shape (N, M) where entry [i, j] is
                U(i, j+1), the productivity utility of worker i on
                productive task j+1. Tasks are 1-indexed in the theory
                but 0-indexed in this array.
            fatigue_levels: Array of shape (N,) with MF_i for each worker.

        Returns:
            AllocationResult with optimal assignments and diagnostics.

        Note:
            Task index 0 in the result means REST.
            Task index j >= 1 corresponds to utility_matrix[:, j-1].
        """
        N = len(fatigue_levels)
        M = utility_matrix.shape[1] if utility_matrix.ndim == 2 else 0

        if utility_matrix.shape[0] != N:
            raise ValueError(
                f"utility_matrix has {utility_matrix.shape[0]} rows but "
                f"there are {N} workers."
            )

        # Compute disagreement utilities
        D = np.array([self.disagreement_utility(mf) for mf in fatigue_levels])

        # Compute surplus matrix for productive tasks
        # surplus_matrix[i, j] = U(i, j+1) - D_i
        surplus_matrix = utility_matrix - D[:, None]

        # Add rest option: surplus = epsilon for all workers
        # (from Eq 31: U(i,0) - D_i = epsilon)
        eps = self.params.epsilon

        # Use greedy assignment for small problems
        # (brute-force permutation for optimal, but greedy is O(NM))
        if N <= 8 and M <= 8:
            return self._solve_exact(N, M, surplus_matrix, D, eps)
        else:
            return self._solve_greedy(N, M, surplus_matrix, D, eps)

    def _solve_exact(
        self,
        N: int,
        M: int,
        surplus_matrix: np.ndarray,
        D: np.ndarray,
        eps: float,
    ) -> AllocationResult:
        """Exact solver via enumeration of valid assignments.

        For each worker, they can be assigned to one of M productive tasks
        or to rest (task 0). No two workers share a productive task.
        """
        best_obj = -float("inf")
        best_assign = {i: 0 for i in range(N)}  # Default: all rest

        def _search(
            worker: int,
            used_tasks: set[int],
            assignment: dict[int, int],
        ) -> None:
            nonlocal best_obj, best_assign

            if worker == N:
                # Compute NSWF objective (Eq 33)
                obj = 0.0
                for i in range(N):
                    j = assignment[i]
                    if j == 0:
                        s = eps
                    else:
                        s = surplus_matrix[i, j - 1]
                    if s <= 0:
                        obj = -float("inf")
                        break
                    obj += safe_log(s)

                if obj > best_obj:
                    best_obj = obj
                    best_assign = dict(assignment)
                return

            # Rest option (task 0) -- always available
            assignment[worker] = 0
            _search(worker + 1, used_tasks, assignment)

            # Productive tasks (1-indexed)
            for j_idx in range(M):
                if j_idx not in used_tasks:
                    s = surplus_matrix[worker, j_idx]
                    if s > 0:
                        assignment[worker] = j_idx + 1  # 1-indexed
                        used_tasks.add(j_idx)
                        _search(worker + 1, used_tasks, assignment)
                        used_tasks.remove(j_idx)

            assignment[worker] = 0  # Reset

        _search(0, set(), {})

        # Build result
        surpluses = {}
        for i in range(N):
            j = best_assign[i]
            if j == 0:
                surpluses[i] = eps
            else:
                surpluses[i] = float(surplus_matrix[i, j - 1])

        return AllocationResult(
            assignments=best_assign,
            objective_value=best_obj,
            surpluses=surpluses,
            disagreement_utilities={i: float(D[i]) for i in range(N)},
        )

    def _solve_greedy(
        self,
        N: int,
        M: int,
        surplus_matrix: np.ndarray,
        D: np.ndarray,
        eps: float,
    ) -> AllocationResult:
        """Greedy approximation for larger instances.

        Assigns workers to tasks in order of maximum log-surplus gain,
        respecting the one-task-per-worker and one-worker-per-task constraints.
        """
        assignments: dict[int, int] = {}
        used_tasks: set[int] = set()
        surpluses: dict[int, float] = {}

        # Build candidate list: (log_surplus_gain, worker, task_1indexed)
        candidates = []
        for i in range(N):
            for j_idx in range(M):
                s = surplus_matrix[i, j_idx]
                if s > 0:
                    gain = safe_log(s) - safe_log(eps)
                    candidates.append((gain, i, j_idx + 1))

        # Sort by gain descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        assigned_workers: set[int] = set()

        for gain, i, j in candidates:
            if i in assigned_workers or (j - 1) in used_tasks:
                continue
            assignments[i] = j
            surpluses[i] = float(surplus_matrix[i, j - 1])
            assigned_workers.add(i)
            used_tasks.add(j - 1)

        # Unassigned workers get rest
        for i in range(N):
            if i not in assignments:
                assignments[i] = 0
                surpluses[i] = eps

        # Compute objective
        obj = sum(safe_log(s) for s in surpluses.values())

        return AllocationResult(
            assignments=assignments,
            objective_value=obj,
            surpluses=surpluses,
            disagreement_utilities={i: float(D[i]) for i in range(N)},
        )
