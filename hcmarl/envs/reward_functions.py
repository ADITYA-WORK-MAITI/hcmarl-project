"""HC-MARL Phase 2 (#32): Reward shaping functions."""
import numpy as np
from typing import Dict

class ProductivityReward:
    """Reward = sum of task demands (higher load = more work done)."""
    def __call__(self, task_demands: Dict[str, float], **kwargs) -> float:
        return sum(task_demands.values())

class SafetyPenalty:
    """Penalty for each muscle exceeding theta_max."""
    def __init__(self, penalty_weight=10.0):
        self.w = penalty_weight
    def __call__(self, fatigue: Dict[str, float], theta_max: Dict[str, float], **kwargs) -> float:
        return -self.w * sum(max(0, fatigue[m] - theta_max[m]) for m in fatigue)

class FatigueCost:
    """Running cost proportional to average fatigue."""
    def __init__(self, cost_weight=0.5):
        self.w = cost_weight
    def __call__(self, fatigue: Dict[str, float], **kwargs) -> float:
        return -self.w * np.mean(list(fatigue.values()))

class CompositeReward:
    """Combines productivity, fatigue cost, and safety penalty."""
    def __init__(self, prod_weight=1.0, fatigue_weight=0.5, safety_weight=10.0):
        self.prod = ProductivityReward()
        self.fatigue = FatigueCost(fatigue_weight)
        self.safety = SafetyPenalty(safety_weight)
        self.prod_weight = prod_weight
    def __call__(self, task_demands, fatigue, theta_max, **kwargs) -> float:
        return (self.prod_weight * self.prod(task_demands) +
                self.fatigue(fatigue) + self.safety(fatigue, theta_max))
