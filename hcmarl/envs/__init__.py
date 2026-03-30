"""HC-MARL Environments package."""
from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
from hcmarl.envs.task_profiles import TaskProfileManager
from hcmarl.envs.reward_functions import nswf_reward, safety_cost, disagreement_utility
from hcmarl.envs.rware_wrapper import RWAREWrapper
from hcmarl.envs.safety_gym_real import SafetyGymECBFWrapper


def __getattr__(name):
    """Lazy import for WarehouseEnv to break circular dependency."""
    if name == "WarehouseEnv":
        from hcmarl.warehouse_env import SingleWorkerWarehouseEnv
        return SingleWorkerWarehouseEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WarehouseEnv", "WarehousePettingZoo", "TaskProfileManager",
    "nswf_reward", "safety_cost", "disagreement_utility",
    "RWAREWrapper", "SafetyGymECBFWrapper",
]
