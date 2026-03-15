"""Thin wrapper: re-exports SingleWorkerWarehouseEnv as WarehouseEnv for hcmarl.envs."""
from hcmarl.warehouse_env import SingleWorkerWarehouseEnv as WarehouseEnv

__all__ = ["WarehouseEnv"]
