"""Thin wrapper: re-exports SingleWorkerWarehouseEnv as WarehouseEnv for hcmarl.envs."""


def __getattr__(name):
    """Lazy import to break circular dependency."""
    if name == "WarehouseEnv":
        from hcmarl.warehouse_env import SingleWorkerWarehouseEnv
        return SingleWorkerWarehouseEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["WarehouseEnv"]
