"""HC-MARL Baselines package — OmniSafe and SafePO wrappers + legacy baselines."""
from hcmarl.baselines.omnisafe_wrapper import OmniSafeWrapper
from hcmarl.baselines.safepo_wrapper import SafePOWrapper

# Re-export Phase 2 legacy baselines so existing imports keep working
from hcmarl.baselines._legacy import (
    RandomBaseline, RoundRobinBaseline, GreedyBaseline, GreedySafeBaseline,
    PPOUnconstrainedBaseline, PPOLagrangianBaseline, MAPPONoFilterBaseline,
    MAPPOLagrangianBaseline, MACPOBaseline, FixedScheduleBaseline,
    create_all_baselines,
)
