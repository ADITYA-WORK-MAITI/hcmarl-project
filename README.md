# HC-MARL: Human-Centric Multi-Agent Reinforcement Learning

**Human-Centric Multi-Agent Control for Safe and Fair Human-Robot Collaboration in Warehouse Environments**

Aditya Maiti | University School of Automation and Robotics, GGSIPU  
Supervised by Dr. Amrit Pal Singh

---

## Overview

HC-MARL integrates four modules into a unified framework for safe, fair, and
physiologically-aware human-robot task allocation:

1. **3CC-r Physiological Model** -- Three-Compartment Controller with
   reperfusion (Liu et al. 2002; Xia & Frey-Law 2008; Looft et al. 2018).
   Models muscle fatigue via ODE dynamics across Resting, Active, and Fatigued
   motor-unit compartments.

2. **ECBF Safety Filter** -- Exponential Control Barrier Functions (Nguyen &
   Sreenath 2016) enforce a fatigue ceiling (relative degree 2) and a resting
   floor (relative degree 1) via a dual-barrier QP solved in real time.

3. **MMICRL Constraint Learner** -- Multi-Modal Inverse Constrained RL (Qiao
   et al. 2023) infers personalised safety constraints from heterogeneous
   worker demonstrations using a weighted information-theoretic objective.

4. **NSWF Task Allocator** -- Nash Social Welfare Function (Kaneko & Nakamura
   1979) maximises the product of worker surpluses with a divergent
   disagreement utility that makes burnout assignments infinitely costly.

## Project Structure

```
hcmarl/
    __init__.py          # Package init, version
    three_cc_r.py        # 3CC-r ODE fatigue model
    ecbf_filter.py       # ECBF dual-barrier safety filter
    nswf_allocator.py    # Nash Social Welfare task allocator
    pipeline.py          # End-to-end orchestration
    utils.py             # Shared utilities
    envs/                # Environment wrappers (Phase 2)
    agents/              # RL agent implementations (Phase 3)
tests/
    __init__.py
    test_three_cc_r.py   # Unit tests for 3CC-r
    test_ecbf.py         # Unit tests for ECBF
    test_nswf.py         # Unit tests for NSWF
    test_pipeline.py     # Integration tests for pipeline
config/
    muscle_params.yaml   # Calibrated muscle parameters (Table 1)
    default_config.yaml  # Default experiment configuration
scripts/               # Training and evaluation scripts (Phase 4)
notebooks/             # Jupyter notebooks (Phase 4)
```

## Installation

```bash
pip install -e .
```

## Requirements

- Python >= 3.9
- NumPy, SciPy (ODE integration)
- CVXPY with OSQP (QP solver for ECBF)
- PyYAML (configuration)
- PyTorch >= 2.0 (RL training, Phase 3+)
- pytest (testing)

## Quick Start

```python
from hcmarl.three_cc_r import ThreeCCr, MuscleParams
from hcmarl.ecbf_filter import ECBFFilter
from hcmarl.nswf_allocator import NSWFAllocator
from hcmarl.pipeline import HCMARLPipeline

# Load default configuration
pipeline = HCMARLPipeline.from_config("config/default_config.yaml")

# Run one allocation round
pipeline.step()
```

## Testing

```bash
pytest tests/ -v
```

## License

Research code. Not for redistribution without permission.
