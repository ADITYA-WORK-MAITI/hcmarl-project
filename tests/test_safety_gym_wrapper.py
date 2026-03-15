"""Tests for Phase 2: Safety-Gym ECBF wrapper (#26)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hcmarl.envs.safety_gym_ecbf import ECBFSafetyWrapper

def test_wrapper_init():
    class DummyEnv:
        def reset(self): return np.zeros(4), {}
        def step(self, a): return np.zeros(4), 0.0, False, {}
    w = ECBFSafetyWrapper(DummyEnv())
    assert w.safe_distance == 0.5; print("  PASS: test_wrapper_init")

def test_filter_preserves_shape():
    class DummyEnv:
        def reset(self): return np.zeros(4), {}
        def step(self, a): return np.zeros(4), 0.0, False, {}
    w = ECBFSafetyWrapper(DummyEnv())
    action = np.array([0.5, -0.3])
    filtered = w._filter_action(action, np.zeros(4))
    assert filtered.shape == action.shape; print("  PASS: test_filter_preserves_shape")

if __name__ == "__main__":
    print("=== Safety-Gym Wrapper Tests ===")
    for t in [test_wrapper_init, test_filter_preserves_shape]: t()
    print("All passed.")
