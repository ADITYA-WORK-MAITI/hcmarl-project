"""Tests for Phase 2: Warehouse environment, MAPPO agent, baselines."""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcmarl.warehouse_env import SingleWorkerWarehouseEnv, WarehouseMultiAgentEnv
from hcmarl.mappo_agent import MAPPOAgent, MAPPOActorNumpy, MAPPOCriticNumpy, MAPPOBuffer
from hcmarl.baselines import create_all_baselines


# ===========================================================================
# Warehouse Environment Tests
# ===========================================================================

def test_single_env_reset():
    env = SingleWorkerWarehouseEnv()
    obs, info = env.reset()
    assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
    assert np.allclose(obs[:3], [1.0, 0.0, 0.0]), "Initial state should be [MR=1, MA=0, MF=0]"
    print("  PASS: test_single_env_reset")

def test_single_env_step():
    env = SingleWorkerWarehouseEnv()
    obs, _ = env.reset()
    obs2, reward, term, trunc, info = env.step(0)  # heavy_lift
    assert obs2.shape == (10,)
    assert isinstance(reward, float)
    assert "task" in info
    print("  PASS: test_single_env_step")

def test_single_env_conservation():
    env = SingleWorkerWarehouseEnv()
    env.reset()
    for _ in range(30):
        env.step(env.action_space.sample())
    for m in env.muscle_names:
        total = env.state[m]["MR"] + env.state[m]["MA"] + env.state[m]["MF"]
        assert abs(total - 1.0) < 1e-6, f"Conservation violated: {total}"
    print("  PASS: test_single_env_conservation")

def test_single_env_rest_recovers():
    env = SingleWorkerWarehouseEnv(max_steps=500)
    env.reset()
    # Work for 20 steps
    for _ in range(20):
        env.step(0)
    # Rest for 300 steps and track MF trajectory
    # MF continues to RISE initially during rest because MA (still high from
    # work) feeds MF via F*MA. Only after MA decays does MF start to drop.
    # This is correct 3CC-r physics — the "pipeline effect".
    rest_idx = env.task_names.index("rest")
    mf_history = []
    for _ in range(300):
        env.step(rest_idx)
        mf_history.append(env.state["grip"]["MF"])
    # MF should peak then decline — verify it's lower at end than at peak
    peak_mf = max(mf_history)
    final_mf = mf_history[-1]
    assert final_mf < peak_mf, f"MF should recover after peak: final={final_mf:.6f} >= peak={peak_mf:.6f}"
    print("  PASS: test_single_env_rest_recovers")

def test_multi_env_reset():
    env = WarehouseMultiAgentEnv(n_workers=4)
    obs, infos = env.reset()
    assert len(obs) == 4
    assert all(obs[a].shape == (10,) for a in obs)
    print("  PASS: test_multi_env_reset")

def test_multi_env_step():
    env = WarehouseMultiAgentEnv(n_workers=4)
    obs, _ = env.reset()
    actions = {agent: 0 for agent in env.agents}
    obs2, rewards, terms, truncs, infos = env.step(actions)
    assert len(obs2) == 4
    assert len(rewards) == 4
    print("  PASS: test_multi_env_step")

def test_multi_env_global_obs():
    env = WarehouseMultiAgentEnv(n_workers=4)
    env.reset()
    global_obs = env._get_global_obs()
    expected_dim = 4 * 3 * 3 + 1  # 4 workers * 3 muscles * 3 compartments + step
    assert global_obs.shape == (expected_dim,), f"Expected ({expected_dim},), got {global_obs.shape}"
    print("  PASS: test_multi_env_global_obs")

def test_multi_env_episode():
    env = WarehouseMultiAgentEnv(n_workers=3, max_steps=20)
    obs, _ = env.reset()
    for step in range(20):
        actions = {agent: np.random.randint(0, 4) for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()):
            break
    assert all(terms.values()), "Episode should terminate after max_steps"
    print("  PASS: test_multi_env_episode")


# ===========================================================================
# MAPPO Agent Tests
# ===========================================================================

def test_actor_forward():
    actor = MAPPOActorNumpy(obs_dim=10, n_actions=4)
    obs = np.random.randn(10).astype(np.float32)
    logits = actor.forward(obs)
    assert logits.shape == (4,)
    print("  PASS: test_actor_forward")

def test_actor_get_action():
    actor = MAPPOActorNumpy(obs_dim=10, n_actions=4)
    obs = np.random.randn(10).astype(np.float32)
    action, log_prob = actor.get_action(obs)
    assert 0 <= action < 4
    assert log_prob <= 0  # log probability is always <= 0
    print("  PASS: test_actor_get_action")

def test_actor_probs_sum_to_one():
    actor = MAPPOActorNumpy(obs_dim=10, n_actions=4)
    obs = np.random.randn(10).astype(np.float32)
    probs = actor.get_probs(obs)
    assert abs(probs.sum() - 1.0) < 1e-5, f"Probs sum to {probs.sum()}"
    assert all(p > 0 for p in probs)
    print("  PASS: test_actor_probs_sum_to_one")

def test_critic_forward():
    critic = MAPPOCriticNumpy(global_obs_dim=37)
    obs = np.random.randn(37).astype(np.float32)
    value = critic.forward(obs)
    assert isinstance(value, float)
    print("  PASS: test_critic_forward")

def test_mappo_agent_get_actions():
    agent = MAPPOAgent(n_agents=4, obs_dim=10, global_obs_dim=37, n_actions=4)
    observations = {f"worker_{i}": np.random.randn(10).astype(np.float32) for i in range(4)}
    global_state = np.random.randn(37).astype(np.float32)
    actions, log_probs, values = agent.get_actions(observations, global_state)
    assert len(actions) == 4
    assert log_probs.shape == (4,)
    assert values.shape == (4,)
    print("  PASS: test_mappo_agent_get_actions")

def test_buffer_store_and_gae():
    buffer = MAPPOBuffer(n_agents=2, obs_dim=10, global_obs_dim=21, buffer_size=16)
    for _ in range(10):
        buffer.store(
            obs=np.random.randn(2, 10).astype(np.float32),
            global_state=np.random.randn(21).astype(np.float32),
            actions=np.random.randint(0, 4, 2),
            rewards=np.random.randn(2).astype(np.float32),
            dones=np.zeros(2, dtype=np.float32),
            log_probs=np.random.randn(2).astype(np.float32),
            values=np.random.randn(2).astype(np.float32),
        )
    buffer.compute_gae(last_values=np.zeros(2))
    assert buffer.ptr == 10
    print("  PASS: test_buffer_store_and_gae")


# ===========================================================================
# Baseline Tests
# ===========================================================================

def test_all_baselines_return_actions():
    obs_dim = 10
    n_actions = 4
    baselines = create_all_baselines(obs_dim, n_actions, n_muscles=3)
    assert len(baselines) == 10, f"Expected 10 baselines, got {len(baselines)}"

    observations = {f"worker_{i}": np.random.randn(obs_dim).astype(np.float32) for i in range(4)}

    for baseline in baselines:
        actions = baseline.get_actions(observations)
        assert len(actions) == 4, f"{baseline.name} returned {len(actions)} actions"
        for agent, action in actions.items():
            assert 0 <= action < n_actions, f"{baseline.name}: invalid action {action}"

    print("  PASS: test_all_baselines_return_actions")

def test_baseline_names_unique():
    baselines = create_all_baselines(10, 4, 3)
    names = [b.name for b in baselines]
    assert len(names) == len(set(names)), f"Duplicate baseline names: {names}"
    print("  PASS: test_baseline_names_unique")

def test_fixed_schedule_alternates():
    from hcmarl.baselines import FixedScheduleBaseline
    baseline = FixedScheduleBaseline(n_tasks=4, work_duration=3, rest_duration=2)
    obs = {"worker_0": np.zeros(10, dtype=np.float32)}
    actions = []
    for _ in range(10):
        a = baseline.get_actions(obs)
        actions.append(a["worker_0"])
    # Should be: 0,0,0,3,3,0,0,0,3,3
    assert actions[:3] == [0, 0, 0]
    assert actions[3:5] == [3, 3]
    print("  PASS: test_fixed_schedule_alternates")

def test_greedy_safe_rests_when_fatigued():
    from hcmarl.baselines import GreedySafeBaseline
    baseline = GreedySafeBaseline(n_tasks=4, n_muscles=3, mf_threshold=0.5)
    # Create obs with high fatigue (MF > 0.5 at index 2)
    obs_high = np.array([0.2, 0.2, 0.6, 0.3, 0.3, 0.4, 0.4, 0.3, 0.3, 0.5], dtype=np.float32)
    actions = baseline.get_actions({"worker_0": obs_high})
    assert actions["worker_0"] == 3, "Should rest when MF > threshold"
    print("  PASS: test_greedy_safe_rests_when_fatigued")


# ===========================================================================
# Integration: MAPPO + Environment
# ===========================================================================

def test_mappo_runs_in_env():
    env = WarehouseMultiAgentEnv(n_workers=3, max_steps=10)
    obs_dim = 3 * env.n_muscles + 1
    global_obs_dim = 3 * env.n_muscles * 3 + 1
    agent = MAPPOAgent(n_agents=3, obs_dim=obs_dim, global_obs_dim=global_obs_dim, n_actions=4)

    obs, _ = env.reset()
    total_reward = 0.0
    for step in range(10):
        global_state = env._get_global_obs()
        actions, _, _ = agent.get_actions(obs, global_state)
        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += sum(rewards.values())
        if all(terms.values()):
            break
    assert isinstance(total_reward, float)
    print("  PASS: test_mappo_runs_in_env")


# ===========================================================================
# Run all
# ===========================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Phase 2 Tests: Warehouse Env + MAPPO + Baselines")
    print("=" * 50)

    tests = [
        # Env
        test_single_env_reset, test_single_env_step,
        test_single_env_conservation, test_single_env_rest_recovers,
        test_multi_env_reset, test_multi_env_step,
        test_multi_env_global_obs, test_multi_env_episode,
        # MAPPO
        test_actor_forward, test_actor_get_action,
        test_actor_probs_sum_to_one, test_critic_forward,
        test_mappo_agent_get_actions, test_buffer_store_and_gae,
        # Baselines
        test_all_baselines_return_actions, test_baseline_names_unique,
        test_fixed_schedule_alternates, test_greedy_safe_rests_when_fatigued,
        # Integration
        test_mappo_runs_in_env,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Phase 2: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 50}")
