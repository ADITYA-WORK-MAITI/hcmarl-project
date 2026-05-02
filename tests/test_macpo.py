"""Smoke tests for MACPO agent.

Verifies: construction, action selection contract (4-tuple incl. cost
values), buffer round-trip, update loop end-to-end, save/load, and the
mathematical machinery (CG solves H x = g, dual returns sane scalars).
"""
import os
import tempfile

import numpy as np
import torch

from hcmarl.agents.macpo import (
    MACPO, _conjugate_gradient, _solve_macpo_dual,
    _flat_params, _set_flat_params,
)


def _seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Mathematical helpers
# ----------------------------------------------------------------------

def test_cg_solves_spd_system():
    """CG must recover x from A x = b on a small SPD matrix."""
    _seed()
    n = 8
    A = torch.randn(n, n)
    A = A @ A.T + 0.5 * torch.eye(n)   # symmetric positive-definite
    x_true = torch.randn(n)
    b = A @ x_true

    matvec = lambda v: A @ v
    x_cg = _conjugate_gradient(matvec, b, n_iters=50, tol=1e-12)

    assert torch.allclose(x_cg, x_true, atol=1e-4), \
        f"CG failed to converge: max diff {(x_cg - x_true).abs().max()}"


def test_dual_pure_reward_when_already_feasible():
    """If c < 0 (cost already below limit) and TRPO step won't violate
    constraint, the dual must return nu = 0 (no cost projection)."""
    # Make the unconstrained TRPO step strictly cost-decreasing: r negative.
    q, r, s, c, delta = 1.0, -0.5, 1.0, -0.5, 0.01
    lam, nu, recovery = _solve_macpo_dual(q, r, s, c, delta)
    assert not recovery
    assert nu == 0.0
    # lam should be the TRPO step size sqrt(q / 2 delta).
    assert abs(lam - np.sqrt(q / (2 * delta))) < 1e-6


def test_dual_recovery_when_infeasible():
    """If c**2 > 2 delta s, no feasible policy exists in the trust region
    and the dual must signal recovery."""
    q, r, s, c, delta = 1.0, 0.0, 0.01, 5.0, 0.01
    lam, nu, recovery = _solve_macpo_dual(q, r, s, c, delta)
    assert recovery, "Dual must flag recovery when constraint cannot be met"
    assert lam is None and nu is None


def test_flat_params_roundtrip():
    """Flatten + unflatten must be the identity on a small MLP."""
    _seed()
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    flat0 = _flat_params(net)
    perturbed = flat0 + 0.1 * torch.randn_like(flat0)
    _set_flat_params(net, perturbed)
    assert torch.allclose(_flat_params(net), perturbed)


# ----------------------------------------------------------------------
# Agent contract
# ----------------------------------------------------------------------

def test_macpo_construct():
    _seed()
    agent = MACPO(obs_dim=19, global_obs_dim=109, n_actions=6, n_agents=4,
                  device="cpu")
    assert len(agent.actors) == 4
    assert agent.critic is not None
    assert agent.cost_critic is not None
    # Independent actor params.
    p0 = next(agent.actors[0].parameters()).detach().clone()
    p1 = next(agent.actors[1].parameters()).detach().clone()
    assert not torch.allclose(p0, p1)


def test_macpo_get_actions_returns_4tuple():
    """MACPO mirrors MAPPOLagrangian: (actions, log_probs, values, cost_values)."""
    _seed()
    agent = MACPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=4,
                  device="cpu")
    obs = {f"worker_{i}": np.random.randn(19).astype(np.float32) for i in range(4)}
    gs = np.random.randn(77).astype(np.float32)
    ret = agent.get_actions(obs, gs)
    assert len(ret) == 4
    actions, log_probs, values, cost_values = ret
    assert set(actions.keys()) == set(obs.keys())
    assert all(isinstance(v, int) for v in actions.values())
    assert all(isinstance(v, float) for v in cost_values.values())


def test_macpo_buffer_and_update_completes():
    """End-to-end: store rollout, run update loop, no exceptions."""
    _seed()
    n_agents = 4
    obs_dim, gobs_dim, n_actions = 19, 77, 6
    agent = MACPO(obs_dim=obs_dim, global_obs_dim=gobs_dim,
                  n_actions=n_actions, n_agents=n_agents,
                  batch_size=32, n_epochs=2,
                  cg_iters=5, line_search_steps=5,
                  device="cpu")
    T = 32
    for t in range(T):
        obs = {f"worker_{i}": np.random.randn(obs_dim).astype(np.float32)
               for i in range(n_agents)}
        gs = np.random.randn(gobs_dim).astype(np.float32)
        actions, log_probs, values, cost_values = agent.get_actions(obs, gs)
        rewards = {f"worker_{i}": float(np.random.randn()) for i in range(n_agents)}
        # Per-agent positive cost so the constraint mechanism actually engages.
        costs = {f"worker_{i}": float(abs(np.random.randn())) for i in range(n_agents)}
        done = (t == T - 1)
        agent.buffer.store_step = lambda *a, **kw: None  # buffer uses store(), not store_step
        # LagrangianRolloutBuffer's per-agent store interface.
        for i, aid in enumerate(sorted(obs.keys())):
            agent.buffer.store(
                obs=obs[aid], global_state=gs,
                action=actions[aid], log_prob=log_probs[aid],
                reward=rewards[aid], cost=costs[aid],
                done=done, values=values, cost_values=cost_values,
            )
    metrics = agent.update()
    assert "critic_loss" in metrics
    assert "cost_critic_loss" in metrics
    assert "n_total_agent_updates" in metrics
    # 2026-05-02 algorithmic-correctness fix: MACPO does ONE trust-region
    # pass per buffer batch (Achiam CPO Algorithm 1) regardless of
    # n_epochs. So n_total_agent_updates == n_agents (sequential per
    # agent in a single permutation), NOT n_agents * n_epochs. The
    # n_epochs hyperparameter now only governs critic gradient updates.
    assert metrics["n_total_agent_updates"] == n_agents
    # Either accepted or recovery -- must be non-zero attempted updates.
    assert metrics["n_total_agent_updates"] > 0
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["cost_critic_loss"])
    # Buffer cleared.
    assert len(agent.buffer) == 0


def test_macpo_save_load_roundtrip():
    _seed()
    a1 = MACPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=4,
               device="cpu")
    a2 = MACPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=4,
               device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "macpo.pt")
        a1.save(path)
        a2.load(path)
    for actor1, actor2 in zip(a1.actors, a2.actors):
        for p1, p2 in zip(actor1.parameters(), actor2.parameters()):
            assert torch.allclose(p1, p2)
    for p1, p2 in zip(a1.critic.parameters(), a2.critic.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(a1.cost_critic.parameters(), a2.cost_critic.parameters()):
        assert torch.allclose(p1, p2)


def test_macpo_lambda_property_and_no_op_update():
    """train.py's outer loop calls agent.update_lambda(mean_cost) and reads
    agent.lam regardless of method. MACPO's update_lambda is a no-op and
    .lam returns the most recent dual nu (0 before any update)."""
    _seed()
    agent = MACPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=4,
                  device="cpu")
    assert agent.lam == 0.0
    agent.update_lambda(mean_cost=0.5)  # must not raise
    assert agent.lam == 0.0
