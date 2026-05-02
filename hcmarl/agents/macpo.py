"""
HC-MARL: Multi-Agent Constrained Policy Optimisation (MACPO) baseline.

Reference:
    Gu, S., Kuba, J. G., Chen, Y., Du, Y., Yang, L., Knoll, A., & Yang, Y.
    (2023). Multi-Agent Constrained Policy Optimisation. arXiv:2110.02793.

Algorithmic identity:
  - Per-agent (heterogeneous) actors -- no parameter sharing.
  - Single centralised reward critic V_r(s, agent_id).
  - Single centralised cost critic V_c(s, agent_id).
  - Sequential per-agent update with random permutation each epoch.
  - Joint advantage decomposition (HAPPO Lemma 1) applied to BOTH reward
    and cost advantage streams via the M^{i_{1:m-1}} importance product.
  - For each agent's update: trust-region constrained step (CPO/MACPO
    Eq. 4.2) -- maximise reward gradient g^T Δθ subject to:
        d + b^T Δθ <= 0           (cost constraint, single constraint m=1)
        ½ Δθ^T H Δθ <= δ          (KL trust region)
    where H is the Fisher information of the agent's policy.
  - Conjugate-gradient natural-gradient direction (Hessian-free).
  - Backtracking line search to satisfy both constraints.
  - Recovery direction when current policy is infeasible (c² > 2 δ s).

Differences from MAPPO-Lagrangian (which is MACPO's "Lagrangian variant"):
  - Trust-region step instead of PPO clipping.
  - Hard cost constraint via dual projection, not soft λ-penalty in actor loss.
  - No PID Lagrangian -- the dual variables ν, λ are recomputed per step.

This file deliberately mirrors mappo_lag.py's buffer / get_actions /
update / save / load contract so train.py and run_baselines.py work
without a special-case branch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Callable

from hcmarl.agents.networks import ActorNetwork, CriticNetwork, CostCriticNetwork
from hcmarl.agents.mappo_lag import LagrangianRolloutBuffer


# =====================================================================
# Helpers: flat-parameter manipulation, Hessian-vector products, CG.
# =====================================================================

def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def _set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


def _flat_grad(loss: torch.Tensor, params, retain_graph: bool = False,
               create_graph: bool = False) -> torch.Tensor:
    """torch.autograd.grad of `loss` wrt `params`, flattened.

    Uses allow_unused=True + zero-fill so that parameters not touched by
    `loss` (e.g. critic params during actor-only loss) yield a clean zero
    contribution rather than crashing.
    """
    grads = torch.autograd.grad(loss, params,
                                retain_graph=retain_graph,
                                create_graph=create_graph,
                                allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).flatten())
        else:
            flat.append(g.contiguous().flatten())
    return torch.cat(flat)


def _conjugate_gradient(matvec_fn: Callable[[torch.Tensor], torch.Tensor],
                         b: torch.Tensor,
                         n_iters: int = 10,
                         tol: float = 1e-10) -> torch.Tensor:
    """Solve A x = b via CG, where A is accessed only through matvec_fn(v) = A @ v.

    Standard conjugate-gradient iteration (textbook reference: Boyd &
    Vandenberghe 2004 chapter on iterative methods, math doc ref [25]).
    No preconditioning -- the Hessian damping in fisher_vector_product
    handles ill-conditioning.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)
    for _ in range(n_iters):
        Ap = matvec_fn(p)
        alpha = rdotr / (p.dot(Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        new_rdotr = r.dot(r)
        if float(new_rdotr) < tol:
            break
        beta = new_rdotr / (rdotr + 1e-12)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def _solve_macpo_dual(q: float, r: float, s: float, c: float,
                       delta: float, eps: float = 1e-8):
    """Solve the CPO/MACPO single-cost-constraint dual.

    Inputs (all scalars):
        q = g^T H^{-1} g                  (always > 0 for non-zero g)
        r = g^T H^{-1} b                  (cross term; sign-free)
        s = b^T H^{-1} b                  (always > 0 for non-zero b)
        c = J_cost(pi_k) - cost_limit     (cost surplus; positive = infeasible)
        delta = KL trust-region radius

    Returns:
        (lam, nu, recovery)
            recovery=True  -> use pure cost-reduction direction, ignore reward.
            recovery=False -> apply step Delta theta = (1/lam)(x_g - nu * x_b)
                              where x_g = H^{-1} g, x_b = H^{-1} b.

    Notes on the case split (Achiam 2017 CPO Appendix B; MACPO inherits):
      - If c^2 > 2 delta s and c > 0, no policy in the trust region can
        satisfy the cost constraint -> recovery.
      - Otherwise we solve the KKT system. With one constraint and the
        Lagrangian L(theta) = -g^T (theta - theta_k) + nu (d + b^T (theta - theta_k))
                              + (lam/2) ((theta - theta_k)^T H (theta - theta_k) - delta),
        the optimum has nu = max(0, (lam r - c) / s) and lam from the
        scalar quadratic 2 lam^2 delta s = q s - r^2 + (lam r - c)^2_+
        which we solve by closed form. The simplified form below uses
        lam_a = sqrt((q s - r^2) / max(2 delta s - c^2, eps)) when both
        constraints are active, else the unconstrained TRPO lam_b.
    """
    s_safe = max(float(s), eps)
    q_safe = max(float(q), eps)

    # KL-only (TRPO) step size, always defined.
    lam_b = float(np.sqrt(q_safe / (2.0 * delta)))

    # Recovery: only triggered when *currently* infeasible (c > 0) AND
    # the cost surplus is too large to be retired within the trust region.
    if c > 0.0 and (c * c) > 2.0 * delta * s_safe:
        return None, None, True

    A = q_safe * s_safe - r * r
    B = 2.0 * delta * s_safe - c * c

    # When c is large-negative (deeply feasible), B can go negative without
    # any infeasibility. In that case the both-active formula is undefined
    # but the pure-reward TRPO step is fine -- use it.
    if B <= 0.0:
        if c <= 0.0:
            return lam_b, 0.0, False
        # c > 0 with B <= 0 corresponds to (c**2 >= 2 delta s); the recovery
        # check above already returned, so this path is unreachable in
        # practice. Be defensive.
        return None, None, True

    # Both constraints potentially active:
    lam_a = float(np.sqrt(max(A, eps) / max(B, eps)))

    # Decide which regime is actually active. The unprojected TRPO step
    # changes expected cost by r / lam_b (since b^T Delta theta_TRPO = r/lam_b).
    # If c is already feasible and the TRPO step keeps us feasible, the cost
    # constraint is inactive: pure reward step with nu = 0.
    if c <= 0.0 and (c + r / lam_b) <= 0.0:
        lam = lam_b
        nu = 0.0
    else:
        lam = lam_a
        nu = max(0.0, (lam * r - c) / s_safe)

    return lam, nu, False


# =====================================================================
# MACPO agent.
# =====================================================================


class MACPO:
    """Multi-Agent Constrained Policy Optimisation.

    Per-agent heterogeneous actors. Each agent's update is a CPO-style
    trust-region step with Conjugate-Gradient natural gradient, single-
    cost-constraint dual projection, and backtracking line search.
    Sequential update with HAPPO-style M^{i_{1:m-1}} advantage product.
    """

    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 lr_critic=1e-3,
                 gamma=0.99, gae_lambda=0.95,
                 entropy_coeff=0.0, max_grad_norm=0.5,
                 n_epochs=10, batch_size=256,
                 hidden_dim=64, critic_hidden_dim=128,
                 cost_limit=0.1,
                 delta_kl=0.01,
                 cg_iters=10, cg_damping=0.1,
                 line_search_steps=10, line_search_decay=0.8,
                 device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cost_limit = cost_limit
        self.delta_kl = delta_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_steps = line_search_steps
        self.line_search_decay = line_search_decay

        # Heterogeneous actors.
        self.actors = nn.ModuleList([
            ActorNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ])
        # No actor optimisers -- the trust-region step is computed by
        # CG + line search and applied directly via _set_flat_params.

        # Centralised reward and cost critics, agent-id one-hot augmented.
        self.critic = CriticNetwork(global_obs_dim + n_agents, critic_hidden_dim).to(self.device)
        self.cost_critic = CostCriticNetwork(global_obs_dim + n_agents).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.cost_critic_optim = optim.Adam(self.cost_critic.parameters(), lr=lr_critic)

        agent_ids = [f"worker_{i}" for i in range(n_agents)]
        self.buffer = LagrangianRolloutBuffer(agent_ids=agent_ids)

        # Diagnostics for the most recent update.
        self._last_diagnostics = {}

    # ------------------------------------------------------------------
    # Compatibility shim with mappo_lag.MAPPOLagrangian's `lam` property.
    # MACPO does not maintain a Lagrangian dual variable (the dual is
    # solved per-update from CG outputs and is not persisted). We expose
    # 0.0 so logger code that reads agent.lam keeps working.
    # ------------------------------------------------------------------
    @property
    def lam(self) -> float:
        return float(self._last_diagnostics.get("nu", 0.0))

    def update_lambda(self, mean_cost: float) -> None:
        """No-op: MACPO solves the dual analytically each step."""
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augment_gs(self, gs_tensor, agent_idx):
        one_hot = torch.zeros(self.n_agents, device=self.device)
        one_hot[agent_idx] = 1.0
        return torch.cat([gs_tensor, one_hot])

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def get_actions(self, observations: Dict[str, np.ndarray], global_state: np.ndarray):
        """Per-agent actor forward + reward and cost critic forwards.

        Returns (actions, log_probs, values, cost_values) -- the same
        4-tuple shape as MAPPOLagrangian, so train.py's mappo_lag
        branch handles MACPO without a special case.
        """
        sorted_agents = sorted(observations.keys())
        actions, log_probs, values, cost_values = {}, {}, {}, {}
        gs_np = np.asarray(global_state, dtype=np.float32)
        for i, agent_id in enumerate(sorted_agents):
            obs_t = torch.from_numpy(observations[agent_id]).float().unsqueeze(0).to(self.device)
            one_hot = np.zeros(self.n_agents, dtype=np.float32)
            one_hot[i] = 1.0
            gs_aug = np.concatenate([gs_np, one_hot]).astype(np.float32)
            gs_t = torch.from_numpy(gs_aug).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, lp, _ = self.actors[i].get_action(obs_t)
                value = self.critic(gs_t)
                cost_value = self.cost_critic(gs_t)
            actions[agent_id] = int(action.item())
            log_probs[agent_id] = float(lp.item())
            values[agent_id] = float(value.item())
            cost_values[agent_id] = float(cost_value.item())
        return actions, log_probs, values, cost_values

    # ------------------------------------------------------------------
    # Per-agent trust-region step.
    # ------------------------------------------------------------------

    def _fisher_vector_product(self, actor: nn.Module,
                                obs: torch.Tensor,
                                vec: torch.Tensor) -> torch.Tensor:
        """H @ vec, where H is the Fisher information of `actor` evaluated at
        its current parameters and `obs`. Approximated as the Hessian of
        the mean KL between the detached old policy and the live policy."""
        params = list(actor.parameters())

        # Recompute logits with grad on -- need create_graph for second deriv.
        logits = actor.forward(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Detach as the "old" policy snapshot (zero gradient to old).
        with torch.no_grad():
            old_log_probs = log_probs.detach()
            old_probs = torch.softmax(logits, dim=-1).detach()
        # KL(pi_old || pi_new) -- has zero value & gradient at theta = theta_k
        # but non-zero Hessian (= Fisher info).
        kl = (old_probs * (old_log_probs - log_probs)).sum(-1).mean()

        flat_grad_kl = _flat_grad(kl, params, retain_graph=True, create_graph=True)
        gv = (flat_grad_kl * vec).sum()
        flat_fvp = _flat_grad(gv, params, retain_graph=False, create_graph=False)
        return flat_fvp + self.cg_damping * vec

    def _kl_at(self, actor: nn.Module,
                obs: torch.Tensor,
                old_log_probs: torch.Tensor) -> torch.Tensor:
        """Mean KL between a fixed `old_log_probs` snapshot and `actor`'s current logits."""
        logits = actor.forward(obs)
        new_log_probs = torch.log_softmax(logits, dim=-1)
        # Use OLD probabilities as the reference distribution.
        old_probs = old_log_probs.exp()
        kl = (old_probs * (old_log_probs - new_log_probs)).sum(-1).mean()
        return kl

    def _agent_update(self, j: int,
                       obs_j: torch.Tensor,
                       acts_j: torch.Tensor,
                       old_lp_j: torch.Tensor,
                       m_adv_r: torch.Tensor,
                       m_adv_c: torch.Tensor,
                       cost_surplus: float):
        """One MACPO update step for agent j.

        Returns dict of diagnostics (q, r, s, c, lam, nu, kl_after, accepted).
        """
        actor = self.actors[j]
        params = list(actor.parameters())

        # Old log-prob distribution snapshot for KL computation in line search.
        with torch.no_grad():
            old_logits = actor.forward(obs_j).detach()
            old_log_probs_full = torch.log_softmax(old_logits, dim=-1)

        # ---- Compute reward and cost gradients g, b at theta_k. ----
        new_lp, entropy = actor.evaluate(obs_j, acts_j)
        ratio = (new_lp - old_lp_j).exp()
        # We MAXIMISE reward, so g = grad of E[ratio * adv_r]. At theta = theta_k
        # the ratio is 1, so g equals the policy gradient.
        surr_r = (ratio * m_adv_r).mean() + self.entropy_coeff * entropy.mean()
        g = _flat_grad(surr_r, params, retain_graph=True).detach()

        # Cost gradient: increasing E[ratio * adv_c] increases expected cost.
        new_lp_c, _ = actor.evaluate(obs_j, acts_j)
        ratio_c = (new_lp_c - old_lp_j).exp()
        surr_c = (ratio_c * m_adv_c).mean()
        b = _flat_grad(surr_c, params, retain_graph=False).detach()

        # Zero-gradient guard. If g is numerically zero (policy near
        # optimum, or all advantages happen to be zero this batch), CG
        # returns zero, and the dual gives lam=0 -> 1/lam in the step
        # formula is a NaN/Inf bomb. Skip cleanly.
        if float(g.norm()) < 1e-8:
            return {
                "q": 0.0, "r": 0.0, "s": 0.0, "c": float(cost_surplus),
                "lam": 0.0, "nu": 0.0, "alpha": 0.0,
                "recovery": False, "accepted": False,
            }

        # ---- Conjugate gradient: x_g = H^{-1} g, x_b = H^{-1} b. ----
        fvp_fn = lambda v: self._fisher_vector_product(actor, obs_j, v)
        x_g = _conjugate_gradient(fvp_fn, g, n_iters=self.cg_iters)
        x_b = _conjugate_gradient(fvp_fn, b, n_iters=self.cg_iters)

        q = float(g.dot(x_g).item())
        r = float(g.dot(x_b).item())
        s = float(b.dot(x_b).item())

        # ---- Solve dual to get (lam, nu) or recovery flag. ----
        lam, nu, recovery = _solve_macpo_dual(q, r, s, cost_surplus, self.delta_kl)

        if recovery:
            # Pure cost-reduction direction. Ignore reward; minimise cost.
            # Step magnitude: sqrt(2 delta / s_safe).
            s_safe = max(s, 1e-8)
            step = -float(np.sqrt(2.0 * self.delta_kl / s_safe)) * x_b
            lam, nu = 0.0, 0.0
        else:
            step = (1.0 / max(lam, 1e-8)) * (x_g - nu * x_b)

        # ---- Backtracking line search on KL + cost feasibility. ----
        # Achiam CPO 2017 Algorithm 1 line-search criterion:
        #   - In RECOVERY mode (current iterate is infeasible AND cannot
        #     be made feasible within the trust region): the step direction
        #     is BY CONSTRUCTION cost-reducing (-sqrt(2*delta/s) * x_b).
        #     Line search only checks KL; cost is being explicitly reduced
        #     by the step's geometry. Don't impose a tight cost-decrease
        #     check inside the line search -- that double-counts the
        #     guarantee and rejects valid recovery steps.
        #   - In NORMAL mode: accept if cost is feasible after step OR if
        #     the surrogate cost is decreasing. This is the "no-worsening"
        #     relaxation common in shipping CPO implementations.
        flat_old = _flat_params(actor)
        accepted = False
        alpha = 1.0
        for _ in range(self.line_search_steps):
            new_flat = flat_old + alpha * step
            _set_flat_params(actor, new_flat)

            with torch.no_grad():
                kl_now = float(self._kl_at(actor, obs_j, old_log_probs_full).item())
                kl_ok = kl_now <= 1.5 * self.delta_kl  # 50% slack for numerics

                if recovery:
                    # Recovery direction is cost-reducing by construction;
                    # only check KL.
                    cost_ok = True
                else:
                    new_lp_after, _ = actor.evaluate(obs_j, acts_j)
                    new_ratio = (new_lp_after - old_lp_j).exp()
                    cost_surr_new = float((new_ratio * m_adv_c).mean().item())
                    # Accept if (a) cost is feasible after the step, or
                    # (b) the surrogate cost change is non-positive.
                    feasible_after = (cost_surplus + cost_surr_new) <= 1e-3
                    cost_decreasing = cost_surr_new <= 1e-3
                    cost_ok = feasible_after or cost_decreasing

            if kl_ok and cost_ok:
                accepted = True
                break
            alpha *= self.line_search_decay

        if not accepted:
            # Revert to old parameters; this update produces no step.
            _set_flat_params(actor, flat_old)

        return {
            "q": float(q), "r": float(r), "s": float(s),
            "c": float(cost_surplus),
            "lam": float(lam), "nu": float(nu),
            "alpha": float(alpha) if accepted else 0.0,
            "recovery": bool(recovery),
            "accepted": bool(accepted),
            "kl_at_step": float(kl_now if accepted else 0.0),
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        T = self.buffer._n_steps
        N = self.n_agents

        # -- Compute per-agent reward + cost advantages.
        with torch.no_grad():
            last_gs = torch.FloatTensor(
                self.buffer._global_states[-1]
            ).to(self.device)
            last_values = {}
            last_cost_values = {}
            for i, a in enumerate(self.buffer.agent_ids):
                gs_aug = self._augment_gs(last_gs, i)
                last_values[a] = self.critic(gs_aug).item()
                last_cost_values[a] = self.cost_critic(gs_aug).item()

        adv_flat, ret_flat, cadv_flat, cret_flat = self.buffer.compute_returns(
            last_values, last_cost_values, self.gamma, self.gae_lambda
        )
        # Reshape to (T, N) per-agent layout.
        adv_TN = adv_flat.reshape(T, N)
        ret_TN = ret_flat.reshape(T, N)
        cadv_TN = cadv_flat.reshape(T, N)
        cret_TN = cret_flat.reshape(T, N)

        # Per-agent tensors.
        per_obs, per_acts, per_old_lp = {}, {}, {}
        per_adv, per_cadv = {}, {}
        per_ret, per_cret = {}, {}
        for i, a in enumerate(self.buffer.agent_ids):
            per_obs[i] = torch.from_numpy(np.array(self.buffer._obs[a], dtype=np.float32)).to(self.device)
            per_acts[i] = torch.from_numpy(np.array(self.buffer._actions[a], dtype=np.int64)).to(self.device)
            per_old_lp[i] = torch.from_numpy(np.array(self.buffer._log_probs[a], dtype=np.float32)).to(self.device)
            adv_i = torch.from_numpy(adv_TN[:, i].astype(np.float32)).to(self.device)
            per_adv[i] = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
            cadv_i = torch.from_numpy(cadv_TN[:, i].astype(np.float32)).to(self.device)
            # Do NOT centre cost advantages (Stooke et al.; same fix as MAPPO-Lag C-10.A).
            per_cadv[i] = cadv_i
            per_ret[i] = torch.from_numpy(ret_TN[:, i].astype(np.float32)).to(self.device)
            per_cret[i] = torch.from_numpy(cret_TN[:, i].astype(np.float32)).to(self.device)

        # Augmented global states per-agent for critic targets.
        gs_np = np.array(self.buffer._global_states, dtype=np.float32)
        per_gs = []
        for i in range(N):
            one_hot = np.zeros(N, dtype=np.float32)
            one_hot[i] = 1.0
            gs_aug = np.concatenate([gs_np, np.broadcast_to(one_hot, (T, N))], axis=1)
            per_gs.append(torch.from_numpy(gs_aug).to(self.device))

        # Cost surplus at theta_k: empirical mean cost across the rollout
        # minus the per-step cost limit. We use the rollout estimate of
        # J_cost(pi_k) since this is what the constraint refers to.
        all_costs = np.concatenate([np.array(self.buffer._costs[a], dtype=np.float32)
                                     for a in self.buffer.agent_ids])
        mean_cost = float(all_costs.mean())
        cost_surplus = mean_cost - self.cost_limit

        last_critic_loss = last_cost_critic_loss = 0.0
        n_accepted = 0
        n_recovery = 0
        n_total_agent_updates = 0
        last_diag = {}

        # ---- ACTOR PASS: ONE trust-region step per buffer batch. ----
        # CRITICAL CORRECTNESS NOTE (2026-05-02): trust-region methods
        # linearize around theta_k. Running n_epochs sequential
        # trust-region passes drifts theta far from the linearization
        # point and produces degenerate dual values; in our pre-launch
        # GPU probe this manifested as actor_loss diverging to -1.4M..-6M
        # and SPS collapsing to ~21 (10x below the runbook floor)
        # because each epoch did 6 agents x CG x line search.
        # Achiam CPO (2017) Algorithm 1, Spinning Up CPO, and SafePO
        # MACPO all run a SINGLE trust-region step per data batch and
        # do critic updates separately. We follow that here.
        perm = np.random.permutation(N)
        M_running = torch.ones(T, device=self.device)

        for j in perm:
            m_adv_r = (M_running.detach() * per_adv[j])
            m_adv_c = (M_running.detach() * per_cadv[j])
            diag = self._agent_update(
                j=j,
                obs_j=per_obs[j],
                acts_j=per_acts[j],
                old_lp_j=per_old_lp[j],
                m_adv_r=m_adv_r,
                m_adv_c=m_adv_c,
                cost_surplus=cost_surplus,
            )
            last_diag = diag
            n_total_agent_updates += 1
            if diag["accepted"]:
                n_accepted += 1
            if diag["recovery"]:
                n_recovery += 1

            # Update M_running with this agent's post-update ratio.
            with torch.no_grad():
                new_lp_after, _ = self.actors[j].evaluate(per_obs[j], per_acts[j])
                new_ratio = (new_lp_after - per_old_lp[j]).exp()
                new_ratio = new_ratio.clamp(0.5, 2.0)  # numerical safety
                M_running = M_running * new_ratio

        # ---- CRITIC PASSES: n_epochs gradient updates on reward + cost
        # critics. These are NOT trust-region constrained; standard MSE
        # regression on per-agent value targets. Pre-compute the
        # flattened tensors once (they don't change across epochs).
        gs_all = torch.cat(per_gs, dim=0)
        ret_all = torch.cat([per_ret[i] for i in range(N)], dim=0)
        cret_all = torch.cat([per_cret[i] for i in range(N)], dim=0)

        for _ in range(self.n_epochs):
            values = self.critic(gs_all)
            critic_loss = nn.functional.mse_loss(values, ret_all)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()
            last_critic_loss = critic_loss.item()

            cost_values = self.cost_critic(gs_all)
            cost_critic_loss = nn.functional.mse_loss(cost_values, cret_all)
            self.cost_critic_optim.zero_grad()
            cost_critic_loss.backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_critic_optim.step()
            last_cost_critic_loss = cost_critic_loss.item()

        self._last_diagnostics = last_diag
        self.buffer.clear()

        # actor_loss diagnostic: report -q/lam as the trust-region step
        # quality only when the step is in normal mode AND lam is sane.
        # In recovery mode lam=0 and the formula -q/1e-8 produces ~10^7
        # garbage values that pollute logs. Use 0.0 as a sentinel in
        # recovery mode (the real "loss" is the cost-reduction step,
        # not a reward-gradient quantity).
        last_lam = float(last_diag.get("lam", 0.0) or 0.0)
        last_q = float(last_diag.get("q", 0.0) or 0.0)
        if last_diag.get("recovery", False) or last_lam < 1e-6:
            actor_loss_report = 0.0
        else:
            actor_loss_report = float(-last_q / last_lam)

        return {
            "actor_loss": actor_loss_report,
            "critic_loss": float(last_critic_loss),
            "cost_critic_loss": float(last_cost_critic_loss),
            "lambda": float(last_diag.get("nu", 0.0)),
            "kl_step": float(last_diag.get("alpha", 0.0)),
            "n_accepted": int(n_accepted),
            "n_recovery": int(n_recovery),
            "n_total_agent_updates": int(n_total_agent_updates),
            "cost_surplus": float(cost_surplus),
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "cost_critic_optim": self.cost_critic_optim.state_dict(),
            "n_agents": self.n_agents,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("n_agents", self.n_agents) != self.n_agents:
            raise ValueError(
                f"MACPO checkpoint has n_agents={ckpt['n_agents']} "
                f"but agent was constructed with n_agents={self.n_agents}."
            )
        for a, sd in zip(self.actors, ckpt["actors"]):
            a.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])
        self.cost_critic.load_state_dict(ckpt["cost_critic"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if "cost_critic_optim" in ckpt:
            self.cost_critic_optim.load_state_dict(ckpt["cost_critic_optim"])
