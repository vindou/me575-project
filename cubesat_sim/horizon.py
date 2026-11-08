"""Shared pieces of the receding-horizon controllers (mpc.py and mppi.py).

A plan is a sequence of normalized wheel torques v = u / u_max in [-1, 1], shape
(horizon_steps, N), each held for one control interval. The predicted cost of a plan uses
the plant's own nonlinear equations (dynamics.rk4_step) with no external torque.

Stage cost for each predicted state (rad^2 or dimensionless):
    ||e_R||^2                                 e_R = 2 vec(q_err), so 4 sin^2(theta/2)
  + (rate_weight_s * ||omega||)^2             rate damping, rate_weight_s in seconds
  + w_rate_limit * relu(||omega||^2 / omega_max^2 - 1)^2   soft slew-rate limit (optional)
  + w_momentum * sum relu(|h_i| / h_warn - 1)^2            soft wheel momentum limit (optional)
plus w_torque * ||v||^2 per interval, and the last stage weighted by w_terminal.
The rate and momentum terms are soft penalties, not hard constraints.
"""
import math

import jax
import jax.numpy as jnp

from . import dynamics as dyn
from . import quaternion as quat


def check_positive(name, value):
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive.")
    return value


def check_nonnegative(name, value):
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def check_count(name, value):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def hold_steps_for(cfg, control_interval_s):
    """Number of simulator steps per control interval; it must be a whole number."""
    dt_sim = float(cfg.sim.dt)
    Tc = check_positive("control_interval_s", control_interval_s)
    hold = int(round(Tc / dt_sim))
    if hold < 1 or abs(Tc / dt_sim - hold) > 1e-9 * max(1.0, Tc / dt_sim):
        raise ValueError(f"control_interval_s must be a whole multiple of sim dt ({dt_sim} s).")
    return Tc, hold


def horizon_cost(cfg, q_target, *, control_interval_s, substeps, rate_weight_s, w_torque,
                 w_terminal, max_rate_deg_s, w_rate_limit, momentum_warning_frac,
                 w_momentum):
    """Check the cost settings and return (cost(V, x0), settings dict).

    cost maps a plan V of normalized torques, shape (horizon, N), and the current state
    x0 to the predicted cost. It is pure JAX: it works with jax.grad and jax.vmap.
    """
    Tc = check_positive("control_interval_s", control_interval_s)
    substeps = check_count("substeps", substeps)
    rate_weight_s = check_nonnegative("rate_weight_s", rate_weight_s)
    w_torque = check_nonnegative("w_torque", w_torque)
    w_terminal = check_nonnegative("w_terminal", w_terminal)
    w_rate_limit = check_nonnegative("w_rate_limit", w_rate_limit)
    w_momentum = check_nonnegative("w_momentum", w_momentum)
    omega_max = None
    if max_rate_deg_s is not None:
        max_rate_deg_s = check_positive("max_rate_deg_s", max_rate_deg_s)
        omega_max = math.radians(max_rate_deg_s)
    h_warn = None
    if momentum_warning_frac is not None:
        momentum_warning_frac = check_positive("momentum_warning_frac", momentum_warning_frac)
        if momentum_warning_frac > 1.0:
            raise ValueError("momentum_warning_frac must be at most 1.")
        h_warn = momentum_warning_frac * cfg.wheels.h_max

    q_target = jnp.asarray(q_target)
    q_target = q_target / jnp.linalg.norm(q_target)
    body, wheels = cfg.body, cfg.wheels
    h_model = Tc / substeps
    no_disturbance = jnp.zeros(3)

    def state_cost(x):
        q, w, h = dyn.split(x)
        q_err = quat.quarternion_product(quat.conjugate(q), q_target)  # as in PD
        # Sign-invariant: ||vec(q_err)||^2 = sin^2(theta/2) for both q_err and -q_err.
        cost = 4.0 * q_err[:3] @ q_err[:3] + rate_weight_s**2 * (w @ w)
        if omega_max is not None:
            # Squared norm keeps the gradient finite at omega = 0.
            cost += w_rate_limit * jax.nn.relu(w @ w / omega_max**2 - 1.0) ** 2
        if h_warn is not None:
            cost += w_momentum * jnp.sum(jax.nn.relu(jnp.abs(h) / h_warn - 1.0) ** 2)
        return cost

    def cost(V, x0):
        def interval(x, v):
            u = wheels.u_max * v
            for _ in range(substeps):
                x = dyn.rk4_step(x, u, no_disturbance, h_model, body, wheels)
            return x, state_cost(x)

        _, costs = jax.lax.scan(interval, x0, V)
        return (jnp.sum(costs[:-1]) + w_terminal * costs[-1]
                + w_torque * jnp.sum(V * V))

    settings = {
        "substeps": substeps, "rate_weight_s": rate_weight_s, "w_torque": w_torque,
        "w_terminal": w_terminal, "max_rate_deg_s": max_rate_deg_s,
        "w_rate_limit": w_rate_limit, "momentum_warning_frac": momentum_warning_frac,
        "w_momentum": w_momentum,
    }
    return cost, settings


def receding_horizon(replan, init_aux, hold_steps, n_wheels):
    """Wrap replan into the simulator's stateful command interface.

    replan(x, aux, first) -> (u_cmd, aux) runs at every hold_steps-th simulator step
    (first is True at step 0); u_cmd is held in between. aux is the solver's carried
    state (warm-start plan, random key, ...), a JAX pytree.
    """
    def command(t, x, state):
        aux, u_hold, k = state
        u_cmd, aux = jax.lax.cond(k % hold_steps == 0,
                                  lambda: replan(x, aux, k == 0),
                                  lambda: (u_hold, aux))
        return u_cmd, (aux, u_cmd, k + 1)

    command.init_state = (init_aux, jnp.zeros(n_wheels), jnp.asarray(0, dtype=jnp.int32))
    return command


def shift_plan(P):
    """Warm start for the next solve: drop the applied interval, repeat the last one."""
    return jnp.concatenate([P[1:], P[-1:]], axis=0)
