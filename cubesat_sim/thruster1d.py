"""1D thruster plant: a point mass pushed by a one-directional, throttled thruster.

State x = [position (m), velocity (m/s)]. Equations:
    m * x_ddot = F_avg + F_d
    D     = sat(F_cmd / F_max, 0, 1)      duty cycle, one direction only
    F_avg = D * F_max                     continuous throttle (not an on/off switch)

The command is the requested force F_cmd (N); the plant applies the [0, F_max] limit, as
dynamics.limit_wheel_torque does for the wheels. The thruster cannot push in -x, so the
plant cannot brake: only a negative F_d slows it down. In the full system the reaction
wheels point the thruster.

All functions are pure JAX and work under jax.jit, jax.grad and jax.vmap.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .sim import simulate_plant


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ThrusterParams:
    mass: jax.Array      # kg
    F_max: jax.Array     # N, thrust at duty cycle D = 1


def limit_force(F_cmd, params):
    """Applied thrust F_avg = D * F_max with D = sat(F_cmd / F_max, 0, 1)."""
    return jnp.clip(F_cmd, 0.0, params.F_max)


def duty_cycle(F_cmd, params):
    return limit_force(F_cmd, params) / params.F_max


def state_derivative(x, F_cmd, F_ext, params):
    F = limit_force(F_cmd, params)
    return jnp.stack([x[1], (F + F_ext) / params.mass])


def rk4_step_with_force(x, F_cmd, F_ext, dt, params):
    """One fixed step. The command and the external force stay constant in the step.

    Returns (x_next, F_applied). The limit does not depend on the state, so the applied
    thrust is the limited command for the whole step.
    """
    k1 = state_derivative(x, F_cmd, F_ext, params)
    k2 = state_derivative(x + 0.5 * dt * k1, F_cmd, F_ext, params)
    k3 = state_derivative(x + 0.5 * dt * k2, F_cmd, F_ext, params)
    k4 = state_derivative(x + dt * k3, F_cmd, F_ext, params)
    x_next = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return x_next, limit_force(F_cmd, params)


def rk4_step(x, F_cmd, F_ext, dt, params):
    return rk4_step_with_force(x, F_cmd, F_ext, dt, params)[0]


class DisturbanceProfile:
    """External force F_d(t): a constant plus piecewise-constant segments [t_start, t_end).

    Equal and hashable by value (like sim.OpenLoopCommand), so run() reuses the compiled
    code for the same profile. Evaluated at the start of each step and held for the step.
    """

    def __init__(self, F_const, segments=()):
        self.F_const = float(F_const)
        self.segments = tuple((float(a), float(b), float(F)) for a, b, F in segments)

    def _key(self):
        return (self.F_const, self.segments)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, DisturbanceProfile) and self._key() == other._key()

    def __repr__(self):
        return f"DisturbanceProfile(F_const={self.F_const}, segments={self.segments})"

    def __call__(self, t, x):
        F = jnp.asarray(self.F_const, dtype=jnp.result_type(float))
        if self.segments:
            t_start = jnp.array([s[0] for s in self.segments])
            t_end = jnp.array([s[1] for s in self.segments])
            F_seg = jnp.array([s[2] for s in self.segments])
            active = (t_start <= t) & (t < t_end)
            F = F + active.astype(F.dtype) @ F_seg
        return F


def zero_force_command(t, x):
    return jnp.asarray(0.0)


def simulate_thruster(params, x0, dt, steps, command, external_force):
    """Propagate x0 = [position, velocity] through `steps` RK4 steps of size dt.

    command(t, x) -> requested force F_cmd (N), or a stateful command (see
        sim.simulate_plant). external_force(t, x) -> F_d (N). Both are evaluated at the
        start of each step and held for the step.

    Returns a dict of arrays, one row per time t[k]: t, x, v, F_cmd, F (applied thrust),
    D (duty cycle), F_d. F[k] and F_d[k] act over [t[k], t[k+1]]; the last row is the
    command at the final state.
    """
    def step(tk, xk, F_cmd):
        return rk4_step_with_force(xk, F_cmd, external_force(tk, xk), dt, params)

    def limit(tk, xk, F_cmd):
        return limit_force(F_cmd, params)

    def observe(tk, xk):
        return {"F_d": jnp.asarray(external_force(tk, xk), dtype=xk.dtype)}

    t, X, F_cmd, F, obs = simulate_plant(step, limit, observe, x0, dt, steps, command)
    return {"t": t, "x": X[:, 0], "v": X[:, 1], "F_cmd": F_cmd, "F": F,
            "D": F / params.F_max, "F_d": obs["F_d"]}


_simulate_thruster_jit = jax.jit(simulate_thruster,
                                 static_argnames=("dt", "steps", "command", "external_force"))


def run_thruster(cfg, command=None, external_force=None):
    """Run one thruster simulation from a ThrusterConfig, compiled with jax.jit.

    command: default zero thrust (coast). external_force: default cfg.disturbance.
    """
    command = command or zero_force_command
    external_force = external_force or cfg.disturbance
    return _simulate_thruster_jit(cfg.params, cfg.sim.x0, dt=cfg.sim.dt,
                                  steps=cfg.sim.steps, command=command,
                                  external_force=external_force)
