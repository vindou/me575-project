"""Simulation loop: propagate the initial conditions through time.

simulate() is a pure function built on jax.lax.scan. Call it inside your own
jax.jit / jax.grad / jax.vmap to differentiate through a whole run or to batch runs.
run() is the convenience wrapper: it compiles simulate() for one config.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import dynamics as dyn


def zero_torque(t, x):
    return jnp.zeros(3)


class OpenLoopCommand:
    """Piecewise-constant wheel torque profile, zero outside all segments.

    Equal and hashable by value, so run() reuses the compiled code for the same profile.
    The segment values are constants, not differentiable inputs.
    """

    def __init__(self, segments, n):
        self.n = n
        self.segments = tuple((float(s.t_start), float(s.t_end),
                               tuple(float(v) for v in np.asarray(s.u))) for s in segments)

    def __hash__(self):
        return hash((self.n, self.segments))

    def __eq__(self, other):
        return (isinstance(other, OpenLoopCommand)
                and (self.n, self.segments) == (other.n, other.segments))

    def __call__(self, t, x):
        if not self.segments:
            return jnp.zeros(self.n)
        t_start = jnp.array([s[0] for s in self.segments])
        t_end = jnp.array([s[1] for s in self.segments])
        U = jnp.array([s[2] for s in self.segments])     # (segments, N)
        active = (t_start <= t) & (t < t_end)
        return active.astype(U.dtype) @ U


def open_loop_command(cfg):
    """Make a command function from the open_loop_command section of the file."""
    return OpenLoopCommand(cfg.open_loop_command, cfg.wheels.count)


def simulate(body, wheels, x0, dt, steps, command, external_torque=zero_torque):
    """Propagate x0 through `steps` RK4 steps of size dt.

    body, wheels, x0 : may be traced (differentiable).
    dt, steps        : Python numbers (they set the loop length).
    command(t, x)         -> wheel torque command, N elements. Must be JAX-traceable.
    external_torque(t, x) -> body torque, 3 elements. Must be JAX-traceable.
    Both are evaluated at the start of each step and held for the step.

    Returns a dict of arrays, one row per time t[k]: t, q, w, h, u, wheel_speed, H_inertial.
    u[k] is the wheel torque applied over [t[k], t[k+1]], so h[k+1] = h[k] + dt * u[k].
    The last row of u is the limited command at the final state.
    """
    t = jnp.arange(steps + 1) * dt

    def step(xk, tk):
        u_cmd = jnp.asarray(command(tk, xk), dtype=xk.dtype)
        x_next, u = dyn.rk4_step_with_torque(xk, u_cmd, external_torque(tk, xk), dt,
                                             body, wheels)
        return x_next, (xk, u, dyn.momentum_inertial(xk, body, wheels))

    x_last, (X, U, H) = jax.lax.scan(step, x0, t[:-1])
    u_last = dyn.limit_wheel_torque(jnp.asarray(command(t[-1], x_last), dtype=x_last.dtype),
                                    x_last[7:], wheels)
    X = jnp.concatenate([X, x_last[None]])
    U = jnp.concatenate([U, u_last[None]])
    H = jnp.concatenate([H, dyn.momentum_inertial(x_last, body, wheels)[None]])

    return {"t": t, "q": X[:, 0:4], "w": X[:, 4:7], "h": X[:, 7:], "u": U,
            "wheel_speed": X[:, 7:] / wheels.rotor_inertia, "H_inertial": H}


_simulate_jit = jax.jit(simulate, static_argnames=("dt", "steps", "command", "external_torque"))


def run(cfg, command=None, external_torque=None):
    """Run one simulation from the config, compiled with jax.jit.

    command(t, x) -> wheel torque command, N elements. Default: the file profile.
        A controller will use this same interface later.
    external_torque(t, x) -> body torque, 3 elements. Default: zero.

    Both functions must be written with jax.numpy. Each new function object triggers
    one compilation; reuse the same object to reuse the compiled code.
    """
    command = command or open_loop_command(cfg)
    external_torque = external_torque or zero_torque
    return _simulate_jit(cfg.body, cfg.wheels, cfg.sim.x0, dt=cfg.sim.dt,
                         steps=cfg.sim.steps, command=command,
                         external_torque=external_torque)
