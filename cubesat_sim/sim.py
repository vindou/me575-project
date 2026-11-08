"""Simulation loop: propagate the initial conditions through time.

simulate_plant() is the plant-agnostic core: a jax.lax.scan over a plant step function
with a stateless or stateful command. simulate() and run() are the attitude wrappers;
thruster1d.py has the wrappers for the 1D thruster.

simulate() is a pure function. Call it inside your own jax.jit / jax.grad / jax.vmap to
differentiate through a whole run or to batch runs. run() compiles simulate() for one config.
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


def simulate_plant(step, limit, observe, x0, dt, steps, command):
    """Propagate x0 through `steps` steps of size dt with a plant and a command.

    step(t, x, u_cmd)  -> (x_next, u_applied): one plant step; the plant applies its limits.
    limit(t, x, u_cmd) -> u_applied: the limited command at a state (used for the last row).
    observe(t, x)      -> dict of extra outputs at the start of each step (may be empty).
    command(t, x) -> u_cmd, or a stateful command with an `init_state` attribute (a JAX
        pytree) called as command(t, x, state) -> (u_cmd, new_state). The state is carried
        from one step to the next and is not returned. The command is evaluated at the
        start of each step and held for the step.

    Returns (t, X, U_cmd, U, obs): one row per time t[k]. U[k] is applied over
    [t[k], t[k+1]]; the last rows of U_cmd and U are the command at the final state.
    """
    t = jnp.arange(steps + 1) * dt
    stateful = hasattr(command, "init_state")

    def evaluate(tk, xk, ck):
        if stateful:
            u_cmd, ck = command(tk, xk, ck)
        else:
            u_cmd = command(tk, xk)
        return jnp.asarray(u_cmd, dtype=xk.dtype), ck

    def scan_step(carry, tk):
        xk, ck = carry
        u_cmd, ck = evaluate(tk, xk, ck)
        x_next, u = step(tk, xk, u_cmd)
        return (x_next, ck), (xk, u_cmd, u, observe(tk, xk))

    c0 = command.init_state if stateful else None
    (x_last, c_last), (X, U_cmd, U, obs) = jax.lax.scan(scan_step, (x0, c0), t[:-1])
    u_cmd_last = evaluate(t[-1], x_last, c_last)[0]
    u_last = limit(t[-1], x_last, u_cmd_last)
    obs_last = observe(t[-1], x_last)

    def append(rows, last):
        return jnp.concatenate([rows, last[None]])

    X, U_cmd, U = append(X, x_last), append(U_cmd, u_cmd_last), append(U, u_last)
    obs = jax.tree_util.tree_map(append, obs, obs_last)
    return t, X, U_cmd, U, obs


def simulate(body, wheels, x0, dt, steps, command, external_torque=zero_torque):
    """Propagate x0 through `steps` RK4 steps of size dt.

    body, wheels, x0 : may be traced (differentiable).
    dt, steps        : Python numbers (they set the loop length).
    command(t, x)         -> wheel torque command T_w (torque of each wheel on the body),
                             N elements. Must be JAX-traceable.
        A stateful command (for example MPC with warm starts) instead has an `init_state`
        attribute and is called as command(t, x, state) -> (u, new_state); see
        simulate_plant.
    external_torque(t, x) -> body torque, 3 elements. Must be JAX-traceable.
    Both are evaluated at the start of each step and held for the step.

    Returns a dict of arrays, one row per time t[k]: t, q, w, h, u, wheel_speed, H_inertial.
    u[k] is the wheel torque applied over [t[k], t[k+1]], so h[k+1] = h[k] - dt * u[k].
    The last row of u is the limited command at the final state.
    """
    def step(tk, xk, u_cmd):
        return dyn.rk4_step_with_torque(xk, u_cmd, external_torque(tk, xk), dt, body, wheels)

    def limit(tk, xk, u_cmd):
        return dyn.limit_wheel_torque(u_cmd, xk[7:], wheels)

    def observe(tk, xk):
        return {"H": dyn.momentum_inertial(xk, body, wheels)}

    t, X, _, U, obs = simulate_plant(step, limit, observe, x0, dt, steps, command)
    H = obs["H"]

    return {"t": t, "q": X[:, 0:4], "w": X[:, 4:7], "h": X[:, 7:], "u": U,
            "wheel_speed": X[:, 7:] / wheels.rotor_inertia, "H_inertial": H}


_simulate_jit = jax.jit(simulate, static_argnames=("dt", "steps", "command", "external_torque"))


def run(cfg, command=None, external_torque=None):
    """Run one simulation from the config, compiled with jax.jit.

    command(t, x) -> wheel torque command, N elements. Default: the file profile.
        Stateful commands (see simulate) are also accepted.
    external_torque(t, x) -> body torque, 3 elements. Default: zero.

    Both functions must be written with jax.numpy. Each new function object triggers
    one compilation; reuse the same object to reuse the compiled code.
    """
    command = command or open_loop_command(cfg)
    external_torque = external_torque or zero_torque
    return _simulate_jit(cfg.body, cfg.wheels, cfg.sim.x0, dt=cfg.sim.dt,
                         steps=cfg.sim.steps, command=command,
                         external_torque=external_torque)
