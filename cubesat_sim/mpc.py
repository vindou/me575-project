"""Nonlinear shooting MPC for a fixed target attitude.

At each replanning instant the controller optimizes a piecewise-constant sequence of
wheel torques over a finite horizon. It predicts the motion with the plant's own
nonlinear equations and differentiates the predicted cost (horizon.py) with jax.grad.
The first torque of the plan is held until the next replanning instant (receding
horizon), and the shifted plan warm-starts the next solve.

Decision variables z, shape (horizon, N), map to wheel torques u = u_max * tanh(z), so
every torque stays inside +/- u_max without a constrained solver. optax L-BFGS with a
line search minimizes the cost. (A first-order method such as Adam takes steps of a
fixed size in v = u / u_max and cannot make the ~1e-3 corrections needed near the target.)
"""
import jax
import jax.numpy as jnp
import optax

from . import horizon as hz

# Warm starts are clipped to |z| <= Z_CLIP (tanh = 0.995) so saturated torques keep a
# usable gradient and can come off the limit in the next solve.
Z_CLIP = 3.0


def mpc_controller(cfg, q_target, *, control_interval_s=0.2, horizon_steps=25, substeps=1,
                   iterations=10, first_iterations=50, rate_weight_s=1.0, w_torque=1e-4,
                   w_terminal=10.0, max_rate_deg_s=None, w_rate_limit=10.0,
                   momentum_warning_frac=None, w_momentum=10.0):
    """Build a stateful nonlinear shooting MPC command.

    control_interval_s : replanning period and torque hold time; a multiple of sim dt.
    horizon_steps      : number of control intervals in the prediction horizon.
    substeps           : RK4 steps per control interval in the prediction model.
    iterations         : L-BFGS iterations per warm-started solve.
    first_iterations   : L-BFGS iterations for the first (cold-started) solve.
    rate_weight_s      : weight on body rate, as a time constant in seconds.
    w_torque           : weight on normalized torque, per wheel and interval.
    w_terminal         : multiplier on the last predicted stage.
    max_rate_deg_s     : soft body-rate limit; None disables the term.
    momentum_warning_frac : soft wheel momentum limit as a fraction of h_max; None disables it.

    The returned command has an `init_state` and is called as
    command(t, x, state) -> (u_cmd, state), the simulator's stateful interface.
    """
    Tc, hold_steps = hz.hold_steps_for(cfg, control_interval_s)
    N = hz.check_count("horizon_steps", horizon_steps)
    iterations = hz.check_count("iterations", iterations)
    first_iterations = hz.check_count("first_iterations", first_iterations)
    cost, cost_settings = hz.horizon_cost(
        cfg, q_target, control_interval_s=Tc, substeps=substeps, rate_weight_s=rate_weight_s,
        w_torque=w_torque, w_terminal=w_terminal, max_rate_deg_s=max_rate_deg_s,
        w_rate_limit=w_rate_limit, momentum_warning_frac=momentum_warning_frac,
        w_momentum=w_momentum)
    u_max = cfg.wheels.u_max
    optimizer = optax.lbfgs()

    def solve(Z, x0, n_iter):
        def f(Z):
            return cost(jnp.tanh(Z), x0)

        value_and_grad = optax.value_and_grad_from_state(f)

        def step(_, carry):
            Z, opt_state = carry
            value, grad = value_and_grad(Z, state=opt_state)
            updates, opt_state = optimizer.update(grad, opt_state, Z, value=value,
                                                  grad=grad, value_fn=f)
            return optax.apply_updates(Z, updates), opt_state

        Z, _ = jax.lax.fori_loop(0, n_iter, step, (Z, optimizer.init(Z)))
        return Z

    def replan(x, Z, first):
        Z = jax.lax.cond(first,
                         lambda Z: solve(Z, x, first_iterations),
                         lambda Z: solve(Z, x, iterations), Z)
        u_cmd = u_max * jnp.tanh(Z[0])
        return u_cmd, jnp.clip(hz.shift_plan(Z), -Z_CLIP, Z_CLIP)

    n = cfg.wheels.count
    command = hz.receding_horizon(replan, jnp.zeros((N, n)), hold_steps, n)
    command.design_info = {
        "solver": "optax.lbfgs, u = u_max * tanh(z)",
        "control_interval_s": Tc, "hold_steps": hold_steps, "horizon_steps": N,
        "horizon_s": N * Tc, "iterations": iterations, "first_iterations": first_iterations,
        **cost_settings,
    }
    return command
