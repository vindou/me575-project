"""Model predictive path integral (MPPI) control for a fixed target attitude.

A sampling-based (zero-order) receding-horizon controller with the same plan, prediction
model, cost and torque hold as mpc.py (see horizon.py). At each replanning instant:

  1. Sample K perturbed plans V_k = clip(V + eps_k, -1, 1). Sample 0 is the unperturbed
     plan V. Each eps_k is Gaussian noise N(0, sigma^2), projected onto the range of A^T
     so it changes only the body torque A u. By default the noise is independent in each
     interval; with `noise_knots` it is drawn at fewer points and linearly interpolated.
  2. Roll out every sample with the nonlinear model (jax.vmap) and compute its cost S_k.
  3. Weight w_k = softmax(-(S_k - S_min) / (temperature * (S_median - S_min))), set
     V = sum_k w_k V_k, project V onto the range of A^T, and clip it to [-1, 1].

Design notes (from closed-loop tests on the default 50 deg slew):
- The cost spread is normalized by S_median - S_min so one dimensionless temperature works
  from the start of a slew (costs ~10) to fine pointing (costs ~1e-6).
- Steps 1-3 repeat `iterations` times per replan with sigma shrinking geometrically from
  noise_sigma to final_sigma_fraction * noise_sigma (annealing). The residual sampling
  noise in the applied torque scales with the final sigma; without the shrink it is orders
  of magnitude larger than the torques needed to hold arcsecond pointing.
- Only the first interval of the plan is applied, but every interval adds noise to S_k, so
  a short horizon (2 s by default, vs 5 s for MPC) gives a much better first torque.
- Wheel null space: torque along the null space of A moves no body axis, so the cost
  cannot see it, yet clipping saturated samples creates some. Without the projection of
  V, the warm start repeats it and the wheel momentum drifts until the wheels saturate.

The torque never exceeds u_max (the plan is clipped). The random key is part of the
controller state, so runs are reproducible for a given seed.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import horizon as hz


def mppi_controller(cfg, q_target, *, control_interval_s=0.2, horizon_steps=10, substeps=1,
                    samples=256, noise_knots=None, iterations=8, first_iterations=40,
                    noise_sigma=0.1, final_sigma_fraction=0.01, temperature=0.01, seed=0, rate_weight_s=1.0,
                    w_torque=1e-4, w_terminal=10.0, max_rate_deg_s=None, w_rate_limit=10.0,
                    momentum_warning_frac=None, w_momentum=10.0):
    """Build a stateful MPPI command.

    control_interval_s : replanning period and torque hold time; a multiple of sim dt.
    horizon_steps      : number of control intervals in the prediction horizon.
    samples            : number of sampled plans K per iteration (including the nominal).
    noise_knots        : noise points over the horizon, at most horizon_steps; None uses
                         one per interval (independent noise in time).
    iterations         : sampling iterations per warm-started replan.
    first_iterations   : sampling iterations for the first (cold-started) replan.
    noise_sigma        : initial sampling standard deviation of v = u / u_max.
    final_sigma_fraction : sigma at the last iteration, as a fraction of noise_sigma.
    temperature        : weighting temperature relative to S_median - S_min.
    seed               : random seed for the sampling.
    The cost parameters are those of mpc_controller (horizon.horizon_cost).

    The returned command has an `init_state` and is called as
    command(t, x, state) -> (u_cmd, state), the simulator's stateful interface.
    """
    Tc, hold_steps = hz.hold_steps_for(cfg, control_interval_s)
    N = hz.check_count("horizon_steps", horizon_steps)
    K = hz.check_count("samples", samples)
    if K < 2:
        raise ValueError("samples must be at least 2.")
    knots = N if noise_knots is None else hz.check_count("noise_knots", noise_knots)
    if knots > N:
        raise ValueError("noise_knots must be at most horizon_steps.")
    iterations = hz.check_count("iterations", iterations)
    first_iterations = hz.check_count("first_iterations", first_iterations)
    sigma0 = hz.check_positive("noise_sigma", noise_sigma)
    sigma_end = hz.check_positive("final_sigma_fraction", final_sigma_fraction)
    if sigma_end > 1.0:
        raise ValueError("final_sigma_fraction must be at most 1.")
    temperature = hz.check_positive("temperature", temperature)
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer.")
    seed = int(seed)
    cost, cost_settings = hz.horizon_cost(
        cfg, q_target, control_interval_s=Tc, substeps=substeps, rate_weight_s=rate_weight_s,
        w_torque=w_torque, w_terminal=w_terminal, max_rate_deg_s=max_rate_deg_s,
        w_rate_limit=w_rate_limit, momentum_warning_frac=momentum_warning_frac,
        w_momentum=w_momentum)
    n = cfg.wheels.count
    u_max = cfg.wheels.u_max
    batch_cost = jax.vmap(cost, in_axes=(0, None))
    # Linear interpolation from the knots to the intervals: M[i, j], rows sum to 1.
    interval_pos = np.arange(N, dtype=float)
    knot_pos = np.linspace(0.0, N - 1, knots) if knots > 1 else np.zeros(1)
    M = jnp.asarray(np.stack([np.interp(interval_pos, knot_pos, np.eye(knots)[j])
                              for j in range(knots)], axis=1))
    # Orthogonal projection onto the range of A^T (removes the wheel null space).
    A = np.asarray(cfg.wheels.A)
    P = jnp.asarray(np.linalg.pinv(A) @ A)

    def update(V, x0, key, sigma):
        white = sigma * jax.random.normal(key, (K, knots, n), dtype=V.dtype)
        eps = jnp.einsum("ij,kjw,vw->kiv", M, white, P)
        eps = eps.at[0].set(0.0)                     # keep the current plan as a sample
        Vs = jnp.clip(V + eps, -1.0, 1.0)
        S = batch_cost(Vs, x0)
        S_min = jnp.min(S)
        spread = jnp.maximum(jnp.median(S) - S_min, jnp.finfo(S.dtype).tiny)
        w = jax.nn.softmax(-(S - S_min) / (temperature * spread))
        # Remove the null-space part that clipping adds, then restore the torque limits.
        return jnp.clip(jnp.tensordot(w, Vs, axes=1) @ P, -1.0, 1.0)

    def solve(V, x0, key, n_iter):
        ratio = sigma_end ** (1.0 / max(n_iter - 1, 1))

        def step(i, carry):
            V, key = carry
            key, sub = jax.random.split(key)
            return update(V, x0, sub, sigma0 * ratio**i), key

        return jax.lax.fori_loop(0, n_iter, step, (V, key))

    def replan(x, aux, first):
        V, key = aux
        V, key = jax.lax.cond(first,
                              lambda V, key: solve(V, x, key, first_iterations),
                              lambda V, key: solve(V, x, key, iterations), V, key)
        return u_max * V[0], (hz.shift_plan(V), key)

    init_aux = (jnp.zeros((N, n)), jax.random.key(seed))
    command = hz.receding_horizon(replan, init_aux, hold_steps, n)
    command.design_info = {
        "solver": "annealed MPPI, median-normalized temperature",
        "control_interval_s": Tc, "hold_steps": hold_steps, "horizon_steps": N,
        "horizon_s": N * Tc, "samples": K, "noise_knots": knots, "iterations": iterations,
        "first_iterations": first_iterations, "noise_sigma": sigma0,
        "final_sigma_fraction": sigma_end, "temperature": temperature, "seed": seed,
        **cost_settings,
    }
    return command
