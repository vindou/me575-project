"""JAX behaviour: compilation, gradients, batching, precision."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from cubesat_sim import run, simulate
from cubesat_sim.config import wheel_axes
from cubesat_sim.quaternion import angle_between
from cubesat_sim.sim import _simulate_jit, open_loop_command
from helpers import make

AZIMUTHS = (0.0, 90.0, 180.0, 270.0)


def no_wheel_torque(t, x):
    return jnp.zeros(4)


def x0_tumbling():
    q0 = jnp.array([0.1, -0.2, 0.3, 0.9])
    return jnp.concatenate([q0 / jnp.linalg.norm(q0), jnp.array([0.02, -0.03, 0.015]),
                            jnp.array([0.004, -0.002, 0.001, 0.003])])


def test_run_matches_uncompiled_simulate(cfg):
    x0 = x0_tumbling()
    c = make(cfg, t_end=0.3, q0=x0[:4], w0=x0[4:7], h0=x0[7:])   # eager mode is slow
    cmd = lambda t, x: 1e-4 * jnp.array([jnp.sin(t), 1.0, -0.5, 0.0])
    compiled = run(c, command=cmd)
    with jax.disable_jit():
        eager = simulate(c.body, c.wheels, c.sim.x0, c.sim.dt, c.sim.steps, cmd)
    for k in compiled:
        assert np.allclose(compiled[k], eager[k], rtol=1e-12, atol=1e-18), k


def test_same_profile_does_not_recompile(cfg):
    run(cfg)
    before = _simulate_jit._cache_size()
    run(cfg)                                   # new OpenLoopCommand object, equal by value
    run(cfg, command=open_loop_command(cfg))
    assert _simulate_jit._cache_size() == before


def test_outputs_are_float64(cfg):
    res = run(make(cfg, t_end=1.0))
    assert all(v.dtype == jnp.float64 for v in res.values())


def test_gradients_match_finite_differences(cfg):
    """Forward and reverse mode against central differences, through the whole scan,
    with respect to inertia, wheel skew angle, initial state and a feedback gain."""

    @jax.jit
    def final_state(J, skew, x0, gain):
        body = dataclasses.replace(cfg.body, J=J)
        wheels = dataclasses.replace(cfg.wheels, A=wheel_axes(skew, AZIMUTHS))

        def command(t, x):                     # rate damping plus a bias, inside the limits
            return gain * wheels.A.T @ x[4:7] + 1e-4 * jnp.sin(t)

        r = simulate(body, wheels, x0, 0.05, 100, command)
        return jnp.concatenate([r["q"][-1], r["w"][-1], r["h"][-1]])

    J = cfg.body.J + 0.005 * jnp.array([[0.0, 1.0, 0.5], [1.0, 0.0, -0.3], [0.5, -0.3, 0.0]])
    check_grads(final_state, (J, 54.7356, x0_tumbling(), 0.05), order=1, modes=("fwd", "rev"))


def test_gradient_of_slew_angle_matches_closed_form(cfg):
    """theta = scale * alpha * (10 s)^2 with alpha proportional to 1/J_xx, so
    d theta / d scale = theta / scale and d theta / d J_xx = -theta / J_xx."""
    cmd = open_loop_command(cfg)

    def angle(scale, J):
        body = dataclasses.replace(cfg.body, J=J)
        r = simulate(body, cfg.wheels, cfg.sim.x0, cfg.sim.dt, cfg.sim.steps,
                     lambda t, x: scale * cmd(t, x))
        return angle_between(r["q"][0], r["q"][-1])

    theta = float(angle(1.0, cfg.body.J))
    g_scale, g_J = jax.jit(jax.grad(angle, argnums=(0, 1)))(1.0, cfg.body.J)
    assert float(g_scale) == pytest.approx(theta, rel=1e-8)
    assert float(g_J[0, 0]) == pytest.approx(-theta / float(cfg.body.J[0, 0]), rel=1e-8)
    assert abs(float(g_J[1, 1])) < 1e-10 and abs(float(g_J[2, 2])) < 1e-10


def test_gradients_are_finite_while_limits_are_active(cfg):
    def final_rate(J):
        body = dataclasses.replace(cfg.body, J=J)
        r = simulate(body, cfg.wheels, cfg.sim.x0, 0.05, 200,
                     lambda t, x: jnp.array([1.0, -1.0, 0.0, 0.0]))   # clipped, then saturated
        return jnp.sum(r["w"][-1] ** 2)

    g = jax.jit(jax.grad(final_rate))(cfg.body.J)
    assert np.all(np.isfinite(g))


def test_forward_and_reverse_jacobians_agree(cfg):
    def final_w(w0):
        x0 = jnp.concatenate([cfg.sim.q0, w0, cfg.sim.h0])
        return simulate(cfg.body, cfg.wheels, x0, 0.05, 100, no_wheel_torque)["w"][-1]

    w0 = jnp.array([0.02, -0.03, 0.015])
    jf = jax.jit(jax.jacfwd(final_w))(w0)
    jr = jax.jit(jax.jacrev(final_w))(w0)
    assert np.all(np.isfinite(jf))
    assert np.allclose(jf, jr, rtol=1e-10, atol=1e-14)


def test_vmap_matches_a_loop(cfg):
    w0s = jnp.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, -0.02, 0.01], [0.03, 0.01, -0.02]])
    cmd = open_loop_command(cfg)

    def final(w0):
        x0 = jnp.concatenate([cfg.sim.q0, w0, cfg.sim.h0])
        r = simulate(cfg.body, cfg.wheels, x0, 0.05, 600, cmd)
        return jnp.concatenate([r["q"][-1], r["w"][-1], r["h"][-1]])

    batched = jax.jit(jax.vmap(final))(w0s)
    looped = jnp.stack([jax.jit(final)(w) for w in w0s])
    assert np.allclose(batched, looped, rtol=1e-12, atol=1e-16)
