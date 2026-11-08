"""Checks of the stateful simulator interface, nonlinear shooting MPC, and MPPI."""
import jax.numpy as jnp
import numpy as np
import pytest

from cubesat_sim import run
from cubesat_sim.controller import CONTROLLERS
from cubesat_sim.mpc import mpc_controller
from cubesat_sim.mppi import mppi_controller
from cubesat_sim.quaternion import angle_between
from helpers import make


def axis_angle(axis, angle_deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    half = np.radians(angle_deg) / 2.0
    return jnp.asarray(np.append(axis * np.sin(half), np.cos(half)))


# Small, fast settings for the tests (horizon 3 s, replan every 0.2 s).
FAST = dict(control_interval_s=0.2, horizon_steps=15, iterations=8, first_iterations=30)


def test_registered():
    assert CONTROLLERS["mpc"] is mpc_controller


def test_stateful_command_matches_stateless(cfg):
    """A stateful command that ignores its state reproduces the stateless run."""
    c = make(cfg, dt=0.05, t_end=5.0)
    gain = jnp.array([1e-3, -1e-3, 1e-3, -1e-3])

    def stateless(t, x):
        return gain * jnp.sin(t)

    def stateful(t, x, k):
        return gain * jnp.sin(t), k + 1

    stateful.init_state = jnp.asarray(0)
    a, b = run(c, command=stateless), run(c, command=stateful)
    for key in a:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]))


def test_stateful_command_carries_state(cfg):
    """The state persists between steps: a step counter drives the command."""
    c = make(cfg, dt=0.1, t_end=1.0)

    def counter(t, x, k):
        return jnp.full(4, 1e-4) * (k == 3), k + 1

    counter.init_state = jnp.asarray(0)
    u = np.asarray(run(c, command=counter)["u"])
    np.testing.assert_allclose(u[3], 1e-4)
    assert np.all(u[np.arange(len(u)) != 3] == 0.0)


def test_mpc_reaches_target(cfg):
    """A 20 deg slew from rest settles inside the 60 arcsec band within 30 s."""
    q_target = axis_angle([1.0, -2.0, 3.0], 20.0)
    c = make(cfg, dt=0.02, t_end=30.0)
    res = run(c, command=mpc_controller(c, q_target, **FAST))
    u = np.asarray(res["u"])
    assert np.abs(u).max() <= float(cfg.wheels.u_max) * (1 + 1e-12)
    final_error_arcsec = np.degrees(float(angle_between(res["q"][-1], q_target))) * 3600
    assert final_error_arcsec < 60.0
    assert np.degrees(np.linalg.norm(res["w"][-1])) < 0.01


def test_mpc_holds_command_between_replans(cfg):
    c = make(cfg, dt=0.02, t_end=2.0)
    res = run(c, command=mpc_controller(c, axis_angle([0, 0, 1], 10.0), **FAST))
    u = np.asarray(res["u"])[:-1].reshape(-1, 10, 4)       # 10 sim steps per 0.2 s interval
    np.testing.assert_array_equal(u, np.repeat(u[:, :1], 10, axis=1))


def test_mpc_soft_rate_limit(cfg):
    """With the rate penalty the slew stays near the limit; without it, much faster."""
    q_target = axis_angle([0, 0, 1], 30.0)
    c = make(cfg, dt=0.02, t_end=20.0)
    free = run(c, command=mpc_controller(c, q_target, **FAST))
    limited = run(c, command=mpc_controller(c, q_target, max_rate_deg_s=0.5, **FAST))
    peak = lambda r: np.degrees(np.linalg.norm(np.asarray(r["w"]), axis=1).max())
    assert peak(free) > 2.0
    assert peak(limited) < 0.55


@pytest.mark.parametrize("change, message", [
    (dict(control_interval_s=0.015), "whole multiple"),
    (dict(control_interval_s=0.0), "positive"),
    (dict(horizon_steps=0), "positive integer"),
    (dict(horizon_steps=2.5), "positive integer"),
    (dict(iterations=True), "positive integer"),
    (dict(w_torque=-1.0), "nonnegative"),
    (dict(max_rate_deg_s=0.0), "positive"),
    (dict(momentum_warning_frac=1.5), "at most 1"),
])
def test_mpc_rejects_bad_parameters(cfg, change, message):
    with pytest.raises(ValueError, match=message):
        mpc_controller(cfg, axis_angle([0, 0, 1], 10.0), **change)


# --- MPPI --------------------------------------------------------------------------------

MPPI_FAST = dict(samples=128, iterations=6, first_iterations=20)


def test_mppi_registered():
    assert CONTROLLERS["mppi"] is mppi_controller


def test_mppi_reaches_target(cfg):
    """A 20 deg slew from rest settles inside the 60 arcsec band within 30 s."""
    q_target = axis_angle([1.0, -2.0, 3.0], 20.0)
    c = make(cfg, dt=0.02, t_end=30.0)
    res = run(c, command=mppi_controller(c, q_target, **MPPI_FAST))
    assert np.abs(np.asarray(res["u"])).max() <= float(cfg.wheels.u_max) * (1 + 1e-12)
    final_error_arcsec = np.degrees(float(angle_between(res["q"][-1], q_target))) * 3600
    assert final_error_arcsec < 60.0
    assert np.degrees(np.linalg.norm(res["w"][-1])) < 0.01


def test_mppi_no_null_space_drift(cfg):
    """The wheel momentum stays in the range needed for the slew (no null-space ramp)."""
    q_target = axis_angle([0, 0, 1], 30.0)
    c = make(cfg, dt=0.02, t_end=30.0)
    res = run(c, command=mppi_controller(c, q_target, **MPPI_FAST))
    A = np.asarray(cfg.wheels.A)
    null = np.linalg.svd(A)[2][-1]                         # unit null vector of A
    h_null = np.asarray(res["h"]) @ null
    assert np.abs(h_null).max() < 0.05 * float(cfg.wheels.h_max)


def test_mppi_reproducible_for_seed(cfg):
    c = make(cfg, dt=0.02, t_end=2.0)
    q_target = axis_angle([0, 0, 1], 10.0)
    a = run(c, command=mppi_controller(c, q_target, seed=3, **MPPI_FAST))["u"]
    b = run(c, command=mppi_controller(c, q_target, seed=3, **MPPI_FAST))["u"]
    d = run(c, command=mppi_controller(c, q_target, seed=4, **MPPI_FAST))["u"]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(d))


@pytest.mark.parametrize("change, message", [
    (dict(samples=1), "at least 2"),
    (dict(noise_knots=11), "at most horizon_steps"),
    (dict(noise_sigma=0.0), "positive"),
    (dict(final_sigma_fraction=2.0), "at most 1"),
    (dict(temperature=-1.0), "positive"),
    (dict(seed=1.5), "integer"),
    (dict(control_interval_s=0.015), "whole multiple"),
])
def test_mppi_rejects_bad_parameters(cfg, change, message):
    with pytest.raises(ValueError, match=message):
        mppi_controller(cfg, axis_angle([0, 0, 1], 10.0), **change)
