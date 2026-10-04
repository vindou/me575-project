"""Wheel torque and momentum limits over whole runs."""
import jax.numpy as jnp
import numpy as np
import pytest

from cubesat_sim import run
from helpers import body_energy, make


def full_positive(t, x):
    return jnp.array([1.0, 0.0, 0.0, 0.0])


def full_negative(t, x):
    return jnp.array([-1.0, 0.0, 0.0, 0.0])


def reverse_at_10s(t, x):
    return jnp.where(t < 10.0, 1.0, -1.0) * jnp.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def saturated(cfg):
    return {k: np.asarray(v) for k, v in run(make(cfg, t_end=20.0), command=full_positive).items()}


def test_torque_is_clipped(cfg, saturated):
    u_max = float(cfg.wheels.u_max)
    assert saturated["u"].max() <= u_max
    assert saturated["u"][0, 0] == u_max                     # before saturation: exactly u_max
    assert np.all(saturated["u"][:, 1:] == 0.0)


def test_wheel_saturates_at_the_expected_time(cfg, saturated):
    # 0.015 N·m·s at 0.004 N·m: the wheel is saturated after 3.75 s.
    h_max, u_max, dt = float(cfg.wheels.h_max), float(cfg.wheels.u_max), cfg.sim.dt
    h = saturated["h"][:, 0]
    k_sat = np.argmax(h >= h_max)
    assert 3.75 - 1e-9 <= saturated["t"][k_sat] <= 3.75 + dt + 1e-9   # within one step
    assert h.max() <= h_max + u_max * dt                     # overshoot of at most one step
    assert np.all(saturated["u"][k_sat + 1:, 0] == 0.0)      # then no more torque
    assert saturated["u"][-1, 0] == 0.0


def test_logged_torque_matches_the_wheel_momentum_change(cfg, saturated):
    """h[k+1] = h[k] + dt u[k] for every step, including the step that reaches the limit."""
    dh = np.diff(saturated["h"], axis=0)
    assert np.allclose(dh, cfg.sim.dt * saturated["u"][:-1], rtol=0, atol=1e-18)


def test_body_energy_is_constant_after_saturation(cfg, saturated):
    E = body_energy(saturated["w"][400:], cfg.body.J)     # t >= 4 s: all wheel torques zero
    assert np.abs(E - E[0]).max() < 1e-10 * E[0]


def test_negative_saturation(cfg):
    res = run(make(cfg, t_end=10.0), command=full_negative)
    h_max, u_max, dt = float(cfg.wheels.h_max), float(cfg.wheels.u_max), cfg.sim.dt
    assert np.asarray(res["u"]).min() >= -u_max
    assert np.asarray(res["h"])[:, 0].min() >= -h_max - u_max * dt
    assert float(res["h"][-1, 0]) == pytest.approx(-h_max, abs=u_max * dt)


def test_reversed_torque_desaturates_then_saturates_the_other_way(cfg):
    res = run(make(cfg, t_end=20.0), command=reverse_at_10s)
    h_max, u_max, dt = float(cfg.wheels.h_max), float(cfg.wheels.u_max), cfg.sim.dt
    h, t = np.asarray(res["h"])[:, 0], np.asarray(res["t"])
    assert h[t == 10.0][0] == pytest.approx(h_max, abs=u_max * dt)
    # Leaves +h_max at -u_max: crosses zero 3.75 s after the reversal.
    k_zero = np.argmax((t > 10.0) & (h <= 0.0))
    assert 13.75 - 1e-9 <= t[k_zero] <= 13.75 + dt + 1e-9
    assert h[-1] == pytest.approx(-h_max, abs=u_max * dt)
    assert np.allclose(np.diff(h), dt * np.asarray(res["u"])[:-1, 0], rtol=0, atol=1e-18)
