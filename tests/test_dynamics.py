"""Equations of motion at the level of one state derivative (no integration error).

Random states and random non-diagonal inertias; wheel torques and momenta inside the limits.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given

from cubesat_sim import dynamics as dyn
from cubesat_sim import quaternion as quat
from cubesat_sim.config import BodyParams, WheelParams
from helpers import inertias, states, vectors

deriv = jax.jit(dyn.state_derivative)
limit = jax.jit(dyn.limit_wheel_torque)


def wheel_torques():
    return vectors(4, 0.0039)


def body(cfg, J):
    return dataclasses.replace(cfg.body, J=jnp.asarray(J))


@given(x=states(), u=wheel_torques(), tau=vectors(3, 1e-3), J=inertias())
def test_body_momentum_rate_is_gyroscopic_plus_external(cfg, x, u, tau, J):
    """d/dt (J w + A h) = -w x (J w + A h) + tau, in the body frame."""
    xd = deriv(x, u, tau, body(cfg, J), cfg.wheels)
    _, w, h = dyn.split(x)
    _, w_dot, h_dot = dyn.split(xd)
    A = cfg.wheels.A
    H = J @ w + A @ h
    assert np.allclose(J @ w_dot + A @ h_dot, -np.cross(w, H) + tau, rtol=1e-10, atol=1e-15)


@given(x=states(), u=wheel_torques(), tau=vectors(3, 1e-3), J=inertias())
def test_inertial_momentum_rate_is_external_torque(cfg, x, u, tau, J):
    """d/dt R(q) (J w + A h) = R(q) tau. Checks the kinematics and dynamics together."""
    b = body(cfg, J)
    xd = deriv(x, u, tau, b, cfg.wheels)
    _, H_dot = jax.jvp(lambda s: dyn.momentum_inertial(s, b, cfg.wheels), (x,), (xd,))
    assert np.allclose(H_dot, quat.rotation_matrix(x[:4]) @ tau, rtol=1e-9, atol=1e-15)


@given(x=states(), u=wheel_torques(), tau=vectors(3, 1e-3), J=inertias())
def test_body_energy_rate_is_the_power_of_applied_torques(cfg, x, u, tau, J):
    """d/dt (w^T J w / 2) = w^T (-A u + tau): the gyroscopic term does no work."""
    xd = deriv(x, u, tau, body(cfg, J), cfg.wheels)
    w, w_dot = x[4:7], xd[4:7]
    power = w @ (-cfg.wheels.A @ u + tau)
    assert float(w @ J @ w_dot) == pytest.approx(float(power), rel=1e-9, abs=1e-15)


@given(x=states(), u=wheel_torques())
def test_quaternion_rate_and_wheel_rate(cfg, x, u):
    xd = deriv(x, u, jnp.zeros(3), cfg.body, cfg.wheels)
    assert np.allclose(xd[:4], quat.kinematics(x[:4], x[4:7]), atol=1e-15)
    assert np.allclose(xd[7:], u, atol=0.0)                  # inside the limits: h_dot = u


def test_euler_equations_by_hand(cfg):
    """Torque-free, no wheel momentum, diagonal J: the classical Euler equations."""
    Jx, Jy, Jz = np.diag(cfg.body.J)
    w = np.array([0.1, -0.2, 0.3])
    x = jnp.concatenate([jnp.array([0.0, 0.0, 0.0, 1.0]), w, jnp.zeros(4)])
    w_dot = deriv(x, jnp.zeros(4), jnp.zeros(3), cfg.body, cfg.wheels)[4:7]
    expected = [(Jy - Jz) / Jx * w[1] * w[2],
                (Jz - Jx) / Jy * w[2] * w[0],
                (Jx - Jy) / Jz * w[0] * w[1]]
    assert np.allclose(w_dot, expected, rtol=1e-12)


@pytest.mark.parametrize("wheel", range(4))
def test_wheel_torque_reacts_on_the_body(cfg, wheel):
    """Motor torque u on wheel i puts -u a_i on the body (a_i = spin axis)."""
    u = jnp.zeros(4).at[wheel].set(1e-3)
    x = jnp.concatenate([jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.zeros(7)])
    w_dot = deriv(x, u, jnp.zeros(3), cfg.body, cfg.wheels)[4:7]
    assert np.allclose(cfg.body.J @ w_dot, -1e-3 * cfg.wheels.A[:, wheel], rtol=1e-12)


U_MAX, H_MAX = 0.004, 0.015
LIMIT_CASES = [
    # u_cmd,   h,       expected
    (0.001,    0.0,     0.001),      # inside both limits
    (-0.001,   0.01,    -0.001),
    (0.01,     0.0,     U_MAX),      # torque clipped
    (-0.01,    0.0,     -U_MAX),
    (0.001,    H_MAX,   0.0),        # exactly at the momentum limit, same direction: blocked
    (0.001,    0.02,    0.0),        # beyond the limit
    (0.01,     H_MAX,   0.0),        # clipped, then blocked
    (-0.001,   -H_MAX,  0.0),
    (-0.001,   H_MAX,   -0.001),     # opposite direction: allowed (desaturates)
    (0.001,    -H_MAX,  0.001),
    (-0.01,    H_MAX,   -U_MAX),
    (0.0,      H_MAX,   0.0),
]


@pytest.mark.parametrize("u_cmd, h, expected", LIMIT_CASES)
def test_wheel_limits(u_cmd, h, expected):
    wheels = WheelParams(jnp.eye(3, 1), jnp.asarray(U_MAX), jnp.asarray(H_MAX), jnp.asarray(1e-5))
    u = limit(jnp.array([u_cmd]), jnp.array([h]), wheels)
    assert float(u[0]) == expected


def test_wheel_limits_act_per_wheel(cfg):
    u_cmd = jnp.array([0.01, 0.001, -0.001, 0.001])
    h = jnp.array([0.0, H_MAX, H_MAX, 0.0])
    assert np.array_equal(limit(u_cmd, h, cfg.wheels), [U_MAX, 0.0, -0.001, 0.001])


def test_body_params_mass_is_not_used_by_the_dynamics(cfg):
    x = jnp.concatenate([jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.array([0.1, -0.2, 0.3]), jnp.zeros(4)])
    heavy = BodyParams(cfg.body.mass * 10, cfg.body.J)
    a = deriv(x, jnp.ones(4) * 1e-3, jnp.zeros(3), cfg.body, cfg.wheels)
    b = deriv(x, jnp.ones(4) * 1e-3, jnp.zeros(3), heavy, cfg.wheels)
    assert np.array_equal(a, b)
