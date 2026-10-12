"""Whole runs compared with known physical results."""
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cubesat_sim import run
from cubesat_sim import quaternion as quat
from cubesat_sim.quaternion import angle_between
from helpers import body_energy, make, with_inertia

# A non-diagonal inertia (principal moments 0.12, 0.09, 0.05, rotated) for tumbling cases.
_R = Rotation.from_rotvec([0.4, -0.3, 0.2]).as_matrix()
J_SKEWED = _R @ np.diag([0.12, 0.09, 0.05]) @ _R.T


def no_wheel_torque(t, x):
    return jnp.zeros(4)


def mixed_wheel_torque(t, x):
    return 1e-4 * jnp.array([jnp.sin(0.1 * t), 1.0, -0.5, jnp.cos(0.2 * t)])


@pytest.mark.parametrize("J", [None, J_SKEWED], ids=["principal", "skewed"])
def test_momentum_is_conserved_while_tumbling_with_wheel_torques(cfg, J):
    c = make(cfg, t_end=120.0, w0=jnp.array([0.02, -0.03, 0.015]),
             h0=jnp.array([0.004, -0.002, 0.001, 0.003]))
    if J is not None:
        c = with_inertia(c, J)
    res = run(c, command=mixed_wheel_torque)
    H = np.asarray(res["H_inertial"])
    assert np.linalg.norm(H - H[0], axis=1).max() < 1e-10 * np.linalg.norm(H[0])
    assert np.allclose(np.linalg.norm(res["q"], axis=1), 1.0, atol=1e-14)


@pytest.mark.parametrize("J", [None, J_SKEWED], ids=["principal", "skewed"])
def test_body_energy_is_conserved_without_wheel_torque(cfg, J):
    """With u = 0 the wheels only add a gyroscopic term, which does no work on the body."""
    c = make(cfg, t_end=120.0, w0=jnp.array([0.05, -0.08, 0.1]),
             h0=jnp.array([0.004, -0.002, 0.001, 0.003]))
    if J is not None:
        c = with_inertia(c, J)
    res = run(c, command=no_wheel_torque)
    E = body_energy(res["w"], c.body.J)
    assert np.abs(E - E[0]).max() < 1e-10 * E[0]
    H_body = np.asarray(res["w"]) @ np.asarray(c.body.J).T + np.asarray(res["h"]) @ np.asarray(c.wheels.A).T
    H_mag = np.linalg.norm(H_body, axis=1)
    assert np.abs(H_mag - H_mag[0]).max() < 1e-10 * H_mag[0]


@pytest.mark.parametrize("h_each", [0.0, 0.003], ids=["rigid", "gyrostat"])
def test_axisymmetric_body_matches_closed_form(cfg, h_each):
    """J = diag(a, a, c), equal wheel momentum (so A h is along z), no torque.

    w_z is constant and (w_x, w_y) rotates at Omega = ((a - c) w_z - h_z) / a.
    """
    a, c_ = 0.1, 0.05
    w0 = np.array([0.02, -0.01, 0.03])
    h0 = np.full(4, h_each)
    hz = float((cfg.wheels.A @ h0)[2])
    c = with_inertia(make(cfg, t_end=120.0, w0=jnp.asarray(w0), h0=jnp.asarray(h0)),
                     np.diag([a, a, c_]))
    res = run(c, command=no_wheel_torque)
    t = np.asarray(res["t"])
    Om = ((a - c_) * w0[2] - hz) / a
    expected = np.column_stack([w0[0] * np.cos(Om * t) + w0[1] * np.sin(Om * t),
                                -w0[0] * np.sin(Om * t) + w0[1] * np.cos(Om * t),
                                np.full_like(t, w0[2])])
    assert np.allclose(res["w"], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("axis, stable", [(0, True), (1, False), (2, True)], ids=["x", "y", "z"])
def test_intermediate_axis_is_unstable(cfg, axis, stable):
    """Default J has Jx > Jy > Jz: spin about y tumbles, spin about x or z stays."""
    w0 = np.full(3, 1e-4)
    w0[axis] = 0.5
    res = run(make(cfg, t_end=60.0, w0=jnp.asarray(w0)), command=no_wheel_torque)
    w = np.asarray(res["w"])
    other = np.delete(w, axis, axis=1)
    if stable:
        assert np.abs(other).max() < 10 * 1e-4
        assert w[:, axis].min() > 0.0
    else:
        assert w[:, axis].min() < 0.0                       # the spin axis flips
    E = body_energy(w, cfg.body.J)
    assert np.abs(E - E[0]).max() < 1e-9 * E[0]


def test_default_profile_matches_constant_acceleration(cfg):
    """Body torque A u = [-4 cos(b) cos(45 deg) 1e-4, 0, 0], so w_x is a triangle and
    theta = alpha * (10 s)^2."""
    tau = np.asarray(cfg.wheels.A @ cfg.open_loop_command[0].u)
    assert np.allclose(tau[1:], 0.0, atol=1e-18)
    tau_x = float(tau[0])
    alpha = tau_x / float(cfg.body.J[0, 0])
    res = run(cfg)
    t = np.asarray(res["t"])
    w_x = alpha * np.clip(t - 5.0, 0.0, 10.0) - alpha * np.clip(t - 15.0, 0.0, 10.0)
    assert np.allclose(res["w"][:, 0], w_x, rtol=0, atol=1e-14)
    assert np.allclose(res["w"][:, 1:], 0.0, atol=1e-16)
    rotvec = Rotation.from_quat(np.asarray(res["q"][-1])).as_rotvec()
    assert rotvec[0] == pytest.approx(alpha * 100.0, rel=1e-9)
    assert rotvec[0] < 0.0                                  # rotation about -x
    assert np.allclose(rotvec[1:], 0.0, atol=1e-15)
    assert np.degrees(abs(rotvec[0])) == pytest.approx(11.10, abs=0.01)


def test_spin_about_principal_axis_gives_expected_angle(cfg):
    rate = 0.01
    res = run(make(cfg, t_end=50.0, w0=jnp.array([0.0, 0.0, rate])), command=no_wheel_torque)
    assert np.allclose(res["w"][-1], [0.0, 0.0, rate])
    assert float(angle_between(res["q"][0], res["q"][-1])) == pytest.approx(rate * 50.0, rel=1e-9)


def test_body_fixed_external_torque_without_rotation(cfg):
    tau = jnp.array([1e-5, 0.0, 0.0])
    res = run(make(cfg, t_end=10.0), command=no_wheel_torque, external_torque=lambda t, x: tau)
    # The body rotates about x only, so the body torque is also the inertial torque.
    assert np.allclose(res["H_inertial"], np.outer(res["t"], tau), rtol=1e-9, atol=1e-15)


def test_inertially_fixed_external_torque_while_tumbling(cfg):
    """H_inertial(t) = H0 + tau_I t. The torque is held over each step (zero-order hold),
    which gives an error of order |w| dt / 2 relative to tau_I t."""
    tau_I = jnp.array([2e-6, -1e-6, 3e-6])
    c = with_inertia(make(cfg, t_end=60.0, w0=jnp.array([0.02, -0.03, 0.015]),
                          h0=jnp.array([0.004, -0.002, 0.001, 0.003])), J_SKEWED)
    ext = lambda t, x: quat.rotation_matrix(x[:4]).T @ tau_I
    res = run(c, command=mixed_wheel_torque, external_torque=ext)
    H = np.asarray(res["H_inertial"])
    expected = H[0] + np.outer(res["t"], tau_I)
    assert np.abs(H - expected).max() < 5e-4 * np.linalg.norm(tau_I) * 60.0


def test_from_rest_body_momentum_cancels_wheel_momentum(cfg):
    """Total momentum starts at zero and stays zero: J w = -A h at all times."""
    c = with_inertia(make(cfg, t_end=60.0), J_SKEWED)
    res = run(c, command=mixed_wheel_torque)
    Jw = np.asarray(res["w"]) @ np.asarray(c.body.J).T
    Ah = np.asarray(res["h"]) @ np.asarray(c.wheels.A).T
    assert np.abs(Jw + Ah).max() < 1e-15


def test_null_space_command_spins_wheels_without_moving_the_body(cfg):
    n = jnp.array([1.0, -1.0, 1.0, -1.0])
    res = run(make(cfg, t_end=20.0), command=lambda t, x: 1e-4 * n)
    assert np.abs(np.asarray(res["w"])).max() < 1e-15
    assert np.allclose(res["q"], res["q"][0], atol=1e-15)
    assert np.allclose(res["h"], -np.outer(res["t"], 1e-4 * n), rtol=1e-12, atol=1e-17)
