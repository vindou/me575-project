"""Quaternion functions against scipy and against their defining properties."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from cubesat_sim import quaternion as quat
from helpers import quaternions, vectors


@given(q=quaternions())
def test_rotation_matrix_is_a_proper_rotation(q):
    R = np.asarray(quat.rotation_matrix(q))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


@given(q=quaternions())
def test_rotation_matrix_agrees_with_scipy(q):
    assert np.allclose(quat.rotation_matrix(q), Rotation.from_quat(q).as_matrix(), atol=1e-12)


@given(q=quaternions())
def test_q_and_minus_q_give_the_same_rotation(q):
    assert np.allclose(quat.rotation_matrix(q), quat.rotation_matrix(-q), atol=1e-14)


@given(v=vectors(3, 1.0), u=vectors(3, 1.0))
def test_skew_is_the_cross_product(v, u):
    assert np.allclose(quat.skew(v) @ u, np.cross(v, u), atol=1e-14)


@given(qa=quaternions(), qb=quaternions())
def test_angle_between_agrees_with_scipy(qa, qb):
    expected = (Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)).magnitude()
    assert float(quat.angle_between(qa, qb)) == pytest.approx(expected, abs=1e-9)


@given(qa=quaternions(), qb=quaternions())
def test_angle_between_is_symmetric_and_sign_invariant(qa, qb):
    a = float(quat.angle_between(qa, qb))
    assert float(quat.angle_between(qb, qa)) == pytest.approx(a, abs=1e-12)
    assert float(quat.angle_between(-qa, qb)) == pytest.approx(a, abs=1e-12)
    assert 0.0 <= a <= np.pi + 1e-12


@pytest.mark.parametrize("angle", [1e-12, 1e-9, 1e-6, 0.5, np.pi - 1e-6])
def test_angle_between_is_accurate_at_small_and_large_angles(angle):
    axis = np.array([0.3, -0.4, 0.866])
    axis /= np.linalg.norm(axis)
    qa = quat.normalize(jnp.array([0.1, 0.2, -0.3, 0.9]))
    qb = (Rotation.from_quat(np.asarray(qa)) * Rotation.from_rotvec(angle * axis)).as_quat()
    assert float(quat.angle_between(qa, qb)) == pytest.approx(angle, rel=1e-6)


def test_angle_between_gradient_is_finite_at_zero_angle():
    q = quat.normalize(jnp.array([0.1, 0.2, -0.3, 0.9]))
    g = jax.grad(quat.angle_between, argnums=1)(q, q)
    assert np.all(np.isfinite(g))


@given(q=quaternions(), w=vectors(3, 1.0))
def test_kinematics_keeps_unit_length(q, w):
    # d/dt |q|^2 = 2 q . q_dot must be zero.
    assert float(q @ quat.kinematics(q, w)) == pytest.approx(0.0, abs=1e-14)


@given(q0=quaternions(), w=vectors(3, 0.3))
def test_kinematics_integrates_to_body_frame_rotation(q0, w):
    """A constant body rate w for time T gives R(T) = R(0) exp([w]x T) (rotation in body frame)."""
    T = 10.0
    f = jax.jit(quat.kinematics)
    sol = solve_ivp(lambda t, q: np.asarray(f(q, w)), (0.0, T), q0,
                    method="DOP853", rtol=1e-12, atol=1e-12)
    expected = (Rotation.from_quat(q0) * Rotation.from_rotvec(w * T)).as_matrix()
    assert np.allclose(quat.rotation_matrix(sol.y[:, -1]), expected, atol=1e-8)
