"""Fixed-step RK4 behaviour: convergence order, time grid, long-horizon drift."""
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cubesat_sim import run
from helpers import make, with_inertia

_R = Rotation.from_rotvec([0.4, -0.3, 0.2]).as_matrix()
J_SKEWED = _R @ np.diag([0.12, 0.09, 0.05]) @ _R.T


def no_wheel_torque(t, x):
    return jnp.zeros(4)


def tumbling(cfg, dt, t_end):
    return with_inertia(make(cfg, dt=dt, t_end=t_end, w0=jnp.array([0.3, -0.2, 0.25]),
                             h0=jnp.array([0.004, -0.002, 0.001, 0.003])), J_SKEWED)


def final_state(cfg, dt, t_end=20.0):
    res = run(tumbling(cfg, dt, t_end), command=no_wheel_torque)
    return np.concatenate([res["q"][-1], res["w"][-1], res["h"][-1]])


def test_rk4_is_fourth_order(cfg):
    ref = final_state(cfg, 0.2 / 32)
    errors = [np.linalg.norm(final_state(cfg, dt) - ref) for dt in (0.2, 0.1, 0.05)]
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all((orders > 3.8) & (orders < 4.2)), (errors, orders)


def test_time_grid(cfg):
    res = run(make(cfg, dt=0.05, t_end=3.0), command=no_wheel_torque)
    t = np.asarray(res["t"])
    assert len(t) == 61
    assert np.array_equal(t, np.arange(61) * 0.05)
    assert t[-1] == pytest.approx(3.0, abs=1e-12)
    assert all(v.shape[0] == 61 for v in res.values())


def test_quaternion_stays_unit_length(cfg):
    res = run(tumbling(cfg, 0.1, 200.0), command=no_wheel_torque)
    assert np.abs(np.linalg.norm(res["q"], axis=1) - 1.0).max() < 1e-14


@pytest.mark.slow
def test_ten_orbits_without_drift(cfg):
    """About 10 orbits at 500 km (57,000 s) with a slow tumble and a small wheel torque."""
    c = with_inertia(make(cfg, dt=0.05, t_end=57000.0, w0=jnp.array([0.002, -0.001, 0.0015]),
                          h0=jnp.array([0.004, -0.002, 0.001, 0.003])), J_SKEWED)
    res = run(c, command=lambda t, x: 1e-7 * jnp.array([jnp.sin(1e-3 * t), 0.0, -jnp.sin(1e-3 * t), 0.0]))
    H = np.asarray(res["H_inertial"])
    assert np.linalg.norm(H - H[0], axis=1).max() < 1e-9 * np.linalg.norm(H[0])
    assert np.abs(np.linalg.norm(res["q"], axis=1) - 1.0).max() < 1e-13
