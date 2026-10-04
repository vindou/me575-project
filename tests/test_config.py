"""Loading, checking and derived values of the configuration file."""
import copy

import jax
import numpy as np
import pytest
import yaml
from hypothesis import given, strategies as st

from cubesat_sim import load_config, run
from cubesat_sim.config import wheel_axes
from helpers import CFG_PATH


@pytest.fixture
def raw():
    return yaml.safe_load(CFG_PATH.read_text())


def load_modified(tmp_path, raw, key_path, value):
    raw = copy.deepcopy(raw)
    *parents, last = key_path.split(".")
    node = raw
    for k in parents:
        node = node[int(k)] if isinstance(node, list) else node[k]
    node[last] = value
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(p)


BAD_VALUES = [
    ("body.inertia_kg_m2", [[0.1, 0.01, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]], "symmetric"),
    ("body.inertia_kg_m2", [[0.1, 0.0], [0.0, 0.1]], "symmetric"),
    ("body.inertia_kg_m2", [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, -0.1]], "positive definite"),
    ("body.mass_kg", 0.0, "mass_kg must be positive"),
    ("wheels.skew_angle_from_z_deg", 90.0, "three body axes"),   # all axes in the xy plane
    ("wheels.skew_angle_from_z_deg", 0.0, "three body axes"),    # all axes along z
    ("wheels.u_max_Nm", -0.004, "u_max_Nm must be positive"),
    ("wheels.h_max_Nms", 0.0, "h_max_Nms must be positive"),
    ("wheels.rotor_inertia_kg_m2", 0.0, "rotor_inertia_kg_m2 must be positive"),
    ("sim.q0", [0.0, 0.0, 0.0, 2.0], "q0"),
    ("sim.q0", [0.0, 0.0, 1.0], "q0"),
    ("sim.w0_rad_s", [0.0, 0.0], "w0_rad_s"),
    ("sim.h0_Nms", [0.0, 0.0, 0.0], "h0_Nms must have 4"),
    ("sim.h0_Nms", [0.0, 0.016, 0.0, 0.0], "larger than"),
    ("sim.dt_s", 0.0, "dt_s must be positive"),
    ("sim.dt_s", -0.01, "dt_s must be positive"),
    ("sim.t_end_s", 0.0, "t_end_s must be positive"),
    ("sim.dt_s", 0.07, "whole multiple"),                        # 60 / 0.07 is not whole
    ("open_loop_command.0.u_Nm", [0.0, 0.0, 0.0], "u_Nm must have 4"),
    ("open_loop_command.0.t_end_s", 5.0, "t_start_s < t_end_s"),
]


@pytest.mark.parametrize("key_path, value, message", BAD_VALUES,
                         ids=[f"{k}={v}" for k, v, _ in BAD_VALUES])
def test_bad_values_are_rejected(tmp_path, raw, key_path, value, message):
    with pytest.raises(ValueError, match=message):
        load_modified(tmp_path, raw, key_path, value)


def test_q0_is_normalized(tmp_path, raw):
    c = load_modified(tmp_path, raw, "sim.q0", [0.0, 0.0, 0.0, 1.0005])
    assert np.linalg.norm(c.sim.q0) == pytest.approx(1.0, abs=1e-15)


def test_missing_command_section_gives_zero_command(tmp_path, raw):
    c = load_modified(tmp_path, raw, "open_loop_command", None)
    assert c.open_loop_command == ()
    res = run(c)
    assert np.all(np.asarray(res["u"]) == 0.0)
    assert np.all(np.asarray(res["w"]) == 0.0)


def test_derived_values(cfg):
    assert cfg.sim.steps == 6000
    assert cfg.wheels.count == 4
    assert np.allclose(cfg.body.J_inv @ cfg.body.J, np.eye(3), atol=1e-12)
    # 6000 rpm at h_max (the stated reason for the rotor inertia value).
    rpm = float(cfg.wheels.h_max / cfg.wheels.rotor_inertia) * 60 / (2 * np.pi)
    assert rpm == pytest.approx(6000, rel=0.01)


def test_inertia_is_the_uniform_box(cfg):
    m, (a, b, c) = 11.0, cfg.visualization["body_size_m"]
    expected = m / 12 * np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2])
    assert np.allclose(cfg.body.J, expected, atol=1e-4)


def test_wheel_matrix_is_isotropic(cfg):
    A = np.asarray(cfg.wheels.A)
    assert np.allclose(np.linalg.norm(A, axis=0), 1.0)
    assert np.allclose(A @ A.T, (4.0 / 3.0) * np.eye(3), atol=1e-5)
    assert np.allclose(A @ np.array([1.0, -1.0, 1.0, -1.0]), 0.0, atol=1e-15)   # null space


@pytest.mark.parametrize("failed", range(4))
def test_one_wheel_out_keeps_three_axis_control(cfg, failed):
    A = np.delete(np.asarray(cfg.wheels.A), failed, axis=1)
    sv = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(sv, [2 / np.sqrt(3), 2 / np.sqrt(3), 1 / np.sqrt(3)], atol=1e-5)


@given(skew=st.floats(0.0, 180.0), az=st.lists(st.floats(-360.0, 360.0), min_size=1, max_size=6))
def test_wheel_axes_are_unit_vectors(skew, az):
    A = np.asarray(wheel_axes(skew, az))
    assert A.shape == (3, len(az))
    assert np.allclose(np.linalg.norm(A, axis=0), 1.0)


def test_params_are_pytrees_with_static_time_fields(cfg):
    leaves = jax.tree_util.tree_leaves(cfg.sim)
    assert len(leaves) == 3                                     # q0, w0, h0
    doubled = jax.tree_util.tree_map(lambda v: 2 * v, cfg.sim)
    assert doubled.dt == cfg.sim.dt and doubled.t_end == cfg.sim.t_end
    assert len(jax.tree_util.tree_leaves(cfg.body)) == 2        # mass, J
    assert len(jax.tree_util.tree_leaves(cfg.wheels)) == 4      # A, u_max, h_max, rotor_inertia
