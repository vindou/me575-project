"""Figures: they build, and what they draw matches the simulation."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cubesat_sim import quaternion as quat
from cubesat_sim import run, viz
from cubesat_sim.sim import open_loop_command
from helpers import make

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def res(cfg):
    return {k: np.asarray(v) for k, v in run(cfg).items()}


def frame_index(frame, res):
    """Sample index of an animation frame, read from its title (t = ... s)."""
    t_f = float(frame.layout.title.text.split("s")[0].split("=")[1])
    return int(np.argmin(np.abs(res["t"] - t_f)))


def traces(fig, row):
    axis = "y" if row == 1 else f"y{row}"
    return [t for t in fig.data if t.yaxis == axis]


def test_rotation_and_angle_helpers_match_the_quaternion_module(res):
    q = res["q"][::500]
    R = viz.rotation_matrices(q)
    assert np.allclose(R, Rotation.from_quat(q).as_matrix(), atol=1e-12)
    angles = viz.attitude_angle(q, q[0])
    assert np.allclose(angles, [float(quat.angle_between(q[0], qk)) for qk in q], atol=1e-14)


def test_time_series_single_run_shows_components(cfg, res):
    fig = viz.time_series(res, cfg)
    assert [len(traces(fig, row)) for row in (1, 2, 3, 4)] == [1, 4, 4, 4]
    wheel1 = traces(fig, 3)[0]
    assert np.allclose(wheel1.y, 100 * res["h"][:, 0] / float(cfg.wheels.h_max))
    torque = traces(fig, 4)[0]
    assert np.allclose(torque.y, 1e3 * res["u"][:, 0])
    assert np.allclose(traces(fig, 1)[0].y[-1], 11.101, atol=1e-3)     # degrees


def test_time_series_comparison_shows_one_line_per_run(cfg, res):
    base = open_loop_command(cfg)
    failed = {k: np.asarray(v) for k, v in
              run(cfg, command=lambda t, x: base(t, x) * jnp.array([0.0, 1, 1, 1])).items()}
    fig = viz.time_series({"nominal": res, "failed": failed}, cfg)
    assert [len(traces(fig, row)) for row in (1, 2, 3, 4)] == [2, 2, 2, 2]
    assert sum(t.showlegend is not False for t in fig.data) == 2       # one legend entry per run
    h_line = traces(fig, 3)[1]
    assert np.allclose(h_line.y, 100 * np.abs(failed["h"]).max(axis=1) / float(cfg.wheels.h_max))


def test_time_series_pointing_mode_uses_arcsec_on_a_log_scale(cfg, res):
    fig = viz.time_series(res, cfg, q_target=res["q"][-1])
    assert fig.layout.yaxis.type == "log"
    error = traces(fig, 1)[0].y
    assert error[-1] == pytest.approx(0.0, abs=1e-6)
    assert error[0] == pytest.approx(11.101 * 3600, rel=1e-3)


def test_time_series_draws_requirement_limits(cfg, res):
    fig = viz.time_series(res, cfg)
    lines = {round(s.y0, 6) for s in fig.layout.shapes}
    assert {cfg.requirements["max_slew_rate_deg_s"], 100.0, 80.0,
            round(1e3 * float(cfg.wheels.u_max), 6)} <= lines


@pytest.fixture(scope="module")
def animation(cfg, res):
    return viz.attitude_animation(res, cfg, max_frames=200)


def test_animation_frames_cover_the_run(cfg, res, animation):
    assert 2 <= len(animation.frames) <= 201
    assert animation.frames[-1].layout.title.text.lstrip().startswith(f"t = {res['t'][-1]:7.2f} s".lstrip())
    steps = animation.layout.sliders[0].steps
    assert len(steps) == len(animation.frames)


def test_animation_body_axes_follow_the_attitude(cfg, res, animation):
    L = cfg.visualization["axis_length_m"]
    n = len(animation.frames)
    for f in (0, n // 2, n - 1):
        frame = animation.frames[f]
        R = Rotation.from_quat(res["q"][frame_index(frame, res)]).as_matrix()
        for i in range(3):
            axis_trace = frame.data[2 + i]
            tip = np.array([axis_trace.x[1], axis_trace.y[1], axis_trace.z[1]])
            assert np.allclose(tip, L * R[:, i], atol=1e-6)       # float32 storage


def test_animation_box_keeps_its_shape(animation):
    first = animation.frames[0].data[0]
    last = animation.frames[-1].data[0]
    def edges(m):
        V = np.column_stack([m.x, m.y, m.z]).astype(float)
        return np.linalg.norm(V[:, None] - V[None], axis=2)
    assert np.allclose(edges(first), edges(last), atol=1e-6)


def test_animation_wheel_colors_and_bars_show_momentum(cfg, res, animation):
    frame = animation.frames[len(animation.frames) // 3]
    h_frac = res["h"][frame_index(frame, res)] / float(cfg.wheels.h_max)
    bars = frame.data[7]
    assert np.allclose(bars.y, 100 * h_frac, atol=1e-4)
    intensity = np.asarray(frame.data[5].intensity)
    assert set(np.round(intensity, 6)) <= set(np.round(h_frac.astype(np.float32), 6))


def test_animation_target_axes_are_drawn(cfg, res):
    fig = viz.attitude_animation(res, cfg, q_target=res["q"][-1], max_frames=20)
    assert sum(t.name.startswith("target") for t in fig.data) == 3


def run_notebook(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "notebooks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app.run()[1]


@pytest.mark.slow
def test_explore_notebook_runs_without_errors():
    defs = run_notebook("explore")
    assert "runs" in defs and len(defs["runs"]) >= 1
    assert defs["time_series_fig"] is not None


@pytest.mark.slow
def test_view_run_notebook_shows_a_saved_run(tmp_path, monkeypatch):
    subprocess.run([sys.executable, str(ROOT / "run_sim.py"), str(ROOT / "config" / "cubesat_6u.yaml"),
                    str(tmp_path / "results")], check=True, capture_output=True, cwd=ROOT)
    monkeypatch.chdir(tmp_path)                     # the notebook defaults to ./results
    defs = run_notebook("view_run")
    assert list(defs["runs"]) == ["results"]       # the folder itself holds run.npz
    assert defs["runs"]["results"]["t"][-1] == pytest.approx(60.0)
    assert defs["animation_fig"] is not None
