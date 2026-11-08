"""1D thruster plant, its config loader, the PID controller, and the runner/registry."""
import dataclasses
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from cubesat_sim.config import load_config, load_plant, load_thruster_config, plant_model
from cubesat_sim.controller import CONTROLLERS, CONTROLLERS_BY_MODEL
from cubesat_sim.thruster1d import (DisturbanceProfile, _simulate_thruster_jit,
                                    run_thruster, simulate_thruster)
from cubesat_sim.thruster_control import pid_controller, pid_gains
from cubesat_sim import viz

ROOT = Path(__file__).resolve().parents[1]
THRUSTER_PATH = ROOT / "config" / "thruster_1d.yaml"


@pytest.fixture(scope="module")
def tcfg():
    return load_thruster_config(THRUSTER_PATH)


def with_(cfg, disturbance=None, **sim_changes):
    """Copy of a thruster config with sim fields and/or the disturbance replaced."""
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, **sim_changes))
    if disturbance is not None:
        cfg = dataclasses.replace(cfg, disturbance=disturbance)
    return cfg


def numpy(res):
    return {k: np.asarray(v) for k, v in res.items()}


def full_thrust(t, x):
    return jnp.asarray(1e3)          # far above F_max: the plant saturates it to D = 1


# --- plant -------------------------------------------------------------------------------

def test_constant_full_thrust_matches_closed_form(tcfg):
    """x(t) = x0 + v0 t + (F_max + F_d) t^2 / (2 m); RK4 is exact for a quadratic."""
    x0, v0, F_d = 0.7, -0.03, -0.012
    c = with_(tcfg, DisturbanceProfile(F_d), t_end=120.0,
              position0=jnp.asarray(x0), velocity0=jnp.asarray(v0))
    r = numpy(run_thruster(c, command=full_thrust))
    m, F_max = float(c.params.mass), float(c.params.F_max)
    a = (F_max + F_d) / m
    np.testing.assert_allclose(r["x"], x0 + v0 * r["t"] + 0.5 * a * r["t"] ** 2,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r["v"], v0 + a * r["t"], rtol=1e-12, atol=1e-14)
    assert np.all(r["D"] == 1.0)


def test_duty_cycle_and_force_stay_in_limits(tcfg):
    """Whole runs with a command far outside [0, F_max] in both directions, and with PID."""
    F_max = float(tcfg.params.F_max)

    def wild(t, x):
        return 10.0 * F_max * jnp.sin(0.3 * t)

    for command in (wild, pid_controller(tcfg, 3.0)):
        r = numpy(run_thruster(tcfg, command=command))
        assert r["D"].min() >= 0.0 and r["D"].max() <= 1.0
        assert r["F"].min() >= 0.0 and r["F"].max() <= F_max
        np.testing.assert_allclose(r["F"], np.clip(r["F_cmd"], 0.0, F_max), rtol=0, atol=1e-15)
    r = numpy(run_thruster(tcfg, command=wild))
    assert np.any(r["D"] == 0.0) and np.any(r["D"] == 1.0)     # both limits were active


def test_disturbance_applied_in_configured_windows(tcfg):
    profile = DisturbanceProfile(-0.01, [(10.0, 20.0, -0.05), (15.0, 30.0, 0.02)])
    c = with_(tcfg, profile, t_end=40.0)
    r = numpy(run_thruster(c, command=lambda t, x: jnp.asarray(0.0)))
    t = r["t"]
    expected = (-0.01 - 0.05 * ((10.0 <= t) & (t < 20.0))
                + 0.02 * ((15.0 <= t) & (t < 30.0)))
    np.testing.assert_allclose(r["F_d"], expected, rtol=0, atol=1e-15)
    # Exactly the steps starting in [10, 20) carry the first segment.
    k = np.flatnonzero(np.isclose(r["F_d"], -0.06) | np.isclose(r["F_d"], -0.04))
    assert t[k[0]] == pytest.approx(10.0) and t[k[-1]] == pytest.approx(20.0 - c.sim.dt)
    # With zero thrust, the velocity change is the impulse of the profile.
    dv = np.sum(r["F_d"][:-1]) * c.sim.dt / float(c.params.mass)
    assert r["v"][-1] - r["v"][0] == pytest.approx(dv, rel=1e-12)


def test_same_disturbance_does_not_recompile(tcfg):
    run_thruster(tcfg, command=full_thrust)
    before = _simulate_thruster_jit._cache_size()
    again = dataclasses.replace(tcfg, disturbance=DisturbanceProfile(
        tcfg.disturbance.F_const, tcfg.disturbance.segments))     # new object, equal by value
    run_thruster(again, command=full_thrust)
    assert _simulate_thruster_jit._cache_size() == before


def test_grad_and_vmap(tcfg):
    """Differentiable in mass, and batchable over initial states (float64 throughout)."""
    c = with_(tcfg, DisturbanceProfile(0.0), t_end=10.0)

    def final_x(mass):
        params = dataclasses.replace(c.params, mass=mass)
        return simulate_thruster(params, c.sim.x0, c.sim.dt, c.sim.steps,
                                 full_thrust, DisturbanceProfile(0.0))["x"][-1]

    m = float(c.params.mass)
    g = jax.grad(final_x)(jnp.asarray(m))
    exact = -0.5 * float(c.params.F_max) * 10.0 ** 2 / m ** 2      # d/dm of F t^2 / (2 m)
    assert float(g) == pytest.approx(exact, rel=1e-10)

    x0s = jnp.array([[0.0, 0.0], [1.0, 0.1], [-2.0, -0.2]])
    batch = jax.vmap(lambda x0: simulate_thruster(c.params, x0, c.sim.dt, c.sim.steps,
                                                  full_thrust, c.disturbance)["x"][-1])(x0s)
    assert batch.dtype == jnp.float64 and batch.shape == (3,)


# --- PID -----------------------------------------------------------------------------------

def integral_from_run(r, cmd, x_ref):
    """Recover the PID integral from u = Kp e + Ki I + Kd e_dot."""
    g = cmd.design_info
    e, e_dot = x_ref - r["x"], -r["v"]
    return (r["F_cmd"] - g["Kp"] * e - g["Kd"] * e_dot) / g["Ki"]


def test_anti_windup_bounds_integral_in_long_saturation(tcfg):
    """From t = 20 s to 170 s a push of -(F_max + F_d) exactly cancels full thrust: the mass
    stops short of x_ref with D = 1 and e > 0. Without anti-windup the integral keeps
    growing; after the push ends the thruster then keeps firing long past x_ref, and with
    no braking the mass overshoots far and does not come back within the run."""
    F_max, F_d, x_ref = float(tcfg.params.F_max), -0.01, 1.0
    c = with_(tcfg, DisturbanceProfile(F_d, [(20.0, 170.0, -(F_max + F_d))]), t_end=600.0)
    runs = {}
    for aw in (True, False):
        cmd = pid_controller(c, x_ref, pole_rad_s=0.05, anti_windup=aw)
        r = numpy(run_thruster(c, command=cmd))
        runs[aw] = (r, integral_from_run(r, cmd, x_ref))
    (r_aw, I_aw), (r_no, I_no) = runs[True], runs[False]
    t = r_aw["t"]
    push = (t >= 40.0) & (t < 170.0)
    assert np.all(r_aw["D"][push] == 1.0) and np.all(r_no["D"][push] == 1.0)
    assert np.all(x_ref - r_aw["x"][push] > 0.1)                # e > 0 throughout
    # With anti-windup the integral is frozen during the saturated push.
    assert np.ptp(I_aw[push]) < 1e-9
    # Without it, it grows linearly through the push (e * 130 s, e ~ 0.43 m).
    assert np.ptp(I_no[push]) > 50.0
    end = np.searchsorted(t, 170.0)
    assert I_no[end] > 10.0 * I_aw[end]
    overshoot = lambda r: max(0.0, (r["x"] - x_ref).max())
    assert overshoot(r_no) > 4.0 * overshoot(r_aw)
    assert abs(r_aw["x"][-1] - x_ref) < 0.05                    # recovered by the end
    assert r_no["x"][-1] - x_ref > 10.0                         # still far past x_ref


def test_reachable_setpoint_settles(tcfg):
    """Approached from below with low gain (no overshoot, so no braking is needed), the
    PID settles at x_ref; the integral supplies the thrust that balances F_d."""
    x_ref = 1.0
    c = with_(tcfg, DisturbanceProfile(-0.01), t_end=900.0)
    r = numpy(run_thruster(c, command=pid_controller(c, x_ref, pole_rad_s=0.02)))
    assert r["x"].max() <= x_ref + 1e-3
    assert abs(r["x"][-1] - x_ref) < 1e-4
    assert abs(r["v"][-1]) < 1e-5
    assert r["F"][-1] == pytest.approx(0.01, rel=1e-3)          # holds against F_d


def test_pid_gains():
    m, p = 11.0, 0.05
    assert pid_gains(m, pole_rad_s=p) == pytest.approx((3 * m * p**2, m * p**3, 3 * m * p))
    assert pid_gains(m, 1.0, 0.1, 2.0) == (1.0, 0.1, 2.0)


@pytest.mark.parametrize("kwargs, message", [
    (dict(Kp=1.0, Kd=1.0), "all of Kp, Ki and Kd"),
    (dict(Kp=1.0, Ki=0.1, Kd=1.0, pole_rad_s=0.1), "not both"),
    (dict(pole_rad_s=-0.1), "positive"),
    (dict(Kp=-1.0, Ki=0.1, Kd=1.0), "nonnegative"),
    (dict(anti_windup="yes"), "true or false"),
])
def test_pid_rejects_bad_parameters(tcfg, kwargs, message):
    with pytest.raises(ValueError, match=message):
        pid_controller(tcfg, 1.0, **kwargs)


def test_pid_design_info(tcfg):
    info = pid_controller(tcfg, 2.0, pole_rad_s=0.1).design_info
    assert info["mode"] == "automatic" and info["anti_windup"] is True
    assert info["Kd"] == pytest.approx(3 * float(tcfg.params.mass) * 0.1)
    assert pid_controller(tcfg, 2.0, Kp=1, Ki=0, Kd=2).design_info["mode"] == "manual"


# --- config ------------------------------------------------------------------------------

@pytest.fixture
def raw():
    return yaml.safe_load(THRUSTER_PATH.read_text())


def load_modified(tmp_path, raw, change):
    change(raw)
    path = tmp_path / "thruster.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_thruster_config(path)


def test_loads_default_file(tcfg):
    assert tcfg.model == "thruster_1d" and tcfg.sim.steps == 6000
    assert isinstance(tcfg.disturbance, DisturbanceProfile)
    assert type(load_plant(THRUSTER_PATH)).__name__ == "ThrusterConfig"
    assert type(load_plant(ROOT / "config" / "cubesat_6u.yaml")).__name__ == "Config"


def set_(key, value):
    return lambda raw: raw.__setitem__(key, value)


def set_segment(key, value):
    return lambda raw: raw["disturbance"]["segments"][0].__setitem__(key, value)


@pytest.mark.parametrize("change, message", [
    (set_("mass_kg", 0.0), "mass_kg must be positive"),
    (set_("mass_kg", float("nan")), "mass_kg must be finite"),
    (set_("F_max_N", -0.1), "F_max_N must be positive"),
    (set_("dt_s", 0.0), "dt_s must be positive"),
    (set_("t_end_s", 300.01), "whole multiple"),
    (set_("x0_m", "far"), "x0_m must be a number"),
    (lambda raw: raw.pop("mass_kg"), "mass_kg is missing"),
    (set_("model", "rocket"), "model must be one of"),
    (set_("model", "attitude"), "not a thruster_1d config"),
    (set_segment("t_end_s", 200.0), "t_start_s < t_end_s"),
    (set_segment("t_end_s", 150.0), "t_start_s < t_end_s"),
    (lambda raw: raw["disturbance"]["segments"][0].pop("F_N"), "needs t_start_s"),
    (lambda raw: raw["disturbance"].__setitem__("F_d_N", float("inf")), "finite"),
    (set_("disturbance", [1, 2]), "must be a mapping"),
])
def test_rejects_bad_inputs(tmp_path, raw, change, message):
    with pytest.raises(ValueError, match=message):
        load_modified(tmp_path, raw, change)


def test_optional_disturbance(tmp_path, raw):
    c = load_modified(tmp_path, raw, lambda raw: raw.pop("disturbance"))
    assert c.disturbance == DisturbanceProfile(0.0)


def test_attitude_loader_rejects_thruster_file():
    with pytest.raises(ValueError, match="thruster_1d config"):
        load_config(THRUSTER_PATH)


def test_model_key_defaults_to_attitude():
    assert plant_model({}) == "attitude"
    assert plant_model({"model": "thruster_1d"}) == "thruster_1d"


# --- registry, runner, plot ---------------------------------------------------------------

def test_registries_are_separate():
    assert CONTROLLERS_BY_MODEL["attitude"] is CONTROLLERS
    assert set(CONTROLLERS_BY_MODEL["attitude"]) == {"pd", "mpc", "mppi"}
    assert set(CONTROLLERS_BY_MODEL["thruster_1d"]) == {"pid"}


def run_sim(*args):
    return subprocess.run([sys.executable, str(ROOT / "run_sim.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT)


def test_runner_thruster_pid(tmp_path):
    out = run_sim(THRUSTER_PATH, tmp_path / "pid", "--controller", "pid")
    assert out.returncode == 0, out.stderr
    for label in ("Final position error", "Overshoot", "Peak speed", "Time at D = 1",
                  "Time at D = 0", "Thrust impulse"):
        assert label in out.stdout
    res = np.load(tmp_path / "pid" / "run.npz")
    assert {"t", "x", "v", "F_cmd", "F", "D", "F_d"} <= set(res.files)
    used = yaml.safe_load((tmp_path / "pid" / "controller_used.yaml").read_text())
    assert used["resolved_design"]["mode"] == "automatic"


def test_runner_rejects_controller_of_other_model(tmp_path):
    out = run_sim(THRUSTER_PATH, tmp_path / "x", "--controller", "pd")
    assert out.returncode != 0 and "does not apply to model 'thruster_1d'" in out.stderr
    out = run_sim(ROOT / "config" / "cubesat_6u.yaml", tmp_path / "y", "--controller", "pid")
    assert out.returncode != 0 and "does not apply to model 'attitude'" in out.stderr


def test_thruster_plot(tcfg):
    r = run_thruster(tcfg, command=pid_controller(tcfg, 1.0))
    fig = viz.thruster_time_series({"a": r, "b": r}, x_ref=1.0)
    assert len(fig.data) == 6                                    # x, v, D per run
    np.testing.assert_allclose(fig.data[2].y, np.asarray(r["D"])[::1])
