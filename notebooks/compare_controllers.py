import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="CubeSat controller comparison")


@app.cell
def _():
    import functools
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    _nb = mo.notebook_dir()
    ROOT = _nb.parent if _nb is not None else Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import numpy as np
    import yaml

    from cubesat_sim import load_config, run, viz
    from run_sim import build_command
    return ROOT, build_command, functools, load_config, mo, np, run, time, viz, yaml


@app.cell
def _(mo):
    mo.md(r"""
    # Controller comparison: PD, MPC and MPPI

    Runs the three attitude controllers on the same plant and target and compares them.
    Every controller uses exactly the settings `run_sim.py` would use: the plant file
    and the control file of the selected scenario.

    - **PD**: fixed-gain quaternion PD (`cubesat_sim/controller.py`).
    - **MPC**: nonlinear shooting MPC, optax L-BFGS over an RK4 rollout (`cubesat_sim/mpc.py`).
    - **MPPI**: sampling-based MPPI with annealed noise (`cubesat_sim/mppi.py`).

    Results are cached per scenario and controller, so switching back is instant. The
    first run of a scenario takes a while: about 20 s for the default scenario and 2–3 min
    for a coupled one (MPC is the slowest, 1–2 min for 600 s of simulated time).
    """)
    return


@app.cell
def _():
    # name -> (plant file, control file), relative to the project root
    SCENARIOS = {
        "Default: 50° about z, 60 s": (
            "config/cubesat_6u.yaml", "config/control/controllers.yaml"),
        "Coupled, fast PD gains (wn 0.2), 600 s": (
            "config/cubesat_6u_coupled.yaml", "config/control/coupled_fast.yaml"),
        "Coupled, slow PD gains (wn 0.02), 600 s": (
            "config/cubesat_6u_coupled.yaml", "config/control/coupled_slow.yaml"),
    }
    CONTROLLER_NAMES = ("pd", "mpc", "mppi")
    # Fixed color per controller (validated categorical palette: CVD-separable), so a
    # controller keeps its color when others are deselected.
    COLORS = {"pd": "#2a78d6", "mpc": "#eb6834", "mppi": "#1baf7a"}
    return COLORS, CONTROLLER_NAMES, SCENARIOS


@app.cell
def _(CONTROLLER_NAMES, SCENARIOS, mo):
    scenario = mo.ui.dropdown(list(SCENARIOS), value=list(SCENARIOS)[0], label="Scenario")
    controllers = mo.ui.multiselect(list(CONTROLLER_NAMES), value=list(CONTROLLER_NAMES),
                                    label="Controllers")
    mo.hstack([scenario, controllers], justify="start", gap=3)
    return controllers, scenario


@app.cell
def _(ROOT, build_command, functools, load_config, np, run, time):
    @functools.lru_cache(maxsize=16)
    def simulate_case(plant_file, control_file, name):
        """One closed-loop run. Returns (result arrays, q_target, settings, wall seconds)."""
        cfg = load_config(ROOT / plant_file)
        command, q_target, settings = build_command(cfg, name, ROOT / control_file)
        t0 = time.perf_counter()
        res = {k: np.asarray(v) for k, v in run(cfg, command=command).items()}
        return res, np.asarray(q_target), settings, time.perf_counter() - t0
    return (simulate_case,)


@app.cell
def _(ROOT, SCENARIOS, controllers, load_config, mo, scenario, simulate_case):
    mo.stop(not controllers.value, mo.callout("Select at least one controller.", kind="warn"))
    plant_file, control_file = SCENARIOS[scenario.value]
    cfg = load_config(ROOT / plant_file)
    runs, settings, seconds = {}, {}, {}
    q_target = None
    for _name in mo.status.progress_bar(list(controllers.value),
                                        title="Simulating", show_eta=False):
        runs[_name], q_target, settings[_name], seconds[_name] = simulate_case(
            plant_file, control_file, _name)
    return cfg, control_file, plant_file, q_target, runs, seconds, settings


@app.cell
def _(cfg, mo, q_target, runs, seconds, viz):
    def _fmt(value, digits=3):
        if value is None:
            return "—"
        return float(f"{value:.{digits}g}")

    perf = {n: viz.performance(r, cfg, q_target) for n, r in runs.items()}
    _rows = [{"controller": n, **{k: _fmt(v) for k, v in p.items()},
              "compute [s]": _fmt(seconds[n], 2)} for n, p in perf.items()]
    _req = cfg.requirements
    _checks = [{"controller": n, **{k: v or "—" for k, v in viz.requirement_checks(p, cfg).items()}}
               for n, p in perf.items()]
    mo.vstack([
        mo.md("## Performance"),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(
            "<small>Settling: pointing error stays below "
            f"{_req.get('settle_band_arcsec', '—')} arcsec and body rate below "
            f"{_req.get('settle_rate_deg_s', '—')} deg/s until the end of the run. "
            "Hold statistics use the samples after settling; 3σ is the 99.73th percentile. "
            "Compute time is wall time for the simulation (including compilation the "
            "first time).</small>"),
        mo.md("### Project requirements"),
        mo.ui.table(_checks, selection=None, pagination=False),
        mo.md(
            f"<small>Hold RMS ≤ {_req.get('pointing_error_arcsec_rms', '—')} arcsec, "
            f"hold 3σ ≤ {_req.get('pointing_error_arcsec_3sigma', '—')} arcsec, "
            f"peak rate ≤ {_req.get('max_slew_rate_deg_s', '—')} deg/s, peak wheel "
            f"momentum ≤ {100 * _req.get('wheel_momentum_warning_frac', float('nan')):.0f} % "
            "of h_max. These are project assumptions, not mission standards.</small>"),
    ])
    return (perf,)


@app.cell
def _(COLORS, cfg, mo, q_target, runs, viz):
    time_series_fig = viz.time_series(runs, cfg, q_target=q_target, colors=COLORS)
    mo.vstack([
        mo.md("## Time histories\n\nPointing error (log scale) with the settling band and "
              "the RMS requirement, body rate with the slew-rate limit, the largest wheel "
              "momentum and the largest wheel torque. Click a controller in the legend to "
              "hide it in every plot."),
        time_series_fig,
    ])
    return (time_series_fig,)


@app.cell
def _(control_file, mo, plant_file, settings, yaml):
    mo.accordion({
        f"Controller settings ({control_file}, plant {plant_file})": mo.md(
            "\n\n".join(f"**{n}**\n\n```yaml\n"
                        f"{yaml.safe_dump(s.get('resolved_design', s), sort_keys=False)}```"
                        for n, s in settings.items())),
    })
    return


@app.cell
def _(mo, runs):
    animated_run = mo.ui.dropdown(list(runs), value=list(runs)[0], label="Animate")
    # marimo shows outputs up to 8 MB; 800 frames is about 6 MB.
    max_frames = mo.ui.slider(100, 800, step=50, value=300, show_value=True,
                              label="Max frames")
    mo.vstack([mo.md("## 3D animation"),
               mo.hstack([animated_run, max_frames], justify="start", gap=3)])
    return animated_run, max_frames


@app.cell
def _(animated_run, cfg, max_frames, q_target, runs, viz):
    animation_fig = viz.attitude_animation(runs[animated_run.value], cfg, q_target=q_target,
                                           max_frames=max_frames.value)
    animation_fig
    return (animation_fig,)


if __name__ == "__main__":
    app.run()
