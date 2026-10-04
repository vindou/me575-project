import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="CubeSat attitude explorer")


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

    import jax
    import jax.numpy as jnp
    import numpy as np

    from cubesat_sim import load_config, simulate, viz
    from cubesat_sim.sim import open_loop_command
    return (ROOT, functools, jax, jnp, load_config, mo, np, open_loop_command, simulate,
            time, viz)


@app.cell
def _(mo):
    mo.md(r"""
    # CubeSat attitude explorer

    Pick a scenario below. Each run is compiled once per command law and duration;
    moving a slider after that reruns the compiled code (milliseconds).

    **Add a controller:** add an entry to `COMMANDS` in the next cell. It must return a
    function `command(t, x)` written with `jax.numpy`. It then appears in the
    *Commands* selector, and the plots and the animation compare it with the others.
    """)
    return


@app.cell
def _(open_loop_command):
    # name -> make(cfg) returning command(t, x) -> wheel torque commands (N,)
    COMMANDS = {
        "open loop (file profile)": lambda cfg: open_loop_command(cfg),
    }
    return (COMMANDS,)


@app.cell
def _(ROOT, mo):
    config_choice = mo.ui.dropdown(sorted(p.name for p in (ROOT / "config").glob("*.yaml")),
                                   value="cubesat_6u.yaml", label="Config")
    return (config_choice,)


@app.cell
def _(ROOT, config_choice, load_config):
    cfg = load_config(ROOT / "config" / config_choice.value)
    return (cfg,)


@app.cell
def _(COMMANDS, cfg, mo, np):
    _w0 = np.degrees(np.asarray(cfg.sim.w0))
    commands_choice = mo.ui.multiselect(list(COMMANDS), value=list(COMMANDS), label="Commands")
    scale = mo.ui.slider(0.0, 40.0, step=0.5, value=1.0, show_value=True,
                         label="Command scale ×")
    failed = mo.ui.dropdown(["none"] + [str(i + 1) for i in range(cfg.wheels.count)],
                            value="none", label="Failed wheel")
    wx, wy, wz = (mo.ui.slider(-2.0, 2.0, step=0.05, value=float(_w0[i]), show_value=True,
                               label=f"ω0 {'xyz'[i]} [deg/s]") for i in range(3))
    duration = mo.ui.number(start=cfg.sim.dt, stop=3600.0, step=1.0, value=cfg.sim.t_end,
                            label="Duration [s]")
    overlay = mo.ui.switch(value=True, label="Overlay the nominal case")
    return commands_choice, duration, failed, overlay, scale, wx, wy, wz


@app.cell
def _(commands_choice, config_choice, duration, failed, mo, overlay, scale, wx, wy, wz):
    mo.hstack([
        mo.vstack([config_choice, commands_choice, duration]),
        mo.vstack([scale, failed, overlay]),
        mo.vstack([wx, wy, wz]),
    ], justify="start", gap=3)
    return


@app.cell
def _(COMMANDS, cfg, functools, jax, jnp, simulate):
    @functools.lru_cache(maxsize=32)
    def compiled_case(name, steps):
        """One compiled simulation per command law and duration.

        The scenario values are traced inputs, so changing them does not recompile.
        A failed wheel gives no torque whatever it is commanded (the command does not know).
        """
        base = COMMANDS[name](cfg)

        def case(scale, healthy, w0):
            x0 = jnp.concatenate([cfg.sim.q0, w0, cfg.sim.h0])
            return simulate(cfg.body, cfg.wheels, x0, cfg.sim.dt, steps,
                            lambda t, x: healthy * (scale * base(t, x)))

        return jax.jit(case)
    return (compiled_case,)


@app.cell
def _(cfg, commands_choice, compiled_case, duration, failed, jnp, mo, np, overlay, scale,
      time, wx, wy, wz):
    mo.stop(not commands_choice.value, mo.callout("Select at least one command.", kind="warn"))

    _steps = int(round(duration.value / cfg.sim.dt))
    _all = jnp.ones(cfg.wheels.count)
    _healthy = _all if failed.value == "none" else _all.at[int(failed.value) - 1].set(0.0)
    _w0 = jnp.radians(jnp.array([wx.value, wy.value, wz.value]))
    _changed = (scale.value != 1.0 or failed.value != "none"
                or not np.allclose(_w0, cfg.sim.w0))

    _t0 = time.perf_counter()
    runs = {}
    for _name in commands_choice.value:
        _f = compiled_case(_name, _steps)
        runs[_name] = {k: np.asarray(v) for k, v in _f(scale.value, _healthy, _w0).items()}
        if overlay.value and _changed:
            runs[f"{_name} — nominal"] = {k: np.asarray(v)
                                          for k, v in _f(1.0, _all, cfg.sim.w0).items()}
    run_seconds = time.perf_counter() - _t0
    return run_seconds, runs


@app.cell
def _(cfg, mo, run_seconds, runs, viz):
    mo.vstack([
        mo.ui.table([{"run": n, **viz.summary(r, cfg)} for n, r in runs.items()],
                    selection=None, pagination=False),
        mo.md(f"<small>Simulated {len(runs)} run(s) in {1e3 * run_seconds:.0f} ms "
              f"(includes compilation on the first run of a command or duration).</small>"),
    ])
    return


@app.cell
def _(cfg, runs, viz):
    time_series_fig = viz.time_series(runs, cfg)
    time_series_fig
    return (time_series_fig,)


@app.cell
def _(mo, runs):
    show_animation = mo.ui.switch(value=True, label="3D animation")
    animated_run = mo.ui.dropdown(list(runs), value=list(runs)[0], label="Run")
    # marimo shows outputs up to 8 MB; 800 frames is about 6 MB.
    max_frames = mo.ui.slider(100, 800, step=50, value=400, show_value=True,
                              label="Max frames")
    mo.hstack([show_animation, animated_run, max_frames], justify="start", gap=3)
    return animated_run, max_frames, show_animation


@app.cell
def _(animated_run, cfg, max_frames, mo, runs, show_animation, viz):
    mo.stop(not show_animation.value)
    animation_fig = viz.attitude_animation(runs[animated_run.value], cfg,
                                           max_frames=max_frames.value)
    animation_fig
    return (animation_fig,)


if __name__ == "__main__":
    app.run()
