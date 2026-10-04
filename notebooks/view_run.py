import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="CubeSat saved runs")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    _nb = mo.notebook_dir()
    ROOT = _nb.parent if _nb is not None else Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import numpy as np

    from cubesat_sim import load_config, viz
    return Path, ROOT, load_config, mo, np, viz


@app.cell
def _(mo):
    mo.md(r"""
    # Saved runs

    Shows runs written by `python run_sim.py [config] [folder]`. Every folder below the
    path that holds a `run.npz` is listed, so save runs side by side (for example
    `results/pd/` and `results/lqr/`) to compare them.
    """)
    return


@app.cell
def _(mo):
    folder = mo.ui.text(value="results", label="Folder", full_width=True)
    folder
    return (folder,)


@app.cell
def _(Path, ROOT, folder, mo):
    def _resolve(text):
        p = Path(text).expanduser()
        if p.is_absolute() or p.exists():
            return p.resolve()
        return (ROOT / p).resolve()               # relative to the project

    base = _resolve(folder.value)
    found = sorted({p.parent for p in base.rglob("run.npz")}) if base.is_dir() else []
    mo.stop(not found, mo.callout(
        f"No `run.npz` under `{base}`. Run `python run_sim.py` first, or change the folder.",
        kind="warn"))
    run_names = {str(p.relative_to(base)) if p != base else p.name: p for p in found}
    return (run_names,)


@app.cell
def _(mo, run_names):
    selected = mo.ui.multiselect(list(run_names), value=list(run_names)[:4], label="Runs")
    selected
    return (selected,)


@app.cell
def _(load_config, mo, np, run_names, selected):
    mo.stop(not selected.value, mo.callout("Select at least one run.", kind="warn"))
    runs, configs = {}, {}
    for _name in selected.value:
        _path = run_names[_name]
        configs[_name] = load_config(_path / "config_used.yaml")
        runs[_name] = dict(np.load(_path / "run.npz"))
    cfg = configs[selected.value[0]]               # limits and geometry for the plots
    return cfg, configs, runs


@app.cell
def _(configs, mo, runs, viz):
    mo.ui.table([{"run": n, "duration [s]": float(r["t"][-1]), **viz.summary(r, configs[n])}
                 for n, r in runs.items()], selection=None, pagination=False)
    return


@app.cell
def _(cfg, configs, mo, np):
    _same = all(np.array_equal(c.wheels.A, cfg.wheels.A)
                and float(c.wheels.h_max) == float(cfg.wheels.h_max)
                and c.requirements == cfg.requirements for c in configs.values())
    mo.stop(_same)
    mo.callout("The selected runs use different wheel or requirement settings. "
               "The limit lines are those of the first run.", kind="info")
    return


@app.cell
def _(cfg, runs, viz):
    time_series_fig = viz.time_series(runs, cfg)
    time_series_fig
    return (time_series_fig,)


@app.cell
def _(mo, runs):
    animated_run = mo.ui.dropdown(list(runs), value=list(runs)[0], label="Animate")
    # marimo shows outputs up to 8 MB; 800 frames is about 6 MB.
    max_frames = mo.ui.slider(100, 800, step=50, value=400, show_value=True,
                              label="Max frames")
    mo.hstack([animated_run, max_frames], justify="start", gap=3)
    return animated_run, max_frames


@app.cell
def _(animated_run, configs, max_frames, runs, viz):
    animation_fig = viz.attitude_animation(runs[animated_run.value],
                                           configs[animated_run.value],
                                           max_frames=max_frames.value)
    animation_fig
    return (animation_fig,)


if __name__ == "__main__":
    app.run()
