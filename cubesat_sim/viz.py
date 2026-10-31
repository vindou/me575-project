"""Plotly figures for simulation results.

time_series()        linked plots of attitude, body rate, wheel momentum and wheel torque,
                     with the requirement limits from the config. Pass several runs to
                     compare them (for example, two controllers).
attitude_animation() 3D animation of the body in the inertial frame.

Both take result dicts from run() / simulate() and the Config, and return a plotly Figure.
Nothing here changes the dynamics.
"""
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import quaternion as quat

AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")                  # x, y, z
WHEEL_COLORS = ("#9467bd", "#ff7f0e", "#17becf", "#8c564b")
RUN_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b")
LIMIT = dict(color="rgba(0,0,0,0.5)", dash="dash", width=1)
WARNING = dict(color="rgba(0,0,0,0.35)", dash="dot", width=1)
MAX_POINTS = 20000                                               # per time-series trace
# Wheel momentum h / h_max: blue negative, grey zero, red positive.
MOMENTUM_COLORS = [[0.0, "#2166ac"], [0.5, "#b0b0b0"], [1.0, "#b2182b"]]

_rotations = jax.jit(jax.vmap(quat.rotation_matrix))
_angles_to_fixed = jax.jit(jax.vmap(quat.angle_between, in_axes=(None, 0)))
_angles_pairwise = jax.jit(jax.vmap(quat.angle_between))


def rotation_matrices(q):
    """Body-to-inertial rotation matrix for each row of q, shape (N, 3, 3)."""
    return np.asarray(_rotations(jnp.asarray(q)))


def attitude_angle(q, q_ref):
    """Angle (rad) from q_ref to each row of q. q_ref is one quaternion or one per row."""
    q, q_ref = jnp.asarray(q), jnp.asarray(q_ref)
    f = _angles_pairwise if q_ref.ndim == 2 else _angles_to_fixed
    return np.asarray(f(q_ref, q))


def _numpy(res):
    return {k: np.asarray(v) for k, v in res.items()}


def summary(res, cfg):
    """Key numbers of one run, as a dict (one table row)."""
    r = _numpy(res)
    rate = np.degrees(np.linalg.norm(r["w"], axis=1))
    cap = cfg.requirements.get("max_slew_rate_deg_s", np.inf)
    dt = float(r["t"][1] - r["t"][0]) if len(r["t"]) > 1 else 0.0
    H = r["H_inertial"]
    return {
        "final rotation [deg]": round(float(np.degrees(attitude_angle(r["q"][-1:], r["q"][0])[0])), 4),
        "peak rate [deg/s]": round(float(rate.max()), 4),
        "time above rate limit [s]": round(float((rate > cap).sum() * dt), 2),
        "final rate [deg/s]": float(f"{rate[-1]:.3g}"),
        "peak |h| [% h_max]": round(float(100 * np.abs(r["h"]).max() / float(cfg.wheels.h_max)), 1),
        "peak |u| [mN·m]": round(float(1e3 * np.abs(r["u"]).max()), 3),
        "momentum drift [N·m·s]": float(f"{np.linalg.norm(H - H[0], axis=1).max():.2e}"),
    }


# --- time series -----------------------------------------------------------------------

def time_series(runs, cfg, q_target=None, height=900):
    """Linked time series for one run or several.

    runs     : a result dict, or {name: result dict} to compare runs.
    q_target : target attitude (one quaternion, or one per sample). If given, the top plot
               is the pointing error in arcsec on a log scale, with the settling band.
               If not, it is the rotation from the initial attitude in degrees.

    One run: each plot shows the components (x, y, z or wheels 1-4).
    Several runs: each plot shows one line per run (vector norm or the largest wheel),
    and clicking a run in the legend hides it in every plot.
    """
    if "t" in runs:
        runs = {"run": runs}
    runs = {name: _numpy(r) for name, r in runs.items()}
    single = len(runs) == 1
    req = cfg.requirements
    h_max = float(cfg.wheels.h_max)
    u_max = 1e3 * float(cfg.wheels.u_max)
    n_wheels = cfg.wheels.count
    pointing = q_target is not None

    titles = ("Pointing error [arcsec]" if pointing else "Rotation from start [deg]",
              "Body rate [deg/s]" if single else "Body rate |ω| [deg/s]",
              "Wheel momentum [% of h_max]" if single else "Largest wheel momentum [% of h_max]",
              "Wheel torque [mN·m]" if single else "Largest wheel torque [mN·m]")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=titles)

    def add(row, x, y, name, color, dash=None, show=True, group=None):
        stride = max(1, len(x) // MAX_POINTS)
        fig.add_trace(go.Scatter(
            x=x[::stride], y=y[::stride], name=name, mode="lines",
            line=dict(color=color, width=1.6, dash=dash),
            legend="legend" if not single else ("legend" if row == 1 else f"legend{row}"),
            legendgroup=group or name, showlegend=show), row=row, col=1)

    for n, (name, r) in enumerate(runs.items()):
        t = r["t"]
        color = RUN_COLORS[n % len(RUN_COLORS)]
        ref = r["q"][0] if q_target is None else q_target
        angle = attitude_angle(r["q"], ref)
        error = np.degrees(angle) * 3600.0 if pointing else np.degrees(angle)
        w = np.degrees(r["w"])
        h_pct = 100.0 * r["h"] / h_max
        u = 1e3 * r["u"]

        if single:
            add(1, t, error, "error" if pointing else "angle", "black")
            for i in range(3):
                add(2, t, w[:, i], f"ω{'xyz'[i]}", AXIS_COLORS[i])
            add(2, t, np.linalg.norm(w, axis=1), "|ω|", "black", dash="dot")
            for i in range(n_wheels):
                add(3, t, h_pct[:, i], f"wheel {i + 1}", WHEEL_COLORS[i % 4])
                add(4, t, u[:, i], f"wheel {i + 1}", WHEEL_COLORS[i % 4],
                    show=False, group=f"wheel {i + 1}")
        else:
            add(1, t, error, name, color)
            add(2, t, np.linalg.norm(w, axis=1), name, color, show=False)
            add(3, t, np.abs(h_pct).max(axis=1), name, color, show=False)
            add(4, t, np.abs(u).max(axis=1), name, color, show=False)

    def limit(row, y, text, style=LIMIT, both_signs=single):
        fig.add_hline(y=y, row=row, col=1, line=style, annotation_text=text,
                      annotation_position="top left", annotation_font_size=10)
        if both_signs:
            fig.add_hline(y=-y, row=row, col=1, line=style)

    if pointing and "settle_band_arcsec" in req:
        limit(1, req["settle_band_arcsec"], "settling band", both_signs=False)
    if pointing and "pointing_error_arcsec_rms" in req:
        limit(1, req["pointing_error_arcsec_rms"], "RMS requirement", WARNING, both_signs=False)
    if "max_slew_rate_deg_s" in req:
        limit(2, req["max_slew_rate_deg_s"], "slew rate limit")
    limit(3, 100.0, "h_max")
    if "wheel_momentum_warning_frac" in req:
        limit(3, 100.0 * req["wheel_momentum_warning_frac"], "warning", WARNING)
    limit(4, u_max, "u_max")

    if pointing:
        fig.update_yaxes(type="log", row=1, col=1)
        # Log-axis annotations use log10 coordinates; shape positions use data values.
        # Keeping a threshold label at y=60 would otherwise stretch the axis to 1e60.
        for annotation in fig.layout.annotations:
            if annotation.yref == "y" and annotation.y is not None:
                annotation.y = float(np.log10(annotation.y))
    fig.update_xaxes(title_text="Time [s]", row=4, col=1)

    # One legend per plot for a single run; one shared legend for several runs.
    legends = {}
    for row in range(1, 5 if single else 2):
        top = fig.layout[f"yaxis{'' if row == 1 else row}"].domain[1]
        legends["legend" if row == 1 else f"legend{row}"] = dict(
            x=1.01, y=top, xanchor="left", yanchor="top", tracegroupgap=2, font_size=11)
    fig.update_layout(height=height, template="plotly_white", hovermode="x unified",
                      margin=dict(l=60, r=20, t=40, b=40), uirevision="keep", **legends)
    return fig


# --- 3D animation ----------------------------------------------------------------------

def _box(size):
    """Vertices, triangles and edge polyline of a box centered at the origin."""
    half = np.asarray(size, dtype=float) / 2.0
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    tri = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    E = np.concatenate([np.vstack([V[i], V[j], [np.nan] * 3]) for i, j in edges])
    return V, tri, E


def _cylinder(center, axis, radius, thickness, n=20):
    """Vertices and triangles of a closed cylinder (a wheel)."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    e1 = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    a = 2 * np.pi * np.arange(n) / n
    ring = radius * (np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2))
    bottom = center - 0.5 * thickness * axis
    top = center + 0.5 * thickness * axis
    V = np.vstack([bottom + ring, top + ring, bottom, top])
    k = np.arange(n)
    k1 = (k + 1) % n
    cb, ct = 2 * n, 2 * n + 1
    tri = np.vstack([np.column_stack([k, k1, n + k]),
                     np.column_stack([n + k, k1, n + k1]),
                     np.column_stack([np.full(n, cb), k1, k]),
                     np.column_stack([np.full(n, ct), n + k, n + k1])])
    return V, tri


def _xyz(P):
    P = np.asarray(P, dtype=np.float32)
    return dict(x=P[:, 0], y=P[:, 1], z=P[:, 2])


def attitude_animation(res, cfg, q_target=None, max_frames=600, height=700):
    """3D animation of the body in the inertial frame.

    Shows the body box, the body axes (x red, y green, z blue), the four wheels colored by
    stored momentum (h / h_max, blue negative, red positive) and a trail of the body z axis.
    If q_target is given (one quaternion), the target body axes are drawn as dotted lines.

    Playback speed and frame rate come from cfg.visualization. Long runs are shown with at
    most max_frames frames, each held longer, so the playback speed stays the same.
    """
    r = _numpy(res)
    v = cfg.visualization
    t, q, h = r["t"], r["q"], r["h"]
    fps = float(v.get("fps", 30))
    speed = float(v.get("playback_speed", 1.0))
    L = float(v.get("axis_length_m", 0.25))
    h_max = float(cfg.wheels.h_max)
    A = np.asarray(cfg.wheels.A)

    # Frame samples: one per 1/fps s of playback, at most max_frames, always the last one.
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    stride = max(1, int(round(speed / fps / dt)), int(np.ceil((len(t) - 1) / max_frames)))
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)
    frame_ms = 1000.0 * stride * dt / speed

    # Geometry in the body frame.
    box_V, box_tri, box_E = _box(v.get("body_size_m", (0.1, 0.2, 0.3)))
    wheel_V, wheel_tri, wheel_of_vertex = [], [], []
    center = np.asarray(v.get("wheel_cluster_center_m", (0.0, 0.0, 0.0)), dtype=float)
    for i in range(A.shape[1]):
        Vi, Ti = _cylinder(center + float(v.get("wheel_offset_m", 0.03)) * A[:, i], A[:, i],
                           float(v.get("wheel_radius_m", 0.02)),
                           float(v.get("wheel_thickness_m", 0.01)))
        wheel_tri.append(Ti + sum(len(x) for x in wheel_V))
        wheel_V.append(Vi)
        wheel_of_vertex.append(np.full(len(Vi), i))
    wheel_V, wheel_tri = np.vstack(wheel_V), np.vstack(wheel_tri)
    wheel_of_vertex = np.concatenate(wheel_of_vertex)

    R = rotation_matrices(q[idx])                                  # (F, 3, 3)
    angle = np.degrees(attitude_angle(q[idx], q[0] if q_target is None else q_target))
    rate = np.degrees(np.linalg.norm(r["w"][idx], axis=1))
    h_frac = h[idx] / h_max                                        # (F, N)
    z_tip = L * R[:, :, 2]                                         # (F, 3)
    trail_frames = max(1, int(round(float(v.get("trail_length_s", 10.0)) / (stride * dt))))
    angle_name = "error" if q_target is not None else "rotation"

    def moving(f):
        """Traces that change from frame to frame, in a fixed order."""
        Rf = R[f]
        tips = L * Rf.T                                            # rows: body axes
        traces = [
            go.Mesh3d(**_xyz(box_V @ Rf.T)),
            go.Scatter3d(**_xyz(box_E @ Rf.T)),
            *[go.Scatter3d(**_xyz(np.vstack([np.zeros(3), tips[i]]))) for i in range(3)],
            go.Mesh3d(**_xyz(wheel_V @ Rf.T),
                      intensity=h_frac[f, wheel_of_vertex].astype(np.float32)),
            go.Scatter3d(**_xyz(z_tip[max(0, f - trail_frames):f + 1])),
            go.Bar(y=(100 * h_frac[f]).astype(np.float32),
                   marker=dict(color=h_frac[f].astype(np.float32))),
        ]
        return traces

    def title(f):
        return (f"t = {t[idx[f]]:7.2f} s    {angle_name} {angle[f]:7.3f}°    "
                f"|ω| {rate[f]:.3f}°/s    largest |h| {100 * np.abs(h_frac[f]).max():.0f}% of h_max")

    # Initial traces with their styles; frames only update coordinates and colors.
    first = moving(0)
    first[0].update(i=box_tri[:, 0], j=box_tri[:, 1], k=box_tri[:, 2], color="lightgray",
                    opacity=0.08, flatshading=True, hoverinfo="skip", name="body")
    first[1].update(mode="lines", line=dict(color="#444", width=3), hoverinfo="skip",
                    name="body edges")
    for i in range(3):
        first[2 + i].update(mode="lines+text", text=["", f"{'xyz'[i]}<sub>b</sub>"],
                            textfont=dict(color=AXIS_COLORS[i], size=14),
                            line=dict(color=AXIS_COLORS[i], width=7), hoverinfo="skip",
                            name=f"body {'xyz'[i]}")
    first[5].update(i=wheel_tri[:, 0], j=wheel_tri[:, 1], k=wheel_tri[:, 2],
                    intensitymode="vertex", colorscale=MOMENTUM_COLORS, cmin=-1.0, cmax=1.0,
                    showscale=False, flatshading=True, hoverinfo="skip", name="wheels")
    first[6].update(mode="lines", line=dict(color=AXIS_COLORS[2], width=3),
                    opacity=0.6, hoverinfo="skip", name="z trail")
    wheel_names = [str(i + 1) for i in range(A.shape[1])]
    first[7].update(x=wheel_names, marker=dict(colorscale=MOMENTUM_COLORS, cmin=-1.0, cmax=1.0,
                                               line=dict(color="#444", width=1)),
                    hovertemplate="wheel %{x}: %{y:.1f}% of h_max<extra></extra>", name="wheels")

    static = []
    for i in range(3):                                            # inertial axes
        e = np.zeros(3)
        e[i] = 1.2 * L
        static.append(go.Scatter3d(**_xyz(np.vstack([np.zeros(3), e])), mode="lines+text",
                                   text=["", f"{'XYZ'[i]}<sub>I</sub>"],
                                   line=dict(color="#999", width=2, dash="dash"),
                                   textfont=dict(color="#777"), hoverinfo="skip",
                                   name=f"inertial {'XYZ'[i]}"))
    if q_target is not None:                                      # target body axes
        Rt = np.asarray(quat.rotation_matrix(jnp.asarray(q_target)))
        for i in range(3):
            static.append(go.Scatter3d(**_xyz(np.vstack([np.zeros(3), L * Rt[:, i]])),
                                       mode="lines", hoverinfo="skip",
                                       line=dict(color=AXIS_COLORS[i], width=3, dash="dot"),
                                       name=f"target {'xyz'[i]}"))

    moving_ids = list(range(len(first)))
    frames = [go.Frame(name=str(f), data=moving(f), traces=moving_ids,
                       layout=dict(title_text=title(f))) for f in range(len(idx))]

    extent = 1.15 * max(L * 1.2, np.linalg.norm(box_V, axis=1).max(),
                        np.linalg.norm(wheel_V, axis=1).max())
    axis = dict(range=[-extent, extent], autorange=False, showbackground=False,
                zerolinecolor="#ddd", gridcolor="#eee")
    play = dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True,
                transition=dict(duration=0), mode="immediate")
    jump = dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0),
                mode="immediate")
    label_every = max(1, len(idx) // 12)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], horizontal_spacing=0.06,
                        specs=[[{"type": "scene"}, {"type": "xy"}]])
    for trace in first + static:
        fig.add_trace(trace, row=1, col=2 if isinstance(trace, go.Bar) else 1)
    fig.frames = frames
    warn = cfg.requirements.get("wheel_momentum_warning_frac")
    for y, style in [(100, LIMIT), (-100, LIMIT)] + ([(100 * warn, WARNING), (-100 * warn, WARNING)]
                                                     if warn else []):
        # add_hline does not work in a figure that also has a 3D scene.
        fig.add_shape(type="line", xref="x domain", x0=0, x1=1, yref="y", y0=y, y1=y,
                      line=style)
    fig.update_yaxes(range=[-110, 110], title_text="wheel momentum [% of h_max]",
                     zeroline=True, zerolinecolor="#888", row=1, col=2)
    fig.update_xaxes(title_text="wheel", type="category", row=1, col=2)
    fig.update_layout(
        title=dict(text=title(0), font=dict(family="monospace", size=13)),
        height=height, template="plotly_white", showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(xaxis=dict(title="X_I [m]", **axis), yaxis=dict(title="Y_I [m]", **axis),
                   zaxis=dict(title="Z_I [m]", **axis), aspectmode="cube",
                   camera=dict(eye=dict(x=1.4, y=1.1, z=0.8))),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0.0, xanchor="left",
                          yanchor="top", pad=dict(t=10, r=10), showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(jump, frame=dict(duration=0,
                                                                            redraw=False))])])],
        sliders=[dict(active=0, x=0.16, y=0.0, len=0.6, xanchor="left", yanchor="top",
                      pad=dict(t=10), ticklen=0, minorticklen=0, font=dict(size=10),
                      currentvalue=dict(prefix="t = ", suffix=" s", xanchor="right"),
                      steps=[dict(method="animate", args=[[str(f)], jump],
                                  label=f"{t[k]:.1f}" if f % label_every == 0 else "")
                             for f, k in enumerate(idx)])],
    )
    return fig
