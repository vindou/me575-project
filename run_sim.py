"""Run one simulation and save the results.

Usage: python run_sim.py [config file] [output folder] --controller NAME

The plant config's `model:` key selects the plant and the controllers on offer:
  attitude (default)  open_loop, pd, mpc, mppi    (default controller: pd)
  thruster_1d         pid                         (default controller: pid)
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import jax.numpy as jnp

from cubesat_sim import run
from cubesat_sim.config import load_plant, plant_model
from cubesat_sim.quaternion import angle_between
from cubesat_sim.controller import CONTROLLERS_BY_MODEL
from cubesat_sim.thruster1d import run_thruster


ROOT = Path(__file__).resolve().parent

# Per model: extra controller names, default controller, default control settings file.
MODEL_DEFAULTS = {
    "attitude": (("open_loop",), "pd", ROOT / "config/control/controllers.yaml"),
    "thruster_1d": ((), "pid", ROOT / "config/control/thruster_pid.yaml"),
}


def available_controllers(model):
    return [*MODEL_DEFAULTS[model][0], *CONTROLLERS_BY_MODEL[model]]


def build_parser():
    """Read the controller choice while preserving the positional file arguments."""
    names = sorted({n for m in MODEL_DEFAULTS for n in available_controllers(m)})
    offered = "; ".join(f"{m}: {', '.join(available_controllers(m))}" for m in MODEL_DEFAULTS)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=Path,
                        default=ROOT / "config/cubesat_6u.yaml")
    parser.add_argument("output", nargs="?", type=Path,
                        help="Output folder; defaults to a new timestamped folder.")
    parser.add_argument("--controller", choices=names, default=None,
                        help=f"Controllers by model ({offered}). Default: pd or pid.")
    parser.add_argument("--control-config", type=Path, default=None,
                        help="Control settings file. Default: config/control/controllers.yaml "
                             "(attitude) or config/control/thruster_pid.yaml (thruster_1d).")
    return parser


def build_command(cfg, name, settings_path):
    """Build a registered attitude controller and return the settings needed to reproduce it."""
    if name == "open_loop":
        return None, None, {"controller": name}

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    target = settings["target"]
    axis = np.asarray(target["axis"], dtype=float)
    angle = float(target["angle_deg"])
    if (axis.shape != (3,) or not np.all(np.isfinite(axis))
            or np.linalg.norm(axis) == 0.0 or not np.isfinite(angle)):
        raise ValueError("Target requires a finite nonzero 3D axis and a finite angle_deg.")
    axis = axis / np.linalg.norm(axis)
    half_angle = np.deg2rad(angle) / 2.0
    q_target = jnp.asarray(np.append(axis * np.sin(half_angle), np.cos(half_angle)))
    parameters = settings["controllers"][name] or {}
    command = CONTROLLERS_BY_MODEL["attitude"][name](cfg, q_target, **parameters)
    metadata = {"controller": name, "target": target,
                "q_target": np.asarray(q_target).tolist(), "parameters": parameters}
    if hasattr(command, "design_info"):
        metadata["resolved_design"] = command.design_info
    return command, q_target, metadata


def build_thruster_command(cfg, name, settings_path):
    """Build a registered thruster controller; returns (command, x_ref, metadata)."""
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    x_ref = float(settings["target"]["x_ref_m"])
    if not np.isfinite(x_ref):
        raise ValueError("target.x_ref_m must be finite.")
    parameters = settings["controllers"][name] or {}
    command = CONTROLLERS_BY_MODEL["thruster_1d"][name](cfg, x_ref, **parameters)
    metadata = {"controller": name, "model": "thruster_1d", "target": settings["target"],
                "parameters": parameters}
    if hasattr(command, "design_info"):
        metadata["resolved_design"] = command.design_info
    return command, x_ref, metadata


def thruster_metrics(res, x_ref, dt):
    """Summary numbers of a thruster run (arrays from run_thruster, as numpy)."""
    x, v = res["x"], res["v"]
    D, F = res["D"][:-1], res["F"][:-1]         # the last row acts over no interval
    sign = np.sign(x_ref - x[0]) or 1.0          # overshoot = travel past x_ref, away from x0
    return {
        "final_error_m": float(x_ref - x[-1]),
        "overshoot_m": float(max(0.0, np.max(sign * (x - x_ref)))),
        "peak_speed_m_s": float(np.abs(v).max()),
        "time_full_thrust_s": float(np.count_nonzero(D >= 1.0) * dt),
        "time_zero_thrust_s": float(np.count_nonzero(D <= 0.0) * dt),
        "thrust_impulse_N_s": float(np.sum(F) * dt),
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        cfg = load_plant(args.config)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"Cannot load plant config: {exc}")
    model = plant_model(cfg.raw)
    _, default_name, default_settings = MODEL_DEFAULTS[model]
    name = args.controller or default_name
    if name not in available_controllers(model):
        parser.error(f"controller {name!r} does not apply to model {model!r}; "
                     f"choose from {', '.join(available_controllers(model))}")
    settings_path = args.control_config or default_settings
    builder = build_thruster_command if model == "thruster_1d" else build_command
    try:
        command, target, settings = builder(cfg, name, settings_path)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"Cannot configure controller: {exc}")
    # Keep separate runs by default; an explicit folder retains the old overwrite behavior.
    group = ROOT / "results" / ("thruster_1d" if model == "thruster_1d" else "")
    out_dir = args.output or (group / name / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if model == "thruster_1d":
        res = {k: np.asarray(v) for k, v in run_thruster(cfg, command=command).items()}
    else:
        res = {k: np.asarray(v) for k, v in run(cfg, command=command).items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "run.npz", **res)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False))
    (out_dir / "controller_used.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    if model == "thruster_1d":
        print_thruster_summary(res, cfg, name, target, out_dir)
    else:
        print_attitude_summary(res, cfg, name, target, out_dir)


def print_attitude_summary(res, cfg, name, q_target, out_dir):
    drift = np.linalg.norm(res["H_inertial"] - res["H_inertial"][0], axis=1).max()
    rpm = np.abs(res["wheel_speed"]).max() * 60.0 / (2.0 * np.pi)
    print(f"Controller:              {name}")
    print(f"Steps:                   {len(res['t']) - 1}")
    print(f"Final rotation angle:    {np.degrees(float(angle_between(res['q'][0], res['q'][-1]))):.3f} deg")
    if q_target is not None:
        final_error = np.degrees(float(angle_between(res["q"][-1], q_target)))
        print(f"Final pointing error:    {final_error:.6f} deg")
    print(f"Peak body rate:          {np.degrees(np.linalg.norm(res['w'], axis=1).max()):.4f} deg/s")
    print(f"Final body rate:         {np.degrees(np.linalg.norm(res['w'][-1])):.2e} deg/s")
    print(f"Peak wheel momentum:     {np.abs(res['h']).max():.2e} N·m·s "
          f"({100 * np.abs(res['h']).max() / cfg.wheels.h_max:.1f}% of limit)")
    print(f"Peak wheel speed:        {rpm:.0f} rpm")
    print(f"Momentum drift (check):  {drift:.2e} N·m·s")
    print(f"Results saved in:        {out_dir}/")


def print_thruster_summary(res, cfg, name, x_ref, out_dir):
    m = thruster_metrics(res, x_ref, cfg.sim.dt)
    print(f"Model:                   thruster_1d")
    print(f"Controller:              {name}")
    print(f"Steps:                   {len(res['t']) - 1}")
    print(f"Setpoint x_ref:          {x_ref:.4f} m")
    print(f"Final position error:    {m['final_error_m']:.4e} m")
    print(f"Overshoot:               {m['overshoot_m']:.4e} m")
    print(f"Peak speed:              {m['peak_speed_m_s']:.4e} m/s")
    print(f"Time at D = 1:           {m['time_full_thrust_s']:.2f} s")
    print(f"Time at D = 0:           {m['time_zero_thrust_s']:.2f} s")
    print(f"Thrust impulse:          {m['thrust_impulse_N_s']:.4e} N·s")
    print(f"Results saved in:        {out_dir}/")


if __name__ == "__main__":
    main()
