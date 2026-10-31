"""Run one simulation and save the results.

Usage: python run_sim.py [config file] [output folder] --controller pd
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import jax.numpy as jnp

from cubesat_sim import load_config, run
from cubesat_sim.quaternion import angle_between
from cubesat_sim.controller import CONTROLLERS


ROOT = Path(__file__).resolve().parent


def build_parser():
    """Read the controller choice while preserving the positional file arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path,
                        default=ROOT / "config/cubesat_6u.yaml")
    parser.add_argument("output", nargs="?", type=Path,
                        help="Output folder; defaults to a new timestamped folder.")
    parser.add_argument("--controller", choices=["open_loop", *CONTROLLERS], default="pd")
    parser.add_argument("--control-config", type=Path,
                        default=ROOT / "config/control/controllers.yaml")
    return parser


def build_command(cfg, name, settings_path):
    """Build a registered controller and return the settings needed to reproduce it."""
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
    command = CONTROLLERS[name](cfg, q_target, **parameters)
    metadata = {"controller": name, "target": target,
                "q_target": np.asarray(q_target).tolist(), "parameters": parameters}
    if hasattr(command, "design_info"):
        metadata["resolved_design"] = command.design_info
    return command, q_target, metadata


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        command, q_target, settings = build_command(cfg, args.controller, args.control_config)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"Cannot configure controller: {exc}")
    # Keep separate runs by default; an explicit folder retains the old overwrite behavior.
    out_dir = args.output or (ROOT / "results" / args.controller
                              / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    res = {k: np.asarray(v) for k, v in run(cfg, command=command).items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "run.npz", **res)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False))
    (out_dir / "controller_used.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    drift = np.linalg.norm(res["H_inertial"] - res["H_inertial"][0], axis=1).max()
    rpm = np.abs(res["wheel_speed"]).max() * 60.0 / (2.0 * np.pi)
    print(f"Controller:              {args.controller}")
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


if __name__ == "__main__":
    main()
