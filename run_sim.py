"""Run one simulation and save the results.

Usage: python run_sim.py [config file] [output folder]
"""
import sys
from pathlib import Path

import numpy as np
import yaml

from cubesat_sim import load_config, run
from cubesat_sim.quaternion import angle_between


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/cubesat_6u.yaml"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    cfg = load_config(cfg_path)
    res = {k: np.asarray(v) for k, v in run(cfg).items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "run.npz", **res)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False))

    drift = np.linalg.norm(res["H_inertial"] - res["H_inertial"][0], axis=1).max()
    rpm = np.abs(res["wheel_speed"]).max() * 60.0 / (2.0 * np.pi)
    print(f"Steps:                   {len(res['t']) - 1}")
    print(f"Final rotation angle:    {np.degrees(float(angle_between(res['q'][0], res['q'][-1]))):.3f} deg")
    print(f"Peak body rate:          {np.degrees(np.linalg.norm(res['w'], axis=1).max()):.4f} deg/s")
    print(f"Final body rate:         {np.degrees(np.linalg.norm(res['w'][-1])):.2e} deg/s")
    print(f"Peak wheel momentum:     {np.abs(res['h']).max():.2e} N·m·s "
          f"({100 * np.abs(res['h']).max() / cfg.wheels.h_max:.1f}% of limit)")
    print(f"Peak wheel speed:        {rpm:.0f} rpm")
    print(f"Momentum drift (check):  {drift:.2e} N·m·s")
    print(f"Results saved in:        {out_dir}/")


if __name__ == "__main__":
    main()
