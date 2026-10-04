# CubeSat attitude forward dynamics (Level 1)

A rigid 6U CubeSat with four reaction wheels. This package contains the plant model only.
It has no controller, no disturbances, and no sensors.

## Use

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_sim.py                 # uses config/cubesat_6u.yaml, writes results/
    python -m pytest                  # verification checks (about 20 s)
    python -m pytest -m slow          # also the long-horizon run and the notebook
    marimo edit notebooks/explore.py  # run scenarios: plots and 3D animation
    marimo edit notebooks/view_run.py # plots and 3D animation of saved runs

## Files

- `config/cubesat_6u.yaml`: all parameters, with a source tag for each value.
- `cubesat_sim/config.py`: loads and checks the file, builds the wheel matrix A.
- `cubesat_sim/dynamics.py`: equations of motion, wheel limits, RK4 step.
- `cubesat_sim/quaternion.py`: quaternion functions.
- `cubesat_sim/sim.py`: simulation loop (`jax.lax.scan`), `simulate()` and `run()`.
- `cubesat_sim/viz.py`: Plotly time series (with requirement limits) and 3D animation.
- `notebooks/explore.py`: marimo notebook to run scenarios and compare command laws.
- `notebooks/view_run.py`: marimo notebook to view and compare runs saved by `run_sim.py`.

## Conventions

- Quaternion: `[x, y, z, w]`, scalar last, body-to-inertial (same as scipy).
- Body rate `w`: in the body frame, rad/s.
- Wheel torque `u`: torque of the motor on the wheel. The body gets `-A u`.
- Wheel angle: `skew_angle_from_z_deg` is measured from body +z.

## Visualization

- `viz.time_series(runs, cfg, q_target=None)`: attitude, body rate, wheel momentum (% of
  h_max) and wheel torque, with the limits from the `requirements` section of the config.
  Pass `{name: result}` to compare runs. With `q_target`, the top plot is the pointing
  error in arcsec on a log scale.
- `viz.attitude_animation(res, cfg, q_target=None)`: the body in the inertial frame, body
  axes (x red, y green, z blue), wheels colored by h / h_max with a bar panel, and a trail
  of the body z axis. Frame rate and playback speed come from `visualization` in the config.
- In the notebook, add a controller to `COMMANDS`; it then appears in the comparison.

## Tests

- `test_quaternion.py`: quaternion functions against scipy and their defining properties.
- `test_config.py`: every rejected input, derived values, wheel geometry, pytree structure.
- `test_dynamics.py`: identities of one state derivative (momentum, energy, Euler's
  equations) for random states and random non-diagonal inertias; wheel limit cases.
- `test_physics.py`: whole runs against closed-form results and conservation laws.
- `test_integrator.py`: RK4 convergence order, time grid, long-horizon drift.
- `test_limits.py`: torque and momentum limits over whole runs.
- `test_jax.py`: compilation, gradients against finite differences and closed forms, vmap.
- `test_viz.py`: the figures draw what the simulation computed (axes, momentum, limits).
- `test_regression.py`: three runs against `tests/data/golden_runs.npz`. After an intended
  model change: `REGENERATE_GOLDEN=1 python -m pytest tests/test_regression.py`.

## JAX

The model is written in JAX with double precision on (set when `cubesat_sim` is imported).

- `run(cfg, command, external_torque)` compiles and runs one simulation. The first call
  compiles (about 0.5 s); later calls with the same function objects reuse the compiled code.
- `simulate(body, wheels, x0, dt, steps, command, external_torque)` is the pure function
  under `run`. Call it inside your own `jax.jit`, `jax.grad` or `jax.vmap`. `body` and
  `wheels` are pytrees, so you can differentiate with respect to inertia, wheel axes, limits,
  the initial state, or anything a command function closes over (gains, torque profiles).
- `dt` and `t_end` are static: changing them recompiles.
- `command` and `external_torque` must use `jax.numpy`, not `numpy`, and must not use
  Python `if` on traced values (use `jnp.where`).
- Gradients through the wheel limits are zero while a limit is active (clip and blocking).

## How to add a controller later

`run(cfg, command=f)` accepts a function `f(t, x)` that returns the four wheel torque
commands. The open-loop profile in the file is the default for this function.
