# CubeSat attitude simulation

A rigid 6U CubeSat with four reaction wheels, a nonlinear plant, and a fixed-target PD
controller. The default simulations have no external disturbances or sensor models.

## Use

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_sim.py --controller pd # uses the PD settings and a new results/pd/<timestamp>/ folder
    python run_sim.py --controller open_loop # uses the plant file's torque profile
    python -m pytest                  # verification checks (about 20 s)
    python -m pytest -m slow          # also the long-horizon run and the notebook
    marimo edit notebooks/explore.py  # run scenarios: plots and 3D animation
    marimo edit notebooks/view_run.py # plots and 3D animation of saved runs

On Windows, use `.venv\Scripts\python.exe` instead of `python` to use the project environment.

## Selecting and configuring controllers

`run_sim.py` defaults to `pd`. Set the absolute target axis/angle and each method's
parameters in `config/control/controllers.yaml`. PD now accepts scalar `wn_rad_s`
and `zeta` and computes `Kp = wn_rad_s**2 * J` and `Kd = 2*zeta*wn_rad_s * J` once
from the selected plant, including off-diagonal inertia terms. `J` remains constant
during each run. This is local small-angle tuning, not a guarantee of a given
settling time, slew-rate bound, or global nonlinear performance. To use manual
3x3 `Kp` and `Kd` instead, supply both and omit `wn_rad_s` and `zeta`.
The actual resolved gain matrices are saved in `controller_used.yaml`.
Plant initial conditions and duration remain in `config/cubesat_6u.yaml`.

    python run_sim.py --controller pd
    python run_sim.py config/cubesat_6u.yaml results/my_pd_run --controller pd
    python run_sim.py --controller pd --control-config config/control/controllers.yaml

Each run saves `run.npz`, `config_used.yaml`, and `controller_used.yaml`, including the
selected controller, target quaternion, and parameters. Without an output argument,
runs get separate timestamped folders. An explicitly reused output folder overwrites
the corresponding files. The runner prints final pointing error for closed-loop runs.
The saved-run notebook currently plots rotation from the initial attitude; target-error
plots require passing the saved target to `viz.time_series(..., q_target=...)`.

### Coupled PD experiment

`config/cubesat_6u_coupled.yaml` is an assumed test scenario with a constant
non-diagonal inertia, a nonidentity initial attitude, initial body rate, and wheel
momentum. It uses the existing nonlinear rigid-body equations; it does not add
flexibility, environment torques, sensor noise, or model uncertainty. Both control
files below specify the same absolute 60-degree target about `[1,-2,3]` and use
600 seconds of simulation. The initial target error is about 73.88 degrees.

    python run_sim.py config/cubesat_6u_coupled.yaml --controller pd --control-config config/control/coupled_fast.yaml
    python run_sim.py config/cubesat_6u_coupled.yaml --controller pd --control-config config/control/coupled_slow.yaml

The fast file uses `wn_rad_s: 0.2`; the slow file uses `0.02`. Both use `zeta: 1.0`.
Reducing wn also reduces damping gain and can expose gyroscopic effects from
nonzero total angular momentum. Neither gain setting is a hard rate limiter.
The 0.5 deg/s rate threshold and 60 arcsec pointing band are project assumptions,
not validated mission requirements or claims from the reference paper.

`analyze_pd_study.py STUDY_FOLDER` reads saved `fast` and `slow` subfolders (and
optional `nominal` and `diagnostic_zero_momentum` subfolders), writes `metrics.yaml`,
and creates `comparison.html` with actual target-error curves and 3D animations.
Settling there is measured from command application, requiring the sampled error
and rate to remain within their bands through the run end with a ten-second
observed hold. It does not validate the proposed twenty-minute pointing hold.

To add a controller without editing the runner:

1. Write a factory in `cubesat_sim/controller.py`, such as
   `my_controller(cfg, q_target, gain)`, returning `command(t, x)`.
2. Add `"my_method": my_controller` to `CONTROLLERS` at the bottom of that file.
3. Add `my_method: {gain: ...}` under `controllers` in the control settings file.
4. Run `python run_sim.py --controller my_method`.

Every command must use JAX-compatible operations and return four wheel torque commands.
Its state is `x = [q (4), omega (3), h (4)]`, with angular velocity in rad/s. Factories may
accept different named parameters; the runner forwards only the selected method's
settings. The current interface supports stateless feedback; controllers with integral
or other persistent state require extending the simulation state. The explorer notebook's
`COMMANDS` registry is separate and still requires registering interactive methods there.

## Files

- `config/cubesat_6u.yaml`: all parameters, with a source tag for each value.
- `config/control/controllers.yaml`: fixed target and controller-specific parameters.
- `cubesat_sim/controller.py`: controller factories and the runner's `CONTROLLERS` registry.
- `cubesat_sim/config.py`: loads and checks the file, builds the wheel matrix A.
- `cubesat_sim/dynamics.py`: equations of motion, wheel limits, RK4 step.
- `cubesat_sim/quaternion.py`: quaternion functions.
- `cubesat_sim/sim.py`: simulation loop (`jax.lax.scan`), `simulate()` and `run()`.
- `cubesat_sim/viz.py`: Plotly time series (with requirement limits) and 3D animation.
- `notebooks/explore.py`: marimo notebook to run scenarios and compare command laws.
- `notebooks/view_run.py`: marimo notebook to view and compare runs saved by `run_sim.py`.

## Conventions

The conventions follow Navabi & Hosseini, *Spacecraft quaternion based attitude
input-output feedback linearization control using reaction wheels* (the PDF in the repo).
Code symbols map to the paper as `J` = I_S, `A` = L, `h` = I_W ω_W, `u` = T_w.

- Quaternion: `[x, y, z, w]`, scalar last, body-to-inertial (same as scipy). This is the
  paper's Q = [q1 q2 q3 q4] with q4 = cos(α/2) (eq. 2), and q̇ = ½[Ω′]Q (eqs. 3–4).
- Body rate `w`: in the body frame, rad/s.
- Wheel torque `u`: the paper's T_w, torque of each wheel on the body. The body gets
  `T_C = A u` (eq. 12) and the wheel momentum changes by `h_dot = -u` (eq. 10).
- Wheel geometry: the paper's pyramid (eq. 13). `beta_from_xy_deg` (β) is measured from
  the body xy plane; `azimuth_deg` starts at the paper's θ = 45° and adds 90° per wheel.

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
commands T_w. For a body torque command T_C, the paper's allocation (eq. 21) gives T_w. The open-loop profile in the file is the default for this function.
