# CubeSat attitude simulation

A rigid 6U CubeSat with four reaction wheels, a nonlinear plant, and fixed-target PD,
nonlinear MPC and MPPI controllers. The default simulations have no external disturbances or sensor models.
A second plant, a 1D one-directional thruster with a PID controller, is described in
[1D thruster plant](#1d-thruster-plant).

## Use

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_sim.py --controller pd # uses the PD settings and a new results/pd/<timestamp>/ folder
    python run_sim.py --controller open_loop # uses the plant file's torque profile
    python run_sim.py config/thruster_1d.yaml --controller pid   # 1D thruster plant
    python -m pytest                  # verification checks (about 20 s)
    python -m pytest -m slow          # also the long-horizon run and the notebook
    marimo edit notebooks/explore.py  # run scenarios: plots and 3D animation
    marimo edit notebooks/view_run.py # plots and 3D animation of saved runs
    marimo edit notebooks/compare_controllers.py # PD vs MPC vs MPPI: metrics, requirements, plots

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

### Nonlinear shooting MPC

`--controller mpc` (`cubesat_sim/mpc.py`) replans every `control_interval_s` (default
0.2 s) and holds each torque in between. Each solve optimizes a piecewise-constant
wheel torque sequence over `horizon_steps` intervals (default 25, a 5 s horizon) with
optax L-BFGS, differentiating a rollout of the plant's own RK4 model with `jax.grad`.
Torques are parameterized as `u = u_max * tanh(z)`, so they never exceed the limit, and
each solve warm-starts from the previous plan shifted by one interval.

The cost penalizes attitude error, `(rate_weight_s * |omega|)^2`, torque, and a weighted
terminal state. Two optional soft penalties grow quickly past `max_rate_deg_s` (body rate)
and `momentum_warning_frac * h_max` (each wheel). These are penalties, not hard
constraints. The prediction model assumes no external torque. The default
`controllers.yaml` leaves the rate limit off: at 0.5 deg/s the 50 deg slew needs about
100 s, longer than the 60 s run. The coupled control files turn it on.

    python run_sim.py --controller mpc
    python run_sim.py config/cubesat_6u_coupled.yaml --controller mpc --control-config config/control/coupled_slow.yaml

MPC runs take longer than PD: about 10 s for the 60 s default run and about 75 s for the
600 s coupled run on a laptop CPU.

### MPPI

`--controller mppi` (`cubesat_sim/mppi.py`) uses the same plan, prediction model, cost
and torque hold as MPC (shared in `cubesat_sim/horizon.py`) but is sampling-based: no
gradients. Each replan runs `iterations` rounds. A round rolls out `samples` noisy copies
of the plan in parallel (`jax.vmap`), weights them by `softmax(-(S - S_min) / (temperature
* (S_median - S_min)))`, and averages. The noise sigma shrinks from `noise_sigma` to
`final_sigma_fraction * noise_sigma` over the rounds; the residual pointing jitter scales
with the final sigma. Noise and plan are projected off the wheel null space; without that
the wheel momentum drifts to saturation. The default horizon is 2 s (10 intervals): longer
horizons add sampling noise to the first torque, the only one applied. Runs are
reproducible for a given `seed`.

    python run_sim.py --controller mppi
    python run_sim.py config/cubesat_6u_coupled.yaml --controller mppi --control-config config/control/coupled_slow.yaml

Measured with the default settings (attitude error after settling): about 3 arcsec RMS on
the default slew (seeds 0-5) and about 4 arcsec on the coupled run, against ~0 for MPC. The
runs take about 4 s and 30 s.

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

To add a controller without editing the runner:

1. Write a factory in `cubesat_sim/controller.py`, such as
   `my_controller(cfg, q_target, gain)`, returning `command(t, x)`.
2. Add `"my_method": my_controller` to `CONTROLLERS` at the bottom of that file.
3. Add `my_method: {gain: ...}` under `controllers` in the control settings file.
4. Run `python run_sim.py --controller my_method`.

Every command must use JAX-compatible operations and return four wheel torque commands.
Its state is `x = [q (4), omega (3), h (4)]`, with angular velocity in rad/s. Factories may
accept different named parameters; the runner forwards only the selected method's
settings. A controller with persistent state (integral terms, warm starts) instead sets
`command.init_state` (a JAX pytree) and is called as `command(t, x, state) -> (u, state)`;
the simulator carries the state between steps. MPC and MPPI use this interface. The explorer notebook's
`COMMANDS` registry is separate and still requires registering interactive methods there.

## 1D thruster plant

The `model:` key in a plant config selects the plant: `attitude` (the CubeSat files; also
the default when the key is missing) or `thruster_1d`. `run_sim.py` offers only the
controllers registered for the selected model (`CONTROLLERS_BY_MODEL` in
`cubesat_sim/controller.py`): `open_loop`, `pd`, `mpc`, `mppi` for attitude, `pid` for the
thruster. The default controller is `pd` or `pid`, and the default control file is
`config/control/controllers.yaml` or `config/control/thruster_pid.yaml`. Without an output
folder, thruster runs go to `results/thruster_1d/<controller>/<timestamp>/`.

    python run_sim.py config/thruster_1d.yaml --controller pid
    python run_sim.py config/thruster_1d.yaml results/my_pid --control-config config/control/thruster_pid.yaml

Model (`cubesat_sim/thruster1d.py`), state `x = [position, velocity]`:

    m * x_ddot = F_avg + F_d,   D = sat(u / F_max, 0, 1),   F_avg = D * F_max

The command is the requested force `u` (N); the plant applies the `[0, F_max]` limit and
reports `F_cmd`, `F` (applied), `D` and `F_d` per step. Throttle is continuous, not on/off.

**One direction only.** The thruster pushes in +x and cannot brake (in the full system the
reaction wheels point it). Only a negative `F_d` slows the mass. With `F_d = 0`, any
overshoot is permanent, and a setpoint can be held only by arriving with zero velocity.
The default file uses `F_d_N: -0.01` so the PID can hold `x_ref` at `D = 0.1`.

Plant file `config/thruster_1d.yaml` (all placeholder values are tagged `[ASSUMPTION]`):

| Field | Default | Meaning |
|---|---|---|
| `model` | `thruster_1d` | plant selection |
| `mass_kg` | 11.0 | moving mass (same as the 6U CubeSat) |
| `F_max_N` | 0.1 | thrust at `D = 1` (small cold-gas class) |
| `x0_m`, `v0_m_s` | 0.0, 0.0 | initial position and velocity |
| `dt_s`, `t_end_s` | 0.05, 300.0 | step and duration; `t_end_s` a whole multiple of `dt_s` |
| `disturbance.F_d_N` | -0.01 | constant external force |
| `disturbance.segments` | one pulse | optional `{t_start_s, t_end_s, F_N}` additions, active for `t_start_s <= t < t_end_s` |

The default pulse (-0.02 N from 200 s to 210 s) shows disturbance rejection during the hold.
The loader checks positive mass and `F_max`, finite values, `t_end` a multiple of `dt`, and
`t_start < t_end`. The disturbance becomes a hashable `DisturbanceProfile`, so repeated runs
with an equal profile reuse the compiled code.

PID (`cubesat_sim/thruster_control.py`), set in `config/control/thruster_pid.yaml`:
`target.x_ref_m` is the setpoint; `u = Kp e + Ki ∫e dt + Kd e_dot` with `e = x_ref - x` and
`e_dot = -v` (fixed setpoint, so no numerical differentiation). The integral is the
controller's carried state. Gains come from `pole_rad_s` (all three linear closed-loop poles
at `-p`: `Kp = 3 m p^2`, `Ki = m p^3`, `Kd = 3 m p`) or from manual `Kp`, `Ki`, `Kd`. The
resolved gains are saved in `controller_used.yaml`. Anti-windup (`anti_windup: true`)
stops the integration while `D = 1` with `e > 0` or `D = 0` with `e < 0`.

The linear pole design ignores the saturation and the missing braking, so larger or faster
moves overshoot more than they would on a linear plant: with the defaults (`x_ref = 1 m`,
`p = 0.05`) the overshoot is about 0.2 m. `p = 0.02` does not overshoot but takes several
hundred seconds. The runner prints final position error, overshoot, peak speed, time at
`D = 1` and at `D = 0`, and total thrust impulse.

`viz.thruster_time_series(res, x_ref=...)` plots x(t), v(t) and D(t) for one run or a
`{name: res}` dict. The 3D attitude animation and `viz.time_series` do not apply, and the
`view_run` notebook skips thruster runs. Load a thruster run by hand:

    import numpy as np; from cubesat_sim import viz
    res = dict(np.load("results/thruster_1d/pid/<timestamp>/run.npz"))
    viz.thruster_time_series(res, x_ref=1.0).show()

The simulation core `sim.simulate_plant(step, limit, observe, x0, dt, steps, command)` is
shared by both plants; `simulate()`/`run()` (attitude) and
`thruster1d.simulate_thruster()`/`run_thruster()` are thin wrappers around it.

## Files

- `config/cubesat_6u.yaml`: all parameters, with a source tag for each value.
- `config/control/controllers.yaml`: fixed target and controller-specific parameters.
- `cubesat_sim/controller.py`: controller factories and the runner's `CONTROLLERS` registry.
- `cubesat_sim/horizon.py`: shared receding-horizon pieces (cost, checks, torque hold).
- `cubesat_sim/mpc.py`: nonlinear shooting MPC (optax L-BFGS over an RK4 rollout).
- `cubesat_sim/mppi.py`: MPPI (sampled rollouts with annealed noise).
- `cubesat_sim/config.py`: loads and checks the file, builds the wheel matrix A.
- `cubesat_sim/dynamics.py`: equations of motion, wheel limits, RK4 step.
- `cubesat_sim/quaternion.py`: quaternion functions.
- `cubesat_sim/sim.py`: simulation loop (`jax.lax.scan`), `simulate()` and `run()`.
- `cubesat_sim/viz.py`: Plotly time series (with requirement limits), 3D animation, thruster plot.
- `cubesat_sim/thruster1d.py`: 1D thruster plant, disturbance profile, `run_thruster()`.
- `cubesat_sim/thruster_control.py`: thruster PID and the `THRUSTER_CONTROLLERS` registry.
- `config/thruster_1d.yaml`, `config/control/thruster_pid.yaml`: thruster plant and PID settings.
- `notebooks/explore.py`: marimo notebook to run scenarios and compare command laws.
- `notebooks/view_run.py`: marimo notebook to view and compare runs saved by `run_sim.py`.
- `notebooks/compare_controllers.py`: marimo notebook that runs PD, MPC and MPPI on one
  scenario and compares performance metrics, requirement checks, time histories and 3D motion.

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
- `test_horizon_controllers.py`: the stateful command interface; MPC and MPPI convergence,
  torque hold, soft rate limit, MPPI null-space drift and seeding, parameter checks.
- `test_thruster1d.py`: thruster closed form, limits, disturbance windows, anti-windup
  contrast, settling, config checks, registries and runner.
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

`run(cfg, command=f)` accepts a function `f(t, x)` (or a stateful command, see above) that returns the four wheel torque
commands T_w. For a body torque command T_C, the paper's allocation (eq. 21) gives T_w. The open-loop profile in the file is the default for this function.
