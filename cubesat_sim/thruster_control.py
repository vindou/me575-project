"""Controllers for the 1D thruster plant (thruster1d.py).

Each factory has the signature factory(cfg, x_ref, **parameters) and returns a command
that outputs the requested force u (N). The plant applies D = sat(u / F_max, 0, 1).
"""
import math

import jax.numpy as jnp


def pid_gains(mass, Kp=None, Ki=None, Kd=None, *, pole_rad_s=None):
    """Resolve manual gains or a triple-pole linear design.

    Automatic mode places all three closed-loop poles of the unsaturated loop
    m s^3 + Kd s^2 + Kp s + Ki at s = -p: Kd = 3 m p, Kp = 3 m p^2, Ki = m p^3.
    The design ignores the [0, F_max] limit and the missing braking, so it does not
    guarantee the overshoot or settling time it would give for a linear plant.
    """
    manual = (Kp, Ki, Kd)
    if any(g is not None for g in manual):
        if any(g is None for g in manual):
            raise ValueError("Provide all of Kp, Ki and Kd for manual tuning.")
        if pole_rad_s is not None:
            raise ValueError("Choose manual Kp/Ki/Kd or automatic pole_rad_s, not both.")
        gains = tuple(float(g) for g in manual)
        if not all(math.isfinite(g) and g >= 0.0 for g in gains):
            raise ValueError("Kp, Ki and Kd must be finite and nonnegative.")
        return gains
    p = 0.05 if pole_rad_s is None else float(pole_rad_s)
    if not (math.isfinite(p) and p > 0.0):
        raise ValueError("pole_rad_s must be finite and positive.")
    m = float(mass)
    return 3.0 * m * p**2, m * p**3, 3.0 * m * p


def pid_controller(cfg, x_ref, Kp=None, Ki=None, Kd=None, *, pole_rad_s=None,
                   anti_windup=True):
    """Build a stateful PID position controller for a fixed setpoint x_ref (m).

    u = Kp e + Ki I + Kd e_dot, with e = x_ref - x and e_dot = -v (x_ref is constant,
    so no numerical differentiation). The carried state is I, the integral of e, updated
    with the simulator step dt (forward Euler).

    Anti-windup (conditional integration): I is not updated while the duty cycle is
    saturated in the direction the error pushes it, i.e. u >= F_max with e > 0 (D = 1)
    or u <= 0 with e < 0 (D = 0). The second case matters most here: after an
    overshoot the thruster cannot push back, and a winding integral would hold D = 0
    long after the mass has come back below x_ref.
    """
    x_ref = float(x_ref)
    if not math.isfinite(x_ref):
        raise ValueError("x_ref must be finite.")
    mode = "manual" if any(g is not None for g in (Kp, Ki, Kd)) else "automatic"
    Kp, Ki, Kd = pid_gains(cfg.params.mass, Kp, Ki, Kd, pole_rad_s=pole_rad_s)
    if not isinstance(anti_windup, bool):
        raise ValueError("anti_windup must be true or false.")
    dt = float(cfg.sim.dt)
    F_max = cfg.params.F_max

    def command(t, x, integral):
        e = x_ref - x[0]
        e_dot = -x[1]
        u = Kp * e + Ki * integral + Kd * e_dot
        if anti_windup:
            saturated = ((u >= F_max) & (e > 0.0)) | ((u <= 0.0) & (e < 0.0))
            integral = jnp.where(saturated, integral, integral + e * dt)
        else:
            integral = integral + e * dt
        return u, integral

    command.init_state = jnp.asarray(0.0)
    command.design_info = {"mode": mode, "Kp": Kp, "Ki": Ki, "Kd": Kd,
                           "pole_rad_s": None if mode == "manual" else
                           (0.05 if pole_rad_s is None else float(pole_rad_s)),
                           "anti_windup": anti_windup, "x_ref_m": x_ref}
    return command


# Add each thruster controller factory here: factory(cfg, x_ref, **parameters).
THRUSTER_CONTROLLERS = {
    "pid": pid_controller,
}
