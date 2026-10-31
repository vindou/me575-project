import jax.numpy as jnp
from . import quaternion as quat


def attitude_error(q_current, q_target):
    """Return conjugate(q_current) ⊗ q_target for unit quaternions."""
    q_current_conjugate = quat.conjugate(q_current)
    return quat.quarternion_product(q_current_conjugate, q_target)


def pd_gains(J, Kp=None, Kd=None, *, wn_rad_s=None, zeta=None):
    """Resolve manual gains or inertia-scaled small-angle PD gains.

    Automatic mode uses scalar wn and zeta: Kp = wn**2 * J,
    Kd = 2*zeta*wn * J. This is a local, unsaturated design; it does not
    cancel gyroscopic coupling or enforce a body-rate limit.
    """
    if Kp is not None or Kd is not None:
        if Kp is None or Kd is None:
            raise ValueError("Provide both Kp and Kd for manual tuning.")
        if wn_rad_s is not None or zeta is not None:
            raise ValueError("Choose manual Kp/Kd or automatic wn_rad_s/zeta, not both.")
        Kp, Kd = jnp.asarray(Kp), jnp.asarray(Kd)
    else:
        wn = 0.2 if wn_rad_s is None else float(wn_rad_s)
        damping = 1.0 if zeta is None else float(zeta)
        if not bool(jnp.isfinite(wn)) or wn <= 0:
            raise ValueError("wn_rad_s must be finite and positive.")
        if not bool(jnp.isfinite(damping)) or damping <= 0:
            raise ValueError("zeta must be finite and positive.")
        Kp = wn**2 * jnp.asarray(J)
        Kd = 2.0 * damping * wn * jnp.asarray(J)
    if Kp.shape != (3, 3) or Kd.shape != (3, 3):
        raise ValueError("Kp and Kd must both be 3x3 matrices.")
    if not bool(jnp.all(jnp.isfinite(Kp))) or not bool(jnp.all(jnp.isfinite(Kd))):
        raise ValueError("Kp and Kd must contain only finite values.")
    return Kp, Kd


def pd_controller(cfg, q_target, Kp=None, Kd=None, *, wn_rad_s=None, zeta=None):
    """Build a fixed-target PD controller using automatic or manual gains.

    The returned command(t, x) maps the current state to four wheel torques.
    State layout: x = [q (4), omega (3), h (4)], with body rates in rad/s.
    """
    # Prepare the fixed target and gains once before the simulation.
    q_target = jnp.asarray(q_target)
    q_target = q_target / jnp.linalg.norm(q_target)

    gain_mode = "automatic" if Kp is None and Kd is None else "manual"
    Kp, Kd = pd_gains(cfg.body.J, Kp, Kd, wn_rad_s=wn_rad_s, zeta=zeta)

    A = cfg.wheels.A
    # For this full-row-rank geometry, pinv(A) = A.T @ inv(A @ A.T).
    allocation = jnp.linalg.pinv(A)

    # The simulator calls this function at the start of each time step.
    def command(t, x):
        # Read the current attitude and body angular velocity from the state.
        q = x[0:4]
        omega = x[4:7]

        # Compute the rotation from the current attitude to the target.
        q_error = attitude_error(q, q_target)
        # Select the equivalent quaternion with a nonnegative scalar part.
        q_error = jnp.where(q_error[3] < 0.0, -q_error, q_error)
        # Near the target, this approximates the rotation-error vector in radians.
        e_R = 2.0 * q_error[:3]

        # Positive proportional feedback follows the current-to-target error.
        # Rate damping assumes a fixed target with zero desired angular velocity.
        tau_cmd = Kp @ e_R - Kd @ omega

        # Allocate body torque to four wheels; the plant applies actuator limits.
        u_cmd = allocation @ tau_cmd

        return u_cmd

    # Save the actual matrices so runs remain reproducible after plant changes.
    command.design_info = {"mode": gain_mode, "Kp": Kp.tolist(), "Kd": Kd.tolist()}
    return command


# Add each controller factory here: factory(cfg, q_target, **parameters).
# Every factory must return command(t, x) with four wheel torque commands.
CONTROLLERS = {
    "pd": pd_controller,
}
