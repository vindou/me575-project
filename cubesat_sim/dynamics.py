"""Equations of motion.

State x = [q (4), w (3), h (N)]
  q : orientation quaternion, body-to-inertial, [x, y, z, w]
  w : body angular velocity in the body frame, rad/s
  h : momentum of each wheel about its spin axis, N·m·s

Conventions and equation numbers follow Navabi & Hosseini, "Spacecraft quaternion based
attitude input-output feedback linearization control using reaction wheels":
J is I_S, A is the distribution matrix L, h is I_W w_W, and u is T_w (eq. 12), the torque
of each wheel on the body after the limits.

Total angular momentum in the body frame:  H = J w + A h          (6)-(8)
Equations:
  q_dot = kinematics(q, w)                                       (3)
  J w_dot = -w x (J w + A h) + A u + tau_ext                     (11), T_C = A u
  h_dot = -u                                                     (10), T_C = -dh_W/dt

All functions are pure and work under jax.jit, jax.grad and jax.vmap.
"""
import jax.numpy as jnp

from . import quaternion as quat


def split(x):
    return x[0:4], x[4:7], x[7:]


def limit_wheel_torque(u_cmd, h, wheels):
    """Apply the torque limit and the momentum limit of each wheel."""
    u = jnp.clip(u_cmd, -wheels.u_max, wheels.u_max)
    # A saturated wheel cannot accept more momentum in the same direction (h_dot = -u).
    blocked = (jnp.abs(h) >= wheels.h_max) & (u * h < 0.0)
    return jnp.where(blocked, 0.0, u)


def state_derivative(x, u_cmd, tau_ext, body, wheels):
    q, w, h = split(x)
    J, A = body.J, wheels.A
    u = limit_wheel_torque(u_cmd, h, wheels)
    q_dot = quat.kinematics(q, w)
    w_dot = jnp.linalg.solve(J, -jnp.cross(w, J @ w + A @ h) + A @ u + tau_ext)
    return jnp.concatenate([q_dot, w_dot, -u])


def rk4_step_with_torque(x, u_cmd, tau_ext, dt, body, wheels):
    """One fixed step. The command and the external torque stay constant in the step.

    Returns (x_next, u_applied). u_applied is the RK4-weighted mean of the limited wheel
    torque over the step, so h_next = h - dt * u_applied. It differs from the limited
    command only in a step where a wheel reaches its momentum limit.
    """
    k1 = state_derivative(x, u_cmd, tau_ext, body, wheels)
    k2 = state_derivative(x + 0.5 * dt * k1, u_cmd, tau_ext, body, wheels)
    k3 = state_derivative(x + 0.5 * dt * k2, u_cmd, tau_ext, body, wheels)
    k4 = state_derivative(x + dt * k3, u_cmd, tau_ext, body, wheels)
    slope = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    x_next = x + dt * slope
    return x_next.at[0:4].set(quat.normalize(x_next[0:4])), -slope[7:]


def rk4_step(x, u_cmd, tau_ext, dt, body, wheels):
    """One fixed step. The command and the external torque stay constant in the step."""
    return rk4_step_with_torque(x, u_cmd, tau_ext, dt, body, wheels)[0]


def momentum_inertial(x, body, wheels):
    """Total angular momentum in the inertial frame. Constant when tau_ext = 0."""
    q, w, h = split(x)
    return quat.rotation_matrix(q) @ (body.J @ w + wheels.A @ h)
