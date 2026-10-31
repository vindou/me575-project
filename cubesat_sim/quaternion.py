"""Quaternion functions.

Convention (same as scipy): q = [x, y, z, w], scalar last, unit length.
This is the paper's Q = [q1 q2 q3 q4] with q1:3 = e sin(a/2), q4 = cos(a/2) (eq. 2).
q gives the body orientation: v_inertial = R(q) @ v_body.
"""
import jax.numpy as jnp


def normalize(q):
    return q / jnp.linalg.norm(q)


def skew(v):
    """Cross-product matrix: skew(v) @ u == cross(v, u)."""
    return jnp.array([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]])


def rotation_matrix(q):
    """Body-to-inertial rotation matrix."""
    v, w = q[:3], q[3]
    return (w * w - v @ v) * jnp.eye(3) + 2.0 * jnp.outer(v, v) + 2.0 * w * skew(v)


def kinematics(q, w_body):
    """Quaternion rate for a body angular velocity given in the body frame.

    Same as Q_dot = 0.5 [Omega'] Q, eqs. (3)-(4) of the paper.
    """
    v, s = q[:3], q[3]
    v_dot = 0.5 * (s * w_body + jnp.cross(v, w_body))
    s_dot = -0.5 * (v @ w_body)
    return jnp.append(v_dot, s_dot)


def angle_between(q_a, q_b):
    """Smallest rotation angle (rad) between two orientations.

    Uses atan2 of the relative rotation instead of arccos of the dot product, so the
    result is accurate near zero angle and its gradient stays finite there.
    """
    va, sa = q_a[:3], q_a[3]
    vb, sb = q_b[:3], q_b[3]
    # Relative rotation conj(q_a) * q_b.
    s = sa * sb + va @ vb
    v = sa * vb - sb * va - jnp.cross(va, vb)
    # Safe norm: the gradient of sqrt is infinite at zero.
    v2 = v @ v
    v_norm = jnp.where(v2 > 0.0, jnp.sqrt(jnp.where(v2 > 0.0, v2, 1.0)), 0.0)
    return 2.0 * jnp.arctan2(v_norm, jnp.abs(s))

def conjugate(q):
    """Return [-v, s] for a scalar-last quaternion q = [v, s]."""
    v, s = q[:3], q[3]
    return jnp.append(-v, s)

def quarternion_product(qa, qb):
    """Return the Hamilton product qa ⊗ qb for scalar-last quaternions."""
    va, sa = qa[:3], qa[3]
    vb, sb = qb[:3], qb[3]
    
    v = sa*vb + sb*va + jnp.cross(va, vb)
    s = sa*sb - va@vb
    return jnp.append(v, s)
