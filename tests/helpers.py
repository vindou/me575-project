"""Shared helpers and hypothesis strategies for the tests."""
import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "cubesat_6u.yaml"


def make(cfg, **sim_changes):
    """Copy of cfg with fields of cfg.sim replaced."""
    return dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, **sim_changes))


def with_inertia(cfg, J):
    return dataclasses.replace(cfg, body=dataclasses.replace(cfg.body, J=jnp.asarray(J)))


def body_energy(w, J):
    """0.5 w^T J w for each row of w."""
    return 0.5 * np.einsum("ki,ij,kj->k", np.asarray(w), np.asarray(J), np.asarray(w))


# --- hypothesis strategies -------------------------------------------------------------

def vectors(n, bound):
    # No subnormals: XLA on CPU flushes them to zero.
    return st.lists(st.floats(-bound, bound, allow_subnormal=False), min_size=n, max_size=n).map(np.array)


@st.composite
def quaternions(draw):
    v = draw(vectors(4, 1.0).filter(lambda v: np.linalg.norm(v) > 0.1))
    return v / np.linalg.norm(v)


@st.composite
def inertias(draw):
    """Random positive definite, physically valid (triangle inequality), non-diagonal J."""
    moments = draw(vectors(3, 1.0)) * 0.025 + 0.095       # in [0.07, 0.12]
    R = Rotation.from_quat(draw(quaternions())).as_matrix()
    return R @ np.diag(moments) @ R.T


@st.composite
def states(draw, h_bound=0.014):
    """State x = [q, w, h] with wheel momentum inside the limit."""
    return np.concatenate([draw(quaternions()), draw(vectors(3, 0.5)), draw(vectors(4, h_bound))])
