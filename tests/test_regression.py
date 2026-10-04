"""Compare three reference runs with saved results, to catch unintended changes.

After an intended change to the model, regenerate the saved file and review the diff:
    REGENERATE_GOLDEN=1 python -m pytest tests/test_regression.py
"""
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from cubesat_sim import run
from helpers import make

GOLDEN = Path(__file__).parent / "data" / "golden_runs.npz"
EVERY = 50   # keep every 50th sample


def tumble_command(t, x):
    return 1e-4 * jnp.array([jnp.sin(0.1 * t), 1.0, -0.5, jnp.cos(0.2 * t)])


def reverse_command(t, x):
    return jnp.where(t < 10.0, 1.0, -1.0) * jnp.array([1.0, 0.0, -0.3, 0.0])


def scenarios(cfg):
    yield "default", run(cfg)
    yield "tumble", run(make(cfg, t_end=120.0, w0=jnp.array([0.02, -0.03, 0.015]),
                             h0=jnp.array([0.004, -0.002, 0.001, 0.003])), command=tumble_command)
    yield "saturate", run(make(cfg, t_end=20.0), command=reverse_command)


def current(cfg):
    return {f"{name}/{k}": np.asarray(v)[::EVERY]
            for name, res in scenarios(cfg) for k, v in res.items()}


def test_matches_saved_results(cfg):
    now = current(cfg)
    if os.environ.get("REGENERATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        np.savez_compressed(GOLDEN, **now)
        pytest.skip(f"wrote {GOLDEN}")
    saved = np.load(GOLDEN)
    assert sorted(saved.files) == sorted(now)
    for k in saved.files:
        scale = max(np.abs(saved[k]).max(), 1e-300)
        assert np.allclose(now[k], saved[k], rtol=1e-12, atol=1e-12 * scale), k
