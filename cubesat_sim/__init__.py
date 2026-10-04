"""Forward dynamics model of a rigid CubeSat with reaction wheels (JAX)."""
import jax

# Double precision: the conservation checks need it, and attitude errors of a few
# arcseconds are below float32 resolution of a quaternion.
jax.config.update("jax_enable_x64", True)

from .config import load_config  # noqa: E402
from .sim import run, simulate  # noqa: E402

__all__ = ["load_config", "run", "simulate"]
