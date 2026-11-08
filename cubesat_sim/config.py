"""Load the YAML configuration file, check it, and calculate derived values.

The `model:` key selects the plant: `attitude` (the default when the key is missing,
load_config) or `thruster_1d` (load_thruster_config). load_plant() dispatches on it.

BodyParams, WheelParams and SimParams are JAX pytrees, so they can be passed through
jax.jit, jax.grad and jax.vmap. Float fields that set array shapes or loop lengths
(dt, t_end) are static metadata, not traced values.
"""
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BodyParams:
    mass: jax.Array
    J: jax.Array         # 3x3 inertia, body frame (paper: I_S)

    @property
    def J_inv(self):
        return jnp.linalg.inv(self.J)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WheelParams:
    A: jax.Array         # 3xN, column i = spin axis of wheel i in the body frame (paper: L)
    u_max: jax.Array
    h_max: jax.Array
    rotor_inertia: jax.Array

    @property
    def count(self):
        return self.A.shape[1]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SimParams:
    dt: float = field(metadata=dict(static=True))
    t_end: float = field(metadata=dict(static=True))
    q0: jax.Array
    w0: jax.Array
    h0: jax.Array

    @property
    def steps(self):
        return int(round(self.t_end / self.dt))

    @property
    def x0(self):
        return jnp.concatenate([self.q0, self.w0, self.h0])


@dataclass(frozen=True)
class CommandSegment:
    t_start: float
    t_end: float
    u: jax.Array


@dataclass(frozen=True)
class Config:
    body: BodyParams
    wheels: WheelParams
    sim: SimParams
    open_loop_command: tuple
    visualization: dict  # not used by the dynamics
    requirements: dict   # not used by the dynamics; limits drawn by the plots
    raw: dict            # the file contents, saved with each result


def wheel_axes(beta_deg, azimuth_deg):
    """Distribution matrix L of the paper (eq. 13): column i = spin axis of wheel i.

    beta is measured from the body xy plane; the paper's theta is the first azimuth.
    """
    b = jnp.radians(beta_deg)
    az = jnp.radians(jnp.asarray(azimuth_deg, dtype=float))
    return jnp.vstack([jnp.cos(b) * jnp.cos(az),
                       jnp.cos(b) * jnp.sin(az),
                       jnp.sin(b) * jnp.ones_like(az)])


def _positive(value, name):
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name} must be positive")
    return value


MODELS = ("attitude", "thruster_1d")


def plant_model(raw):
    """The plant model named by a config file's contents; `attitude` when not given."""
    model = raw.get("model", "attitude") if isinstance(raw, dict) else None
    if model not in MODELS:
        raise ValueError(f"model must be one of {', '.join(MODELS)}; got {model!r}")
    return model


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if plant_model(raw) != "attitude":
        raise ValueError(f"{path} is a {raw['model']} config; use load_plant() or "
                         "load_thruster_config()")

    J = np.array(raw["body"]["inertia_kg_m2"], dtype=float)
    if J.shape != (3, 3) or not np.allclose(J, J.T):
        raise ValueError("body.inertia_kg_m2 must be a symmetric 3x3 matrix")
    if np.any(np.linalg.eigvalsh(J) <= 0.0):
        raise ValueError("body.inertia_kg_m2 must be positive definite")
    body = BodyParams(jnp.asarray(_positive(raw["body"]["mass_kg"], "body.mass_kg")),
                      jnp.asarray(J))

    w = raw["wheels"]
    A = wheel_axes(float(w["beta_from_xy_deg"]), w["azimuth_deg"])
    if np.linalg.matrix_rank(np.asarray(A)) < 3:
        raise ValueError("the wheel axes do not give torque on all three body axes")
    wheels = WheelParams(A,
                         jnp.asarray(_positive(w["u_max_Nm"], "wheels.u_max_Nm")),
                         jnp.asarray(_positive(w["h_max_Nms"], "wheels.h_max_Nms")),
                         jnp.asarray(_positive(w["rotor_inertia_kg_m2"],
                                               "wheels.rotor_inertia_kg_m2")))
    n = wheels.count

    s = raw["sim"]
    q0 = np.array(s["q0"], dtype=float)
    if q0.shape != (4,) or abs(np.linalg.norm(q0) - 1.0) > 1e-3:
        raise ValueError("sim.q0 must have 4 elements and a length of 1")
    h0 = np.array(s["h0_Nms"], dtype=float)
    if h0.shape != (n,):
        raise ValueError(f"sim.h0_Nms must have {n} elements")
    if np.any(np.abs(h0) > float(wheels.h_max)):
        raise ValueError("sim.h0_Nms is larger than wheels.h_max_Nms")
    w0 = np.array(s["w0_rad_s"], dtype=float)
    if w0.shape != (3,):
        raise ValueError("sim.w0_rad_s must have 3 elements")
    dt = _positive(s["dt_s"], "sim.dt_s")
    t_end = _positive(s["t_end_s"], "sim.t_end_s")
    if abs(t_end / dt - round(t_end / dt)) > 1e-9 * (t_end / dt):
        raise ValueError("sim.t_end_s must be a whole multiple of sim.dt_s")
    sim = SimParams(dt, t_end, jnp.asarray(q0 / np.linalg.norm(q0)),
                    jnp.asarray(w0), jnp.asarray(h0))

    segments = []
    for seg in raw.get("open_loop_command") or []:
        u = np.array(seg["u_Nm"], dtype=float)
        if u.shape != (n,):
            raise ValueError(f"each open_loop_command.u_Nm must have {n} elements")
        if not float(seg["t_start_s"]) < float(seg["t_end_s"]):
            raise ValueError("each open_loop_command segment needs t_start_s < t_end_s")
        segments.append(CommandSegment(float(seg["t_start_s"]), float(seg["t_end_s"]),
                                       jnp.asarray(u)))

    return Config(body, wheels, sim, tuple(segments),
                  raw.get("visualization", {}), raw.get("requirements") or {}, raw)


# --- 1D thruster -------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ThrusterSimParams:
    dt: float = field(metadata=dict(static=True))
    t_end: float = field(metadata=dict(static=True))
    position0: jax.Array
    velocity0: jax.Array

    @property
    def steps(self):
        return int(round(self.t_end / self.dt))

    @property
    def x0(self):
        return jnp.stack([self.position0, self.velocity0])


@dataclass(frozen=True)
class ThrusterConfig:
    params: "ThrusterParams"
    sim: ThrusterSimParams
    disturbance: "DisturbanceProfile"   # hashable external_force(t, x)
    raw: dict                           # the file contents, saved with each result
    model: str = "thruster_1d"


def _finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_positive(value, name):
    value = _finite(value, name)
    if not value > 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def load_thruster_config(path):
    from .thruster1d import DisturbanceProfile, ThrusterParams

    raw = yaml.safe_load(Path(path).read_text())
    if plant_model(raw) != "thruster_1d":
        raise ValueError(f"{path} is not a thruster_1d config (model: thruster_1d)")
    for key in ("mass_kg", "F_max_N", "x0_m", "v0_m_s", "dt_s", "t_end_s"):
        if key not in raw:
            raise ValueError(f"{key} is missing")
    params = ThrusterParams(jnp.asarray(_finite_positive(raw["mass_kg"], "mass_kg")),
                            jnp.asarray(_finite_positive(raw["F_max_N"], "F_max_N")))
    dt = _finite_positive(raw["dt_s"], "dt_s")
    t_end = _finite_positive(raw["t_end_s"], "t_end_s")
    if abs(t_end / dt - round(t_end / dt)) > 1e-9 * (t_end / dt):
        raise ValueError("t_end_s must be a whole multiple of dt_s")
    sim = ThrusterSimParams(dt, t_end, jnp.asarray(_finite(raw["x0_m"], "x0_m")),
                            jnp.asarray(_finite(raw["v0_m_s"], "v0_m_s")))

    d = raw.get("disturbance") or {}
    if not isinstance(d, dict):
        raise ValueError("disturbance must be a mapping")
    F_const = _finite(d.get("F_d_N", 0.0), "disturbance.F_d_N")
    segments = []
    for i, seg in enumerate(d.get("segments") or []):
        name = f"disturbance.segments.{i}"
        try:
            a, b, F = seg["t_start_s"], seg["t_end_s"], seg["F_N"]
        except (KeyError, TypeError):
            raise ValueError(f"{name} needs t_start_s, t_end_s and F_N") from None
        a, b = _finite(a, f"{name}.t_start_s"), _finite(b, f"{name}.t_end_s")
        if not a < b:
            raise ValueError(f"{name} needs t_start_s < t_end_s")
        segments.append((a, b, _finite(F, f"{name}.F_N")))
    return ThrusterConfig(params, sim, DisturbanceProfile(F_const, segments), raw)


def load_plant(path):
    """Load any plant config: returns load_config(path) or load_thruster_config(path)."""
    raw = yaml.safe_load(Path(path).read_text())
    if plant_model(raw) == "thruster_1d":
        return load_thruster_config(path)
    return load_config(path)
