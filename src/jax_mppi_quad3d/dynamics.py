from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
from jax_mppi_quad3d.quat import quat_normalize, quat_mul, quat_to_R

@dataclass
class Quad3DConfig:
    m: float = 1.2
    g: float = 9.81
    Ixx: float = 0.02
    Iyy: float = 0.02
    Izz: float = 0.04
    thrust_min: float = 0.0
    thrust_max: float = 25.0
    tau_max: float = 1.0

def clamp_u(u: jnp.ndarray, cfg: Quad3DConfig) -> jnp.ndarray:
    T = jnp.clip(u[0], cfg.thrust_min, cfg.thrust_max)
    tau = jnp.clip(u[1:4], -cfg.tau_max, cfg.tau_max)
    return jnp.concatenate([jnp.array([T], dtype=u.dtype), tau], axis=0)

def rhs(x: jnp.ndarray, u: jnp.ndarray, cfg: Quad3DConfig) -> jnp.ndarray:
    """
    x = [p(3), v(3), q(4), w(3)]
    u = [T, tau(3)]
    """
    u = clamp_u(u, cfg)
    p = x[0:3]
    v = x[3:6]
    q = quat_normalize(x[6:10])
    w = x[10:13]

    R = quat_to_R(q)

    T = u[0]
    tau = u[1:4]

    # thrust in world: R * [0,0,T]
    thrust_w = R @ jnp.array([0.0, 0.0, T], dtype=x.dtype)

    p_dot = v
    v_dot = thrust_w / cfg.m + jnp.array([0.0, 0.0, -cfg.g], dtype=x.dtype)

    # q_dot = 0.5 q ⊗ [0,w]
    w_quat = jnp.concatenate([jnp.array([0.0], dtype=x.dtype), w], axis=0)
    q_dot = 0.5 * quat_mul(q, w_quat)

    # w_dot = I^{-1} (tau - w × (I w))
    I = jnp.diag(jnp.array([cfg.Ixx, cfg.Iyy, cfg.Izz], dtype=x.dtype))
    Iinv = jnp.diag(jnp.array([1.0/cfg.Ixx, 1.0/cfg.Iyy, 1.0/cfg.Izz], dtype=x.dtype))
    Iw = I @ w
    w_cross_Iw = jnp.cross(w, Iw)
    w_dot = Iinv @ (tau - w_cross_Iw)

    return jnp.concatenate([p_dot, v_dot, q_dot, w_dot], axis=0)

def step_euler(x: jnp.ndarray, u: jnp.ndarray, dt: float, cfg: Quad3DConfig) -> jnp.ndarray:
    x_next = x + dt * rhs(x, u, cfg)
    # renormalize quaternion
    qn = quat_normalize(x_next[6:10])
    return jnp.concatenate([x_next[0:6], qn, x_next[10:13]], axis=0)
