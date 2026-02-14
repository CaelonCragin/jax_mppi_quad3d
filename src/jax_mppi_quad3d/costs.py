from __future__ import annotations
import jax.numpy as jnp

def obs_cost_spheres(p: jnp.ndarray, spheres: jnp.ndarray, margin: float, w_obs: float) -> jnp.ndarray:
    """
    spheres: (M,4) [cx,cy,cz,r]
    penalty when inside (r + margin)
    """
    if spheres.size == 0:
        return 0.0
    c = spheres[:, 0:3]
    r = spheres[:, 3]
    d = jnp.linalg.norm(p[None, :] - c, axis=-1)
    pen = (r + margin) - d
    pen = jnp.maximum(pen, 0.0)
    return w_obs * jnp.sum(pen ** 2)

def total_cost(
    X: jnp.ndarray,   # (H+1,13)
    U: jnp.ndarray,   # (H,4)
    Pref: jnp.ndarray,# (H+1,3)
    spheres: jnp.ndarray, # (M,4)
    w_track: float,
    w_vel: float,
    w_w: float,
    w_u: float,
    w_obs: float,
    obs_margin: float,
) -> jnp.ndarray:
    P = X[:, 0:3]
    V = X[:, 3:6]
    W = X[:, 10:13]

    track = jnp.sum(jnp.sum((P - Pref) ** 2, axis=-1))
    vel_pen = jnp.sum(jnp.sum(V ** 2, axis=-1))
    ang_pen = jnp.sum(jnp.sum(W ** 2, axis=-1))
    u_pen = jnp.sum(jnp.sum(U ** 2, axis=-1))

    obs_pen = jnp.sum(jnp.array([obs_cost_spheres(P[i], spheres, obs_margin, w_obs) for i in range(P.shape[0])]))
    return w_track * track + w_vel * vel_pen + w_w * ang_pen + w_u * u_pen + obs_pen
