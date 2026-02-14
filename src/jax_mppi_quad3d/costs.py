from __future__ import annotations
import jax.numpy as jnp
from jax_mppi_quad3d.quat import quat_normalize, quat_to_R

def obs_cost_spheres(p: jnp.ndarray, spheres: jnp.ndarray, margin: float, w_obs: float) -> jnp.ndarray:
    if spheres.size == 0:
        return 0.0
    c = spheres[:, 0:3]
    r = spheres[:, 3]
    d = jnp.linalg.norm(p[None, :] - c, axis=-1)
    pen = (r + margin) - d
    pen = jnp.maximum(pen, 0.0)
    return w_obs * jnp.sum(pen ** 2)

def upright_cost_from_q(q: jnp.ndarray, w_upright: float) -> jnp.ndarray:
    """
    Penalize tilt: keep body z-axis aligned with world +z.
    b3 = R(q)[:,2] should be [0,0,1].
    """
    q = quat_normalize(q)
    R = quat_to_R(q)
    b3 = R[:, 2]
    e3 = jnp.array([0.0, 0.0, 1.0], dtype=q.dtype)
    return w_upright * jnp.sum((b3 - e3) ** 2)

def total_cost(
    X: jnp.ndarray,          # (H+1,13)
    U: jnp.ndarray,          # (H,4)   [T, tau_x, tau_y, tau_z]
    Pref: jnp.ndarray,       # (H+1,3)
    spheres: jnp.ndarray,    # (M,4)
    hover_thrust: float,     # m*g
    w_track: float,
    w_vel: float,
    w_w: float,
    w_upright: float,
    w_T: float,
    w_tau: float,
    w_obs: float,
    obs_margin: float,
) -> jnp.ndarray:
    P = X[:, 0:3]
    V = X[:, 3:6]
    Q = X[:, 6:10]
    W = X[:, 10:13]

    track = jnp.sum(jnp.sum((P - Pref) ** 2, axis=-1))
    vel_pen = jnp.sum(jnp.sum(V ** 2, axis=-1))
    ang_pen = jnp.sum(jnp.sum(W ** 2, axis=-1))

    # Upright penalty across horizon
    upright_pen = jnp.sum(jnp.array([upright_cost_from_q(Q[i], w_upright) for i in range(Q.shape[0])]))

    # Control penalty: thrust deviation from hover + torque magnitude
    T = U[:, 0]
    tau = U[:, 1:4]
    thrust_pen = jnp.sum((T - hover_thrust) ** 2)
    tau_pen = jnp.sum(jnp.sum(tau ** 2, axis=-1))

    obs_pen = jnp.sum(jnp.array([obs_cost_spheres(P[i], spheres, obs_margin, w_obs) for i in range(P.shape[0])]))

    return (
        w_track * track
        + w_vel * vel_pen
        + w_w * ang_pen
        + upright_pen
        + w_T * thrust_pen
        + w_tau * tau_pen
        + obs_pen
    )
