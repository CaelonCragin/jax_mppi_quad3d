from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from jax_mppi_quad3d.dynamics import Quad3DConfig, step_euler
from jax_mppi_quad3d.costs import total_cost

@dataclass
class MPPIConfig3D:
    H: int = 45
    N: int = 1024
    dt: float = 0.02

    sigma_T: float = 1.5
    sigma_tau: float = 0.12
    lam: float = 1.0
    u_smooth: float = 0.25

    # costs
    w_track: float = 10.0
    w_vel: float = 0.2
    w_w: float = 0.03
    w_upright: float = 3.0   # <<< NEW: keep upright
    w_T: float = 0.02
    w_tau: float = 0.02
    w_obs: float = 220.0
    obs_margin: float = 0.35

class MPPIQuad3D:
    def __init__(self, cfg: MPPIConfig3D, quad: Quad3DConfig, seed: int = 0):
        self.cfg = cfg
        self.quad = quad
        self.hover = float(quad.m * quad.g)

        self.key = jax.random.PRNGKey(seed)

        # Initialize nominal controls at hover
        self.U = jnp.zeros((cfg.H, 4), dtype=jnp.float32)
        self.U = self.U.at[:, 0].set(self.hover)

        self._update_jit = jax.jit(self._update_once)

    def reset(self):
        self.U = jnp.zeros_like(self.U)
        self.U = self.U.at[:, 0].set(self.hover)

    def shift(self):
        tail = jnp.array([[self.hover, 0.0, 0.0, 0.0]], dtype=self.U.dtype)
        self.U = jnp.vstack([self.U[1:], tail])

    def act(self, x0: jnp.ndarray, Pref: jnp.ndarray, spheres: jnp.ndarray):
        self.key, sub = jax.random.split(self.key)
        self.U, bestJ = self._update_jit(self.U, x0, Pref, spheres, sub)
        u0 = self.U[0]
        self.shift()
        return u0, bestJ

    def _rollout(self, x0: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        def body(x, u):
            x_next = step_euler(x, u, self.cfg.dt, self.quad)
            return x_next, x_next
        _, xs = jax.lax.scan(body, x0, U)
        return jnp.vstack([x0[None, :], xs])

    def _update_once(self, U_nom, x0, Pref, spheres, key):
        cfg = self.cfg

        key1, key2 = jax.random.split(key)
        eps_T = cfg.sigma_T * jax.random.normal(key1, (cfg.N, cfg.H, 1), dtype=U_nom.dtype)
        eps_tau = cfg.sigma_tau * jax.random.normal(key2, (cfg.N, cfg.H, 3), dtype=U_nom.dtype)
        eps = jnp.concatenate([eps_T, eps_tau], axis=-1)

        U_samp = U_nom[None, :, :] + eps

        X_samp = jax.vmap(lambda U: self._rollout(x0, U), in_axes=0)(U_samp)

        J = jax.vmap(
            lambda X, U: total_cost(
                X, U, Pref, spheres,
                self.hover,
                cfg.w_track, cfg.w_vel, cfg.w_w,
                cfg.w_upright,
                cfg.w_T, cfg.w_tau,
                cfg.w_obs, cfg.obs_margin
            ),
            in_axes=(0, 0)
        )(X_samp, U_samp)

        Jmin = jnp.min(J)
        w = jnp.exp(-(J - Jmin) / cfg.lam)
        w_sum = jnp.sum(w) + 1e-12

        U_new = jnp.sum(w[:, None, None] * U_samp, axis=0) / w_sum
        U_out = (1.0 - cfg.u_smooth) * U_nom + cfg.u_smooth * U_new
        return U_out, Jmin
