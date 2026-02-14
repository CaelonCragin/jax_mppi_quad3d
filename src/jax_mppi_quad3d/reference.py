from __future__ import annotations
import jax.numpy as jnp

def fig8_ref_3d(t: jnp.ndarray) -> jnp.ndarray:
    """
    Figure-8 in XY with mild Z oscillation.
      x = a sin(w t)
      y = b sin(w t) cos(w t)
      z = z0 + z_amp sin(0.5 w t)
    """
    a = 2.5
    b = 1.8
    w = 0.45

    x = a * jnp.sin(w * t)
    y = b * jnp.sin(w * t) * jnp.cos(w * t)

    z0 = 1.2
    z_amp = 0.25
    z = z0 + z_amp * jnp.sin(0.5 * w * t)

    return jnp.array([x, y, z], dtype=jnp.float32)

def make_ref_horizon(t0: float, dt: float, H: int) -> jnp.ndarray:
    ts = t0 + dt * jnp.arange(H + 1)
    Pref = jnp.stack([fig8_ref_3d(ts[i]) for i in range(H + 1)], axis=0)
    return Pref
