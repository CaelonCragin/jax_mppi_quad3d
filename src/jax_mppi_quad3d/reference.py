from __future__ import annotations
import jax.numpy as jnp

def helix_ref(t: jnp.ndarray) -> jnp.ndarray:
    # smooth 3D path: circle in XY + gentle z oscillation
    R = 2.5
    w = 0.35
    x = R * jnp.cos(w * t)
    y = R * jnp.sin(w * t)
    z = 1.2 + 0.6 * jnp.sin(0.5 * w * t)
    return jnp.array([x, y, z], dtype=jnp.float32)

def make_ref_horizon(t0: float, dt: float, H: int) -> jnp.ndarray:
    ts = t0 + dt * jnp.arange(H + 1)
    Pref = jnp.stack([helix_ref(ts[i]) for i in range(H + 1)], axis=0)
    return Pref
