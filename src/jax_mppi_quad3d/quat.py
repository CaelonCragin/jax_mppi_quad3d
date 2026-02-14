from __future__ import annotations
import jax.numpy as jnp

def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    return q / (jnp.linalg.norm(q) + 1e-12)

def quat_mul(q: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    # q,r: [w,x,y,z]
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return jnp.array([
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
    ], dtype=q.dtype)

def quat_to_R(q: jnp.ndarray) -> jnp.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return jnp.array([
        [ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz],
    ], dtype=q.dtype)
