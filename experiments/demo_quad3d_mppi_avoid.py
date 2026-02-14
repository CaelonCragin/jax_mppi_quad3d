from __future__ import annotations
import os, shutil
import numpy as np
import jax.numpy as jnp

from jax_mppi_quad3d.dynamics import Quad3DConfig, step_euler
from jax_mppi_quad3d.reference import make_ref_horizon
from jax_mppi_quad3d.mppi import MPPIConfig3D, MPPIQuad3D
from jax_mppi_quad3d.viz import ensure_dir, save_frame_xy, make_gif, plot_xyz, plot_inputs, plot_cost

def main():
    outdir = "assets"
    ensure_dir(outdir)

    frames_dir = os.path.join(outdir, "frames_quad3d")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    quad = Quad3DConfig()
    mppi_cfg = MPPIConfig3D(H=35, N=1024, dt=0.02)
    ctrl = MPPIQuad3D(mppi_cfg, quad, seed=0)

    # Spherical obstacles (cx,cy,cz,r)
    spheres = np.array([
        [ 1.2,  0.2, 1.2, 0.6],
        [-1.4, -0.8, 1.0, 0.7],
        [ 0.0,  1.6, 1.3, 0.6],
    ], dtype=np.float32)
    spheres_j = jnp.array(spheres)

    # Initial state: p,v,q,w
    p0 = jnp.array([2.8, 0.0, 0.4], dtype=jnp.float32)
    v0 = jnp.zeros(3, dtype=jnp.float32)
    q0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    w0 = jnp.zeros(3, dtype=jnp.float32)
    x = jnp.concatenate([p0, v0, q0, w0], axis=0)

    # Give nominal hover thrust as a starting bias (optional but helps)
    ctrl.U = ctrl.U.at[:,0].set(quad.m * quad.g)

    Tfinal = 10.0
    dt = mppi_cfg.dt
    steps = int(Tfinal / dt)

    P = np.zeros((steps, 3), dtype=np.float32)
    Pref = np.zeros((steps, 3), dtype=np.float32)
    Uhist = np.zeros((steps, 4), dtype=np.float32)
    Jhist = np.zeros((steps,), dtype=np.float32)
    thist = np.zeros((steps,), dtype=np.float32)

    save_every = 5
    frame_idx = 0

    for k in range(steps):
        t = k * dt
        thist[k] = t

        Pref_h = make_ref_horizon(t, dt, mppi_cfg.H)
        u, bestJ = ctrl.act(x, Pref_h, spheres_j)

        # step dynamics
        x = step_euler(x, u, dt, quad)

        P[k] = np.array(x[0:3])
        Pref[k] = np.array(Pref_h[0])
        Uhist[k] = np.array(u)
        Jhist[k] = float(bestJ)

        if (k % save_every) == 0:
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            save_frame_xy(
                P[:k+1], Pref[:k+1], spheres, P[k],
                frame_path, title=f"Quad3D MPPI Avoid | t={t:.2f}s"
            )
            frame_idx += 1

    plot_xyz(thist, P, os.path.join(outdir, "quad3d_xyz.png"))
    plot_inputs(thist, Uhist, os.path.join(outdir, "quad3d_inputs.png"))
    plot_cost(thist, Jhist, os.path.join(outdir, "quad3d_cost.png"))

    gif_path = os.path.join(outdir, "demo_quad3d_xy.gif")
    make_gif(frames_dir, gif_path, fps=25)

    print("Saved:")
    print(" - assets/demo_quad3d_xy.gif")
    print(" - assets/quad3d_xyz.png")
    print(" - assets/quad3d_inputs.png")
    print(" - assets/quad3d_cost.png")

if __name__ == "__main__":
    main()
