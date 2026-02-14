from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_gif(frames_dir: str, gif_path: str, fps: int = 25) -> None:
    import imageio.v2 as imageio
    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    images = [imageio.imread(os.path.join(frames_dir, f)) for f in files]
    duration_ms = int(1000 / fps)
    imageio.mimsave(gif_path, images, duration=duration_ms)

def save_frame_xy(P: np.ndarray, Pref: np.ndarray, spheres: np.ndarray, p_now: np.ndarray, outpath: str, title: str):
    plt.figure()
    plt.plot(Pref[:,0], Pref[:,1], "--", linewidth=1, label="ref (xy)")
    plt.plot(P[:,0], P[:,1], linewidth=1, label="quad (xy)")
    plt.scatter([p_now[0]], [p_now[1]], s=60)

    ax = plt.gca()
    for s in spheres:
        cx, cy, cz, r = s
        circ = plt.Circle((cx, cy), r, fill=False)
        ax.add_patch(circ)

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def plot_xyz(t: np.ndarray, P: np.ndarray, outpath: str):
    plt.figure()
    plt.plot(t, P[:,0], label="x")
    plt.plot(t, P[:,1], label="y")
    plt.plot(t, P[:,2], label="z")
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Position vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_inputs(t: np.ndarray, U: np.ndarray, outpath: str):
    plt.figure()
    plt.plot(t, U[:,0], label="T")
    plt.plot(t, U[:,1], label="tau_x")
    plt.plot(t, U[:,2], label="tau_y")
    plt.plot(t, U[:,3], label="tau_z")
    plt.xlabel("time [s]")
    plt.ylabel("control")
    plt.title("Controls vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_cost(t: np.ndarray, J: np.ndarray, outpath: str):
    plt.figure()
    plt.plot(t, J)
    plt.xlabel("time [s]")
    plt.ylabel("best cost")
    plt.title("MPPI best cost vs time")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_projections(P: np.ndarray, Pref: np.ndarray, out_xy: str, out_xz: str, out_yz: str):
    # XY
    plt.figure()
    plt.plot(Pref[:,0], Pref[:,1], "--", label="ref")
    plt.plot(P[:,0], P[:,1], label="quad")
    plt.axis("equal")
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title("Trajectory projection: XY")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_xy, dpi=150)
    plt.close()

    # XZ
    plt.figure()
    plt.plot(Pref[:,0], Pref[:,2], "--", label="ref")
    plt.plot(P[:,0], P[:,2], label="quad")
    plt.axis("equal")
    plt.xlabel("x [m]"); plt.ylabel("z [m]")
    plt.title("Trajectory projection: XZ")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_xz, dpi=150)
    plt.close()

    # YZ
    plt.figure()
    plt.plot(Pref[:,1], Pref[:,2], "--", label="ref")
    plt.plot(P[:,1], P[:,2], label="quad")
    plt.axis("equal")
    plt.xlabel("y [m]"); plt.ylabel("z [m]")
    plt.title("Trajectory projection: YZ")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_yz, dpi=150)
    plt.close()

def plot_trajectory_3d(P: np.ndarray, Pref: np.ndarray, outpath: str):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Pref[:,0], Pref[:,1], Pref[:,2], linestyle="--", label="ref")
    ax.plot(P[:,0], P[:,1], P[:,2], label="quad")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("3D trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
