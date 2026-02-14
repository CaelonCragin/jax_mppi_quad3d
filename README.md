# jax-mppi-quad3d

**3D quadrotor + MPPI MPC + obstacle avoidance** implemented in **JAX** with batched rollouts + JIT.

## Run demo

```bash
pip install -r requirements.txt
PYTHONPATH=src python experiments/demo_quad3d_mppi_avoid.py
Outputs:

assets/demo_quad3d_xy.gif

assets/quad3d_xyz.png

assets/quad3d_inputs.png

assets/quad3d_cost.png

What it does
Simulates a rigid-body quadrotor with quaternion attitude

Uses MPPI (sampling-based MPC) to track a 3D trajectory

Avoids 3D spherical obstacles via soft penalties in the cost function

Runs batched rollouts in JAX (vectorized + JIT compiled)
