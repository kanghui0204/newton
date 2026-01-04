# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cube Free Fall Penetration Test
#
# Scene:
#   - Infinite ground plane at z = 0
#   - One cube with size 1x1x1, initially centered at (0, 0, 100)
#
# Purpose:
#   - Let the cube fall under gravity and rest on the ground.
#   - At each rendered frame, print the cube center-of-mass position.
#   - When at rest, if center z < 0.5, the cube has penetrated the ground.
#   - You can switch solver (XPBD / VBD) to compare penetration behavior.
#
# Usage (headless):
#   python -m newton.examples cube_drop --viewer null --solver xpbd
#   python -m newton.examples cube_drop --viewer null --solver vbd
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        # ---- simulation timing ----
        self.fps = 30                    # rendered frames per second
        self.frame_dt = 1.0 / self.fps    # time between rendered frames
        self.sim_time = 0.0
        self.sim_substeps = 2            # physics substeps per frame
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"

        # ---- build model: ground + one cube ----
        builder = newton.ModelBuilder(
            up_axis=newton.Axis.Z,
            gravity=-9.81,       
        )
        # Optional: tweak default material for VBD solver
        if self.solver_type == "vbd":
            builder.default_shape_cfg.ke = 1.0e6  # contact stiffness
            builder.default_shape_cfg.kd = 5.0e1  # contact damping
            builder.default_shape_cfg.mu = 0.5    # friction coefficient
        else:  # xpbd
            # Strongly damped, low-bounce contact for both ground and cube
            cfg = builder.default_shape_cfg
            cfg.ke = 1.0e4      # softer stiffness
            cfg.kd = 5.0e3      # very high damping
            cfg.mu = 0.8


        builder.add_ground_plane()

        # Cube: size 1x1x1 -> half extents (0.5, 0.5, 0.5)
        # Initial center position at (0, 0, 100)
        cube_start_z = 100.0
        self.cube_pos = wp.vec3(0.0, 0.0, cube_start_z)
        cube_body = builder.add_body(
            xform=wp.transform(p=self.cube_pos, q=wp.quat_identity()),
            key="cube",
        )

        builder.add_shape_box(cube_body, hx=0.5, hy=0.5, hz=0.5)

        # >>> ADD THIS FOR VBD <<<
        if self.solver_type == "vbd":
            # Compute body coloring (independent sets) for the VBD solver
            builder.color()

        # Finalize model (allocate buffers, precompute data, etc.)
        self.model = builder.finalize()

        # ---- choose solver ----
        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=30,
            )
        else:
            # XPBD is the default position-based solver
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=10,
            )

        # Allocate states and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # ---- collision pipeline ----
        # We do not expect many contact points, so 10 is enough per pair
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            rigid_contact_max_per_pair=10,
        )
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Register model with viewer (even if viewer=null, this is harmless)
        self.viewer.set_model(self.model)

        # Pre-capture CUDA graph if running on GPU
        self.capture()

    def capture(self):
        # If running on CUDA, capture the simulation into a CUDA graph
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        # Run physics substeps for one rendered frame
        for _ in range(self.sim_substeps):
            # Clear forces (so they are not accumulated across steps)
            self.state_0.clear_forces()

            # Apply user forces (e.g., from viewer UI)
            self.viewer.apply_forces(self.state_0)

            # Collision detection
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            # Integrate one physics substep
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states for next substep
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # One rendered frame step: either launch CUDA graph or run simulate()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # body_q is shape (num_bodies, 7): [x, y, z, qx, qy, qz, qw]
        body_q_np = self.state_0.body_q.numpy()    # (N, 7)
        cube_pos = body_q_np[0, 0:3]               # first body, first 3 columns: (x, y, z)
        
        print(
            f"solver type = {self.solver_type} time={self.sim_time:.4f}  "
            f"cube_center=({cube_pos[0]:.6f}, {cube_pos[1]:.6f}, {cube_pos[2]:.6f})"
        )

        # Interpretation:
        #   - When the cube is resting stably on the ground,
        #     its center z should ideally be 0.5 (half the cube height).
        #   - If z < 0.5, the cube has penetrated into the ground.
        #   - You can compare this value across different solvers (XPBD vs VBD).

    def test_final(self):
        # No automated test for this example; it is intended for manual inspection.
        pass

    def render(self):
        # If viewer is not null, this logs state and contacts for visualization
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Shared examples parser: adds common options like --viewer, --device, etc.
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        choices=["vbd", "xpbd"],
        help="Solver type: xpbd (default) or vbd",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)

