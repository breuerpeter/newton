import math

import warp as wp

import newton
import newton.examples


class Drone:
    def __init__(self, viewer, platform: str = "astro", args=None):
        self.platform = platform
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Drone body (rigid, symmetric in xz and xy plane)
        self.carbon_fiber_density = 1750  # [kg/m^3]

        # Astro (default)
        self.body_density = 800  # [kg/m^3]
        # Half lengths
        self.body_hx_m = 0.125
        self.body_hy_m = self.body_hx_m
        self.body_hz_m = 0.05

        self.body_boom_diam_m = 0.05
        self.body_boom_len_m = 0.20

        # Alta X Gen2
        if self.platform == "altaxgen2":
            # double all values for now
            for name in [n for n in dir(self) if n.startswith("body_")]:
                setattr(self, name, getattr(self, name) * 2)

        builder = newton.ModelBuilder()

        init_pos_body = wp.vec3(0.0, 0.0, 3.0)
        init_att_body = wp.quat_identity()
        init_tf_body = wp.transform(init_pos_body, init_att_body)

        body = builder.add_body(xform=init_tf_body)
        builder.add_shape_box(
            body,
            hx=self.body_hx_m,
            hy=self.body_hy_m,
            hz=self.body_hz_m,
            cfg=newton.ModelBuilder.ShapeConfig(density=self.body_density),
            key="fuselage",
        )

        boom_rot_x = wp.quat_from_axis_angle(
            wp.vec3(0, 1, 0), wp.half_pi
        )  # rotate cylinder to point along x
        boom_half_length = self.body_boom_len_m / 2
        boom_radius = self.body_boom_diam_m / 2
        diagonal = math.sqrt(2) * self.body_hx_m + boom_half_length

        for id in range(4):
            boom_angle = (2 * id + 1) * wp.pi / 4
            boom_rot_z = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), boom_angle)
            boom_rot = wp.mul(boom_rot_z, boom_rot_x)

            boom_shift = wp.vec3(
                diagonal * math.cos(boom_angle),
                diagonal * math.sin(boom_angle),
                0.0,
            )

            builder.add_shape_cylinder(
                body,
                xform=wp.transform(
                    boom_shift,
                    boom_rot,
                ),
                radius=boom_radius,
                half_height=boom_half_length,
                cfg=newton.ModelBuilder.ShapeConfig(density=self.carbon_fiber_density),
            )

        # Propellers (no mass, for visualization only)

        builder.add_ground_plane()

        self.model = builder.finalize()

        print(f"body_q={self.model.body_q}")
        print(f"body_qd={self.model.body_qd}")
        print(f"joint_q={self.model.joint_q}")
        print(f"joint_qd={self.model.joint_qd}")

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=24)

        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None  # no contacts for now

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state0.clear_forces()
            self.viewer.apply_forces(self.state0)
            self.solver.step(
                self.state0, self.state1, self.control, self.contacts, self.sim_dt
            )
            self.state0, self.state1 = self.state1, self.state0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--platform",
        type=str,
        default="astro",
        help="Which drone platform to simulate.",
    )
    viewer, args = newton.examples.init(parser)

    drone = Drone(viewer, args.platform)
    newton.examples.run(drone, args)
