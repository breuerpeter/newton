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

        self.motor_diam_m = 0.08
        self.motor_height_m = 0.05

        self.lnd_gear_angle_rad = wp.pi / 6
        self.lnd_gear_length_m = 0.2
        self.lnd_gear_diam_m = 0.02

        # Alta X Gen2
        if self.platform == "altaxgen2":
            self.body_hx_m *= 2
            self.body_hy_m *= 2
            self.body_hz_m *= 2

            self.body_boom_diam_m *= 2
            self.body_boom_len_m *= 2

            self.motor_diam_m *= 2
            self.motor_height_m *= 2

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

        boom_rot_y = wp.quat_from_axis_angle(
            wp.vec3(0, 1, 0), wp.half_pi
        )  # rotate cylinder to point along x

        boom_half_length = self.body_boom_len_m / 2
        body_diagonal_xy = math.sqrt(self.body_hx_m**2 + self.body_hy_m**2)
        boom_radius = self.body_boom_diam_m / 2
        diagonal_boom = body_diagonal_xy + boom_half_length - boom_radius
        diagonal_motor = diagonal_boom + boom_half_length

        lnd_gear_rot_y = wp.quat_from_axis_angle(
            wp.vec3(0, 1, 0), -self.lnd_gear_angle_rad
        )

        for id in range(4):
            # add booms
            boom_angle = (2 * id + 1) * wp.pi / 4
            boom_rot_z = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), boom_angle)
            boom_rot = (
                boom_rot_z * boom_rot_y
            )  # rotate around world y-axis first, then world z-axis

            boom_shift = wp.vec3(
                diagonal_boom * math.cos(boom_angle),
                diagonal_boom * math.sin(boom_angle),
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

            # add motors
            motor_shift = wp.vec3(
                diagonal_motor * math.cos(boom_angle),
                diagonal_motor * math.sin(boom_angle),
                0.0,
            )

            builder.add_shape_cylinder(
                body,
                xform=wp.transform(motor_shift, wp.quat_identity()),
                radius=self.motor_diam_m / 2,
                half_height=self.motor_height_m / 2,
                cfg=newton.ModelBuilder.ShapeConfig(density=1000),
            )

            # add landing gear
            lnd_gear_rot = (
                boom_rot_z * lnd_gear_rot_y
            )  # rotate around world y-axis first, then world z-axis

            top_local = wp.vec3(0.0, 0.0, self.lnd_gear_length_m / 2)
            top_offset = wp.quat_rotate(lnd_gear_rot, top_local)

            attachment_point = wp.vec3(
                body_diagonal_xy * math.cos(boom_angle),
                body_diagonal_xy * math.sin(boom_angle),
                -self.body_hz_m,
            )

            lnd_gear_shift = attachment_point - top_offset

            builder.add_shape_cylinder(
                body,
                xform=wp.transform(lnd_gear_shift, lnd_gear_rot),
                radius=self.lnd_gear_diam_m / 2,
                half_height=self.lnd_gear_length_m / 2,
                cfg=newton.ModelBuilder.ShapeConfig(density=self.carbon_fiber_density),
            )

        # TODO: propellers (no mass, for visualization only)

        builder.add_ground_plane()

        self.model = builder.finalize()

        print(f"body_q={self.model.body_q}")
        print(f"body_qd={self.model.body_qd}")
        print(f"joint_q={self.model.joint_q}")
        print(f"joint_qd={self.model.joint_qd}")

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=224)

        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state0)

        print(f"control dim {self.model.joint_dof_count}")

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
            self.control.joint_f.assign([0.0, 0.0, 80.0, 0.0, 0.0, 0.0])
            self.contacts = self.model.collide(self.state0)
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
