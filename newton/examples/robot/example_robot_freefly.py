import math
import time

import warp as wp
from pymavlink import mavutil

import newton
import newton.examples


class MAVLinkInterface:
    """Handles MAVLink communication with PX4 SITL."""

    def __init__(
        self,
        connection_string: str = "tcpin:0.0.0.0:4560",
        system_id: int = 245,
        component_id: int = getattr(
            mavutil.mavlink,
            "MAV_COMP_ID_SIMULATOR",
            mavutil.mavlink.MAV_COMP_ID_SYSTEM_CONTROL,  # fallback for older dialects
        ),
    ):
        """
        Initialize MAVLink connection.

        Args:
            connection_string: MAVLink connection string.
                For PX4 SITL lockstep, use "tcpin:0.0.0.0:4560" to accept
                connection from PX4 (bidirectional TCP).
        """
        print(
            f"Waiting for PX4 connection on {connection_string} "
            f"(sysid={system_id}, compid={component_id})..."
        )
        self.mav = mavutil.mavlink_connection(
            connection_string,
            # Use a simulator/system id distinct from PX4 (default sysid 1) to avoid
            # PX4 treating our traffic as if it originated from itself, which causes
            # "Ignore command 512 from 1/1 to 0/0" spam.
            source_system=system_id,
            source_component=component_id,
        )
        # Explicitly set our src ids on the wire and the PX4 targets. Some pymavlink
        # transports ignore the constructor args when the connection is inbound.
        self.mav.mav.srcSystem = system_id
        self.mav.mav.srcComponent = component_id
        self.mav.target_system = 1
        self.mav.target_component = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1

        print(
            f"Using MAVLink ids src={self.mav.mav.srcSystem}/{self.mav.mav.srcComponent} "
            f"target={self.mav.target_system}/{self.mav.target_component}"
        )

        self.actuator_controls = [0.0] * 16
        self.armed = False
        self.last_heartbeat_time = time.time()
        self.heartbeat_interval = 1.0

        print(f"MAVLink interface initialized: {connection_string}")

    def send_heartbeat(self):
        """Send heartbeat to PX4."""
        current_time = time.time()
        if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                0,  # base_mode
                0,  # custom_mode
                mavutil.mavlink.MAV_STATE_ACTIVE,
            )
            self.last_heartbeat_time = current_time

    def receive_actuator_commands(self):
        """
        Receive actuator commands from PX4.

        Returns:
            List of actuator values (normalized 0-1 for motors).
        """
        # Non-blocking receive
        msg = self.mav.recv_match(blocking=False)

        while msg is not None:
            msg_type = msg.get_type()

            if msg_type == "HIL_ACTUATOR_CONTROLS":
                self.actuator_controls = list(msg.controls)
                self.armed = (msg.mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

            elif msg_type == "HEARTBEAT":
                pass  # Track PX4 heartbeat

            # Check for more messages
            msg = self.mav.recv_match(blocking=False)

        return self.actuator_controls

    def send_hil_sensor(
        self,
        time_usec: int,
        xacc: float = 0.0,
        yacc: float = 0.0,
        zacc: float = -9.81,
        xgyro: float = 0.0,
        ygyro: float = 0.0,
        zgyro: float = 0.0,
        xmag: float = 0.2,
        ymag: float = 0.0,
        zmag: float = 0.4,
        abs_pressure: float = 1013.25,
        diff_pressure: float = 0.0,
        pressure_alt: float = 0.0,
        temperature: float = 25.0,
        fields_updated: int = 0x1FFF,  # All fields updated
    ):
        """
        Send HIL_SENSOR message to PX4.

        Args:
            time_usec: Timestamp in microseconds
            xacc, yacc, zacc: Accelerometer readings [m/s^2]
            xgyro, ygyro, zgyro: Gyroscope readings [rad/s]
            xmag, ymag, zmag: Magnetometer readings [gauss]
            abs_pressure: Absolute pressure [hPa]
            diff_pressure: Differential pressure [hPa]
            pressure_alt: Altitude from pressure [m]
            temperature: Temperature [degC]
            fields_updated: Bitmask of updated fields
        """
        self.mav.mav.hil_sensor_send(
            time_usec,
            xacc,
            yacc,
            zacc,
            xgyro,
            ygyro,
            zgyro,
            xmag,
            ymag,
            zmag,
            abs_pressure,
            diff_pressure,
            pressure_alt,
            temperature,
            fields_updated,
            0,  # id (sensor instance)
        )

    def send_hil_gps(
        self,
        time_usec: int,
        lat: int = 473977418,  # 47.3977418 degrees (Zurich)
        lon: int = 85455939,  # 8.5455939 degrees
        alt: int = 488000,  # 488m above MSL in mm
        eph: int = 100,  # GPS HDOP
        epv: int = 100,  # GPS VDOP
        vel: int = 0,  # GPS ground speed [cm/s]
        vn: int = 0,  # GPS velocity north [cm/s]
        ve: int = 0,  # GPS velocity east [cm/s]
        vd: int = 0,  # GPS velocity down [cm/s]
        cog: int = 0,  # Course over ground [cdeg]
        fix_type: int = 3,  # 3D fix
        satellites_visible: int = 10,
    ):
        """
        Send HIL_GPS message to PX4.

        Args:
            time_usec: Timestamp in microseconds
            lat: Latitude [degE7]
            lon: Longitude [degE7]
            alt: Altitude MSL [mm]
            eph: GPS HDOP [cm]
            epv: GPS VDOP [cm]
            vel: GPS ground speed [cm/s]
            vn, ve, vd: GPS velocity NED [cm/s]
            cog: Course over ground [cdeg]
            fix_type: GPS fix type (0=no fix, 3=3D fix)
            satellites_visible: Number of satellites
        """
        self.mav.mav.hil_gps_send(
            time_usec,
            fix_type,
            lat,
            lon,
            alt,
            eph,
            epv,
            vel,
            vn,
            ve,
            vd,
            cog,
            satellites_visible,
        )

    def send_hil_state_quaternion(
        self,
        time_usec: int,
        attitude_quaternion: list = None,
        rollspeed: float = 0.0,
        pitchspeed: float = 0.0,
        yawspeed: float = 0.0,
        lat: int = 473977418,
        lon: int = 85455939,
        alt: int = 488000,
        vx: int = 0,
        vy: int = 0,
        vz: int = 0,
        ind_airspeed: int = 0,
        true_airspeed: int = 0,
        xacc: int = 0,
        yacc: int = 0,
        zacc: int = 0,
    ):
        """
        Send HIL_STATE_QUATERNION message to PX4.

        Args:
            time_usec: Timestamp in microseconds
            attitude_quaternion: [w, x, y, z] quaternion
            rollspeed, pitchspeed, yawspeed: Angular velocities [rad/s]
            lat, lon: Position [degE7]
            alt: Altitude MSL [mm]
            vx, vy, vz: Velocity NED [cm/s]
            ind_airspeed, true_airspeed: Airspeed [cm/s]
            xacc, yacc, zacc: Acceleration [mG]
        """
        if attitude_quaternion is None:
            attitude_quaternion = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion

        self.mav.mav.hil_state_quaternion_send(
            time_usec,
            attitude_quaternion,
            rollspeed,
            pitchspeed,
            yawspeed,
            lat,
            lon,
            alt,
            vx,
            vy,
            vz,
            ind_airspeed,
            true_airspeed,
            xacc,
            yacc,
            zacc,
        )


class Drone:
    def __init__(self, viewer, platform: str = "astro", use_px4: bool = False, args=None):
        self.platform = platform
        self.viewer = viewer
        self.use_px4 = use_px4

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # PX4 MAVLink interface
        self.mavlink = None
        if self.use_px4:
            self.mavlink = MAVLinkInterface()
            self.sensor_update_interval = 1.0 / 250.0  # 250 Hz IMU updates
            self.gps_update_interval = 0.1  # 10 Hz GPS updates
            self.last_sensor_time = 0.0
            self.last_gps_time = 0.0

        # Motor configuration for quadrotor (max thrust per motor in Newtons)
        self.max_motor_thrust = 25.0  # Adjust based on drone mass

        # FPS tracking
        self.fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
        self.last_fps_time = time.time()
        self.frame_count = 0

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
        # Disable CUDA graph capture when using PX4 (dynamic MAVLink I/O is incompatible)
        if wp.get_device().is_cuda and not self.use_px4:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        # Get actuator commands from PX4 if enabled
        if self.mavlink:
            self.mavlink.send_heartbeat()
            actuator_controls = self.mavlink.receive_actuator_commands()

            # Convert actuator commands to thrust force
            # PX4 sends normalized values [0, 1] for motors
            # Sum thrust from all 4 motors (channels 0-3)
            total_thrust = 0.0
            for i in range(4):
                motor_cmd = max(0.0, min(1.0, actuator_controls[i]))
                total_thrust += motor_cmd * self.max_motor_thrust

            # Apply as vertical force (z-axis in body frame)
            thrust_force = [0.0, 0.0, total_thrust, 0.0, 0.0, 0.0]
        else:
            # Default hover thrust when not connected to PX4
            thrust_force = [0.0, 0.0, 80.0, 0.0, 0.0, 0.0]

        for _ in range(self.sim_substeps):
            self.state0.clear_forces()
            self.viewer.apply_forces(self.state0)
            self.control.joint_f.assign(thrust_force)
            self.contacts = self.model.collide(self.state0)
            self.solver.step(
                self.state0, self.state1, self.control, self.contacts, self.sim_dt
            )
            self.state0, self.state1 = self.state1, self.state0

        # Send sensor data to PX4 if enabled
        if self.mavlink:
            self._send_sensor_data()

    def _send_sensor_data(self):
        """Send simulated sensor data to PX4."""
        time_usec = int(self.sim_time * 1e6)

        # Get current state from simulation
        # body_q contains position and quaternion for each body
        # body_qd contains linear and angular velocity for each body
        body_q = self.state0.body_q.numpy()
        body_qd = self.state0.body_qd.numpy()

        # Extract drone state (body index 0)
        # Position: [x, y, z]
        pos = body_q[0, :3]
        # Quaternion: [x, y, z, w] (warp convention)
        quat_xyzw = body_q[0, 3:7]
        # Convert to [w, x, y, z] for MAVLink
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        # Velocity: [vx, vy, vz, wx, wy, wz] (linear, angular)
        vel_linear = body_qd[0, :3]
        vel_angular = body_qd[0, 3:6]

        # Generic/placeholder sensor data
        # In a real implementation, these would be computed from simulation state

        # IMU data (accelerometer includes gravity in body frame)
        # For now, use placeholder values - gravity pointing down in world frame
        # TODO: Transform gravity to body frame using quaternion
        xacc = 0.0
        yacc = 0.0
        zacc = -9.81  # Gravity (placeholder, should be in body frame)

        # Gyroscope (angular velocity in body frame)
        # TODO: Transform to body frame
        xgyro = vel_angular[0]
        ygyro = vel_angular[1]
        zgyro = vel_angular[2]

        # Magnetometer (placeholder values for northern hemisphere)
        xmag = 0.2
        ymag = 0.0
        zmag = 0.4

        # Barometer
        # Approximate pressure from altitude (simplified model)
        altitude = pos[2]  # z is up in simulation
        sea_level_pressure = 1013.25  # hPa
        # Barometric formula approximation
        abs_pressure = sea_level_pressure * (1 - 2.25577e-5 * altitude) ** 5.25588
        pressure_alt = altitude

        # Send HIL_SENSOR at simulation rate
        self.mavlink.send_hil_sensor(
            time_usec=time_usec,
            xacc=xacc,
            yacc=yacc,
            zacc=zacc,
            xgyro=xgyro,
            ygyro=ygyro,
            zgyro=zgyro,
            xmag=xmag,
            ymag=ymag,
            zmag=zmag,
            abs_pressure=abs_pressure,
            pressure_alt=pressure_alt,
        )

        # Send GPS at lower rate (10 Hz)
        if self.sim_time - self.last_gps_time >= self.gps_update_interval:
            self.last_gps_time = self.sim_time

            # Convert simulation position to GPS coordinates
            # Using a reference point (Zurich) and adding local offsets
            # 1 degree latitude ~= 111km, 1 degree longitude ~= 111km * cos(lat)
            ref_lat = 47.3977418  # Reference latitude
            ref_lon = 8.5455939  # Reference longitude
            ref_alt = 488.0  # Reference altitude MSL [m]

            # Position offset in meters (x=East, y=North in typical GPS convention)
            # Simulation uses x=forward, y=left, z=up
            lat_offset = pos[0] / 111000.0  # Approximate
            lon_offset = pos[1] / (111000.0 * math.cos(math.radians(ref_lat)))

            lat = int((ref_lat + lat_offset) * 1e7)
            lon = int((ref_lon + lon_offset) * 1e7)
            alt = int((ref_alt + altitude) * 1000)  # mm

            # Velocity in cm/s (NED frame)
            # TODO: Proper frame transformation
            vn = int(vel_linear[0] * 100)
            ve = int(vel_linear[1] * 100)
            vd = int(-vel_linear[2] * 100)  # Down is negative z

            vel = int(math.sqrt(vel_linear[0] ** 2 + vel_linear[1] ** 2) * 100)

            self.mavlink.send_hil_gps(
                time_usec=time_usec,
                lat=lat,
                lon=lon,
                alt=alt,
                vn=vn,
                ve=ve,
                vd=vd,
                vel=vel,
            )

            # Also send full state for visualization/logging
            self.mavlink.send_hil_state_quaternion(
                time_usec=time_usec,
                attitude_quaternion=quat_wxyz,
                rollspeed=vel_angular[0],
                pitchspeed=vel_angular[1],
                yawspeed=vel_angular[2],
                lat=lat,
                lon=lon,
                alt=alt,
                vx=vn,
                vy=ve,
                vz=vd,
                xacc=int(xacc * 1000 / 9.81),  # Convert to mG
                yacc=int(yacc * 1000 / 9.81),
                zacc=int(zacc * 1000 / 9.81),
            )

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # FPS tracking
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed >= self.fps_update_interval:
            fps = self.frame_count / elapsed
            print(f"\rFPS: {fps:.1f}  ", end="", flush=True)
            self.frame_count = 0
            self.last_fps_time = current_time

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
    parser.add_argument(
        "--px4",
        action="store_true",
        help="Enable PX4 SITL communication via MAVLink.",
    )
    viewer, args = newton.examples.init(parser)

    drone = Drone(viewer, args.platform, use_px4=args.px4)
    newton.examples.run(drone, args)
