#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool


class Quest3PoseToTwist(Node):
    """Map Quest 3 controller relative pose into a Cartesian twist command."""

    def __init__(self) -> None:
        super().__init__('quest3_pose_to_twist')

        self.declare_parameter('controller_pose_topic', '/quest3/right_controller/pose')
        self.declare_parameter('enable_topic', '/quest3/right_controller/grip_pressed')
        self.declare_parameter('calibrate_topic', '/quest3/right_controller/calibrate_pressed')
        self.declare_parameter('reset_calibration_topic', '/quest3/right_controller/reset_calibration_pressed')
        self.declare_parameter('head_forward_topic', '/quest3/head_forward')
        self.declare_parameter('twist_topic', '/vr_cartesian_velocity_controller/twist_cmd')
        self.declare_parameter('input_timeout_sec', 0.25)
        self.declare_parameter('publish_rate_hz', 60.0)
        self.declare_parameter('linear_gain', 0.8)
        self.declare_parameter('angular_gain', 0.8)
        self.declare_parameter('max_linear_velocity_mps', 0.08)
        self.declare_parameter('max_angular_velocity_radps', 0.35)
        self.declare_parameter('deadband_position_m', 0.01)
        self.declare_parameter('deadband_rotation_rad', 0.06)
        self.declare_parameter('deadband_linear_velocity_mps', 0.01)
        self.declare_parameter('deadband_angular_velocity_radps', 0.03)
        self.declare_parameter('motion_mode', 'hand_velocity')
        self.declare_parameter('input_smoothing_alpha', 0.15)
        self.declare_parameter('idle_decay_alpha', 0.08)
        self.declare_parameter('max_controller_linear_velocity_mps', 0.4)
        self.declare_parameter('max_controller_angular_velocity_radps', 1.2)
        self.declare_parameter('linear_axis_map', ['x', 'z', 'y'])
        self.declare_parameter('angular_axis_map', ['x', 'z', 'y'])
        self.declare_parameter('use_heading_calibration', False)
        self.declare_parameter('calibration_forward_axis', 'x')
        self.declare_parameter('calibration_min_forward_norm', 0.2)
        self.declare_parameter('invert_x', False)
        self.declare_parameter('invert_y', False)
        self.declare_parameter('invert_z', False)
        self.declare_parameter('send_zero_when_disabled', True)

        self.controller_pose_topic = self.get_parameter('controller_pose_topic').value
        self.enable_topic = self.get_parameter('enable_topic').value
        self.calibrate_topic = self.get_parameter('calibrate_topic').value
        self.reset_calibration_topic = self.get_parameter('reset_calibration_topic').value
        self.head_forward_topic = self.get_parameter('head_forward_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.input_timeout_sec = float(self.get_parameter('input_timeout_sec').value)
        self.publish_rate_hz = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.linear_gain = float(self.get_parameter('linear_gain').value)
        self.angular_gain = float(self.get_parameter('angular_gain').value)
        self.max_linear_velocity_mps = float(self.get_parameter('max_linear_velocity_mps').value)
        self.max_angular_velocity_radps = float(self.get_parameter('max_angular_velocity_radps').value)
        self.deadband_position_m = float(self.get_parameter('deadband_position_m').value)
        self.deadband_rotation_rad = float(self.get_parameter('deadband_rotation_rad').value)
        self.deadband_linear_velocity_mps = float(self.get_parameter('deadband_linear_velocity_mps').value)
        self.deadband_angular_velocity_radps = float(self.get_parameter('deadband_angular_velocity_radps').value)
        self.motion_mode = str(self.get_parameter('motion_mode').value).lower()
        if self.motion_mode not in ('hand_velocity', 'anchor_offset'):
            raise ValueError('motion_mode must be "hand_velocity" or "anchor_offset"')
        self.input_smoothing_alpha = max(
            0.0,
            min(1.0, float(self.get_parameter('input_smoothing_alpha').value)),
        )
        self.idle_decay_alpha = max(
            0.0,
            min(1.0, float(self.get_parameter('idle_decay_alpha').value)),
        )
        self.max_controller_linear_velocity_mps = max(
            0.0,
            float(self.get_parameter('max_controller_linear_velocity_mps').value),
        )
        self.max_controller_angular_velocity_radps = max(
            0.0,
            float(self.get_parameter('max_controller_angular_velocity_radps').value),
        )
        self.linear_axis_map = [str(axis) for axis in self.get_parameter('linear_axis_map').value]
        if len(self.linear_axis_map) != 3:
            raise ValueError('linear_axis_map must contain exactly 3 entries, e.g. ["x", "z", "y"]')
        self.angular_axis_map = [str(axis) for axis in self.get_parameter('angular_axis_map').value]
        if len(self.angular_axis_map) != 3:
            raise ValueError('angular_axis_map must contain exactly 3 entries, e.g. ["x", "z", "y"]')
        self.use_heading_calibration = bool(self.get_parameter('use_heading_calibration').value)
        self.calibration_forward_axis = str(self.get_parameter('calibration_forward_axis').value).lower()
        if self.calibration_forward_axis not in ('x', 'y'):
            raise ValueError('calibration_forward_axis must be "x" or "y"')
        self.calibration_min_forward_norm = max(
            1e-6,
            float(self.get_parameter('calibration_min_forward_norm').value),
        )
        self.invert_x = bool(self.get_parameter('invert_x').value)
        self.invert_y = bool(self.get_parameter('invert_y').value)
        self.invert_z = bool(self.get_parameter('invert_z').value)
        self.send_zero_when_disabled = bool(self.get_parameter('send_zero_when_disabled').value)

        self.latest_pose: Optional[PoseStamped] = None
        self.latest_head_forward: Optional[Tuple[float, float, float]] = None
        self.latest_pose_time = self.get_clock().now()
        self.previous_pose: Optional[PoseStamped] = None
        self.previous_pose_time = None
        self.filtered_twist = [0.0] * 6
        self.enabled = False
        self.anchor_pose: Optional[PoseStamped] = None
        self.heading_offset_rad = 0.0
        self.calibrated = False
        self.calibrate_pressed = False
        self.reset_calibration_pressed = False

        self.twist_pub = self.create_publisher(TwistStamped, self.twist_topic, 10)
        self.create_subscription(PoseStamped, self.controller_pose_topic, self._pose_callback, 10)
        self.create_subscription(Bool, self.enable_topic, self._enable_callback, 10)
        self.create_subscription(Bool, self.calibrate_topic, self._calibrate_callback, 10)
        self.create_subscription(Bool, self.reset_calibration_topic, self._reset_calibration_callback, 10)
        self.create_subscription(Vector3Stamped, self.head_forward_topic, self._head_forward_callback, 10)
        self.create_timer(1.0 / self.publish_rate_hz, self._timer_callback)

        self.get_logger().info(
            f'Quest3 pose bridge ready. pose={self.controller_pose_topic} '
            f'enable={self.enable_topic} calibrate={self.calibrate_topic} twist={self.twist_topic}'
        )

    def _pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg
        self.latest_pose_time = self.get_clock().now()

    def _head_forward_callback(self, msg: Vector3Stamped) -> None:
        self.latest_head_forward = (msg.vector.x, msg.vector.y, msg.vector.z)

    def _enable_callback(self, msg: Bool) -> None:
        new_enabled = bool(msg.data)
        if new_enabled and not self.enabled:
            self.anchor_pose = self.latest_pose
            self.previous_pose = self.latest_pose
            self.previous_pose_time = self.latest_pose_time
            self.filtered_twist = [0.0] * 6
            self.get_logger().info('VR teleop enabled; controller anchor captured.')
        if not new_enabled and self.enabled:
            self.anchor_pose = None
            self.previous_pose = None
            self.previous_pose_time = None
            self.filtered_twist = [0.0] * 6
            self._publish_zero()
            self.get_logger().info('VR teleop disabled.')
        self.enabled = new_enabled

    def _calibrate_callback(self, msg: Bool) -> None:
        pressed = bool(msg.data)
        if pressed and not self.calibrate_pressed:
            self._calibrate_heading()
        self.calibrate_pressed = pressed

    def _reset_calibration_callback(self, msg: Bool) -> None:
        pressed = bool(msg.data)
        if pressed and not self.reset_calibration_pressed:
            self.heading_offset_rad = 0.0
            self.calibrated = False
            self.get_logger().info('VR heading calibration reset.')
        self.reset_calibration_pressed = pressed

    def _timer_callback(self) -> None:
        if not self.enabled or self.latest_pose is None or self.anchor_pose is None:
            if self.send_zero_when_disabled:
                self._publish_zero()
            return
        age = (self.get_clock().now() - self.latest_pose_time).nanoseconds * 1e-9
        if age > self.input_timeout_sec:
            self._publish_zero()
            return

        if self.motion_mode == 'hand_velocity':
            motion = self._compute_hand_velocity()
            if motion is None:
                self._publish_twist_command(
                    self._smooth_twist((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), self.idle_decay_alpha)
                )
                return
            dx, dy, dz, rx, ry, rz = motion
            linear_deadband = self.deadband_linear_velocity_mps
            angular_deadband = self.deadband_angular_velocity_radps
        else:
            dx = self.latest_pose.pose.position.x - self.anchor_pose.pose.position.x
            dy = self.latest_pose.pose.position.y - self.anchor_pose.pose.position.y
            dz = self.latest_pose.pose.position.z - self.anchor_pose.pose.position.z
            q_anchor = self._quat(self.anchor_pose)
            q_current = self._quat(self.latest_pose)
            q_delta = self._quat_multiply(q_current, self._quat_inverse(q_anchor))
            rx, ry, rz = self._quat_to_rotvec(q_delta)
            linear_deadband = self.deadband_position_m
            angular_deadband = self.deadband_rotation_rad

        if self.invert_x:
            dx = -dx
        if self.invert_y:
            dy = -dy
        if self.invert_z:
            dz = -dz

        mapped_dx, mapped_dy, mapped_dz = self._map_vector(
            (dx, dy, dz),
            self.linear_axis_map,
        )
        if self.use_heading_calibration and self.calibrated:
            mapped_dx, mapped_dy = self._rotate_xy(
                mapped_dx,
                mapped_dy,
                self.heading_offset_rad,
            )
        vx = self._deadband(mapped_dx, linear_deadband) * self.linear_gain
        vy = self._deadband(mapped_dy, linear_deadband) * self.linear_gain
        vz = self._deadband(mapped_dz, linear_deadband) * self.linear_gain

        rx, ry, rz = self._map_vector((rx, ry, rz), self.angular_axis_map)
        wx = self._deadband(rx, angular_deadband) * self.angular_gain
        wy = self._deadband(ry, angular_deadband) * self.angular_gain
        wz = self._deadband(rz, angular_deadband) * self.angular_gain

        command = (
            self._clamp(vx, self.max_linear_velocity_mps),
            self._clamp(vy, self.max_linear_velocity_mps),
            self._clamp(vz, self.max_linear_velocity_mps),
            self._clamp(wx, self.max_angular_velocity_radps),
            self._clamp(wy, self.max_angular_velocity_radps),
            self._clamp(wz, self.max_angular_velocity_radps),
        )
        self._publish_twist_command(self._smooth_twist(command, self.input_smoothing_alpha))

    def _calibrate_heading(self) -> None:
        if not self.use_heading_calibration:
            self.get_logger().info('VR heading calibration ignored because it is disabled.')
            return
        if self.latest_head_forward is None:
            self.get_logger().warn('Cannot calibrate VR heading yet: no head_forward data received.')
            return

        mx, my, _ = self._map_vector(self.latest_head_forward, self.linear_axis_map)
        norm = math.hypot(mx, my)
        if norm < self.calibration_min_forward_norm:
            self.get_logger().warn('Cannot calibrate VR heading: headset forward vector is too small.')
            return

        current_heading = math.atan2(my, mx)
        desired_heading = 0.0 if self.calibration_forward_axis == 'x' else math.pi / 2.0
        self.heading_offset_rad = self._normalize_angle(desired_heading - current_heading)
        self.calibrated = True
        self.anchor_pose = self.latest_pose
        self._publish_zero()
        self.get_logger().info(
            f'VR heading calibrated. offset={math.degrees(self.heading_offset_rad):.1f}deg '
            f'forward_axis={self.calibration_forward_axis}'
        )

    def _compute_hand_velocity(self):
        if self.previous_pose is None or self.previous_pose_time is None:
            self.previous_pose = self.latest_pose
            self.previous_pose_time = self.latest_pose_time
            return None

        dt = (self.latest_pose_time - self.previous_pose_time).nanoseconds * 1e-9
        if dt <= 1e-4:
            return None

        dx = (self.latest_pose.pose.position.x - self.previous_pose.pose.position.x) / dt
        dy = (self.latest_pose.pose.position.y - self.previous_pose.pose.position.y) / dt
        dz = (self.latest_pose.pose.position.z - self.previous_pose.pose.position.z) / dt
        dx = self._clamp(dx, self.max_controller_linear_velocity_mps)
        dy = self._clamp(dy, self.max_controller_linear_velocity_mps)
        dz = self._clamp(dz, self.max_controller_linear_velocity_mps)

        q_previous = self._quat(self.previous_pose)
        q_current = self._quat(self.latest_pose)
        q_delta = self._quat_multiply(q_current, self._quat_inverse(q_previous))
        rx, ry, rz = self._quat_to_rotvec(q_delta)
        rx = self._clamp(rx / dt, self.max_controller_angular_velocity_radps)
        ry = self._clamp(ry / dt, self.max_controller_angular_velocity_radps)
        rz = self._clamp(rz / dt, self.max_controller_angular_velocity_radps)

        self.previous_pose = self.latest_pose
        self.previous_pose_time = self.latest_pose_time
        return dx, dy, dz, rx, ry, rz

    def _smooth_twist(self, target, alpha: float):
        for i, value in enumerate(target):
            self.filtered_twist[i] += alpha * (value - self.filtered_twist[i])
        return tuple(self.filtered_twist)

    def _publish_twist_command(self, command) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'fr3_link0'
        msg.twist.linear.x = command[0]
        msg.twist.linear.y = command[1]
        msg.twist.linear.z = command[2]
        msg.twist.angular.x = command[3]
        msg.twist.angular.y = command[4]
        msg.twist.angular.z = command[5]
        self.twist_pub.publish(msg)

    def _publish_zero(self) -> None:
        self.filtered_twist = [0.0] * 6
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'fr3_link0'
        self.twist_pub.publish(msg)

    @staticmethod
    def _deadband(value: float, threshold: float) -> float:
        return 0.0 if abs(value) < threshold else value

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(-limit, min(limit, value))

    @staticmethod
    def _rotate_xy(x: float, y: float, angle: float) -> Tuple[float, float]:
        c = math.cos(angle)
        s = math.sin(angle)
        return c * x - s * y, s * x + c * y

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _map_vector(values: Tuple[float, float, float], axis_map) -> Tuple[float, float, float]:
        source = {'x': values[0], 'y': values[1], 'z': values[2]}
        mapped = []
        for axis in axis_map:
            sign = -1.0 if axis.startswith('-') else 1.0
            key = axis[1:] if axis.startswith('-') else axis
            if key not in source:
                raise ValueError(f'Invalid axis mapping entry: {axis}')
            mapped.append(sign * source[key])
        return mapped[0], mapped[1], mapped[2]

    @staticmethod
    def _quat(msg: PoseStamped) -> Tuple[float, float, float, float]:
        q = msg.pose.orientation
        return q.x, q.y, q.z, q.w

    @staticmethod
    def _quat_inverse(q):
        x, y, z, w = q
        norm = x * x + y * y + z * z + w * w
        if norm <= 1e-12:
            return 0.0, 0.0, 0.0, 1.0
        return -x / norm, -y / norm, -z / norm, w / norm

    @staticmethod
    def _quat_multiply(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _quat_to_rotvec(q):
        x, y, z, w = q
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 1e-12:
            return 0.0, 0.0, 0.0
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        if w < 0.0:
            x, y, z, w = -x, -y, -z, -w
        sin_half = math.sqrt(max(0.0, 1.0 - w * w))
        if sin_half < 1e-6:
            return 2.0 * x, 2.0 * y, 2.0 * z
        angle = 2.0 * math.atan2(sin_half, w)
        return x / sin_half * angle, y / sin_half * angle, z / sin_half * angle


def main() -> None:
    rclpy.init()
    node = Quest3PoseToTwist()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
