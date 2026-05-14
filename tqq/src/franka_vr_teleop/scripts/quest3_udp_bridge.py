#!/usr/bin/env python3
import json
import socket
import threading
from typing import Any, Optional, Sequence, Tuple

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Bool


class Quest3UdpBridge(Node):
    """Receive Quest 3 JSON packets over UDP and republish them as ROS topics."""

    def __init__(self) -> None:
        super().__init__('quest3_udp_bridge')

        self.declare_parameter('listen_host', '0.0.0.0')
        self.declare_parameter('listen_port', 5055)
        self.declare_parameter('frame_id', 'quest3_right_controller')
        self.declare_parameter('pose_topic', '/quest3/right_controller/pose')
        self.declare_parameter('grip_topic', '/quest3/right_controller/grip_pressed')
        self.declare_parameter('trigger_topic', '/quest3/right_controller/trigger_pressed')
        self.declare_parameter('calibrate_topic', '/quest3/right_controller/calibrate_pressed')
        self.declare_parameter('reset_calibration_topic', '/quest3/right_controller/reset_calibration_pressed')
        self.declare_parameter('head_forward_topic', '/quest3/head_forward')
        self.declare_parameter('max_packet_bytes', 65535)

        self.listen_host = str(self.get_parameter('listen_host').value)
        self.listen_port = int(self.get_parameter('listen_port').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.grip_topic = str(self.get_parameter('grip_topic').value)
        self.trigger_topic = str(self.get_parameter('trigger_topic').value)
        self.calibrate_topic = str(self.get_parameter('calibrate_topic').value)
        self.reset_calibration_topic = str(self.get_parameter('reset_calibration_topic').value)
        self.head_forward_topic = str(self.get_parameter('head_forward_topic').value)
        self.max_packet_bytes = max(1, int(self.get_parameter('max_packet_bytes').value))

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.grip_pub = self.create_publisher(Bool, self.grip_topic, 10)
        self.trigger_pub = self.create_publisher(Bool, self.trigger_topic, 10)
        self.calibrate_pub = self.create_publisher(Bool, self.calibrate_topic, 10)
        self.reset_calibration_pub = self.create_publisher(Bool, self.reset_calibration_topic, 10)
        self.head_forward_pub = self.create_publisher(Vector3Stamped, self.head_forward_topic, 10)

        self._stop_event = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self._socket.bind((self.listen_host, self.listen_port))
            self._socket.settimeout(0.25)
        except OSError:
            self._socket.close()
            raise

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'Quest3 UDP bridge listening on {self.listen_host}:{self.listen_port} '
            f'-> pose={self.pose_topic} grip={self.grip_topic} trigger={self.trigger_topic} '
            f'calibrate={self.calibrate_topic}'
        )

    def destroy_node(self) -> None:
        self._stop_event.set()
        try:
            self._socket.close()
        except OSError:
            pass
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        super().destroy_node()

    def _recv_loop(self) -> None:
        while not self._stop_event.is_set() and rclpy.ok():
            try:
                packet, addr = self._socket.recvfrom(self.max_packet_bytes)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                payload = json.loads(packet.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.get_logger().warn(f'Ignoring invalid UDP packet from {addr[0]}: {exc}')
                continue

            if not isinstance(payload, dict):
                self.get_logger().warn(f'Ignoring non-object UDP payload from {addr[0]}.')
                continue

            self._publish_pose_if_present(payload)
            self._publish_vector_if_present(payload, 'head_forward', self.head_forward_pub)
            self._publish_button_if_present(payload, 'grip_pressed', self.grip_pub, ('enabled', 'grip'))
            self._publish_button_if_present(
                payload,
                'trigger_pressed',
                self.trigger_pub,
                ('trigger', 'trigger_down'),
            )
            self._publish_button_if_present(
                payload,
                'calibrate_pressed',
                self.calibrate_pub,
                ('primary_pressed', 'primary_button', 'a_pressed', 'a_button'),
            )
            self._publish_button_if_present(
                payload,
                'reset_calibration_pressed',
                self.reset_calibration_pub,
                ('secondary_pressed', 'secondary_button', 'b_pressed', 'b_button'),
            )

    def _publish_pose_if_present(self, payload: dict[str, Any]) -> None:
        position, orientation = self._extract_pose_components(payload)
        if position is None or orientation is None:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        self.pose_pub.publish(msg)

    def _publish_button_if_present(
        self,
        payload: dict[str, Any],
        primary_key: str,
        publisher,
        aliases: Tuple[str, ...] = (),
    ) -> None:
        value = self._find_bool(payload, primary_key)
        if value is None:
            for alias in aliases:
                value = self._find_bool(payload, alias)
                if value is not None:
                    break
        if value is None:
            return
        msg = Bool()
        msg.data = value
        publisher.publish(msg)

    def _publish_vector_if_present(self, payload: dict[str, Any], key: str, publisher) -> None:
        vector = self._find_vector(payload, key)
        if vector is None:
            return

        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.vector.x = float(vector[0])
        msg.vector.y = float(vector[1])
        msg.vector.z = float(vector[2])
        publisher.publish(msg)

    @staticmethod
    def _find_bool(payload: dict[str, Any], key: str) -> Optional[bool]:
        if key not in payload:
            buttons = payload.get('buttons')
            if isinstance(buttons, dict) and key in buttons:
                return bool(buttons[key])
            return None
        return bool(payload[key])

    def _find_vector(self, payload: dict[str, Any], key: str) -> Optional[Tuple[float, float, float]]:
        value = payload.get(key)
        if isinstance(value, dict):
            value = [value.get('x'), value.get('y'), value.get('z')]
        vector = self._sequence_or_none(value)
        if vector is None or len(vector) != 3:
            return None
        return vector[0], vector[1], vector[2]

    def _extract_pose_components(
        self,
        payload: dict[str, Any],
    ) -> Tuple[Optional[Sequence[float]], Optional[Sequence[float]]]:
        pose = payload.get('pose')
        if isinstance(pose, dict):
            position = self._sequence_or_none(pose.get('position'))
            orientation = self._sequence_or_none(pose.get('orientation'))
            if orientation is None:
                orientation = self._sequence_or_none(pose.get('quat'))
            if position is not None and orientation is not None:
                return position, orientation

        position = self._sequence_or_none(payload.get('position'))
        orientation = self._sequence_or_none(payload.get('orientation'))
        if position is not None and orientation is not None:
            return position, orientation

        flat_position = self._sequence_or_none(
            [payload.get('x'), payload.get('y'), payload.get('z')]
        )
        if flat_position is not None and all(key in payload for key in ('qx', 'qy', 'qz', 'qw')):
            return flat_position, self._sequence_or_none(
                [payload.get('qx'), payload.get('qy'), payload.get('qz'), payload.get('qw')]
            )

        if flat_position is not None and all(key in payload for key in ('ox', 'oy', 'oz', 'ow')):
            return flat_position, self._sequence_or_none(
                [payload.get('ox'), payload.get('oy'), payload.get('oz'), payload.get('ow')]
            )

        return None, None

    @staticmethod
    def _sequence_or_none(values: Any) -> Optional[Tuple[float, ...]]:
        if not isinstance(values, (list, tuple)):
            return None
        if len(values) not in (3, 4):
            return None
        try:
            return tuple(float(v) for v in values)
        except (TypeError, ValueError):
            return None


def main() -> None:
    rclpy.init()
    node = Quest3UdpBridge()
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
