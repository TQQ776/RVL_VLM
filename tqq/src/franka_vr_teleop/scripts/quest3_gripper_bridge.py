#!/usr/bin/env python3
import rclpy
from franka_msgs.action import Move
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool


class Quest3GripperBridge(Node):
    def __init__(self) -> None:
        super().__init__('quest3_gripper_bridge')
        self.declare_parameter('close_button_topic', '/quest3/right_controller/trigger_pressed')
        self.declare_parameter('gripper_action', '/franka_gripper/move')
        self.declare_parameter('open_position_m', 0.08)
        self.declare_parameter('closed_position_m', 0.0)
        self.declare_parameter('speed_mps', 0.05)

        self.close_button_topic = self.get_parameter('close_button_topic').value
        self.gripper_action = self.get_parameter('gripper_action').value
        self.open_position_m = float(self.get_parameter('open_position_m').value)
        self.closed_position_m = float(self.get_parameter('closed_position_m').value)
        self.speed_mps = float(self.get_parameter('speed_mps').value)
        self.last_state = None

        self.client = ActionClient(self, Move, self.gripper_action)
        self.create_subscription(Bool, self.close_button_topic, self._button_callback, 10)
        self.get_logger().info(
            f'Quest3 gripper bridge ready. button={self.close_button_topic} '
            f'action={self.gripper_action}'
        )

    def _button_callback(self, msg: Bool) -> None:
        close = bool(msg.data)
        if self.last_state is not None and close == self.last_state:
            return
        self.last_state = close
        if not self.client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn(f'Gripper action not available: {self.gripper_action}')
            return
        goal = Move.Goal()
        goal.width = self.closed_position_m if close else self.open_position_m
        goal.speed = self.speed_mps
        self.client.send_goal_async(goal)
        self.get_logger().info(f"Sent gripper {'close' if close else 'open'} command.")


def main() -> None:
    rclpy.init()
    node = Quest3GripperBridge()
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
