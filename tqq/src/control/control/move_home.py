import math
import time
from typing import Dict, List, Optional

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MoveHome(Node):
    """Move the FR3 arm back to a named home joint configuration."""

    def __init__(self) -> None:
        super().__init__('move_home')

        self.latest_joint_positions: Optional[Dict[str, float]] = None

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('trajectory_action', '/fr3_arm_controller/follow_joint_trajectory')
        self.declare_parameter('joint_names', [
            'fr3_joint1',
            'fr3_joint2',
            'fr3_joint3',
            'fr3_joint4',
            'fr3_joint5',
            'fr3_joint6',
            'fr3_joint7',
        ])
        self.declare_parameter(
            'home_joint_positions_deg',
            [74.0, -3.0, -7.0, -115.0, -1.0, 110.0, 22.0],
        )
        self.declare_parameter('move_duration_sec', 4.0)
        self.declare_parameter('trajectory_dt_sec', 0.05)
        self.declare_parameter('trajectory_start_delay_sec', 0.1)
        self.declare_parameter('wait_for_joint_state_sec', 3.0)
        self.declare_parameter('wait_for_server_sec', 10.0)

        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.trajectory_action = str(self.get_parameter('trajectory_action').value)
        self.joint_names = [str(name) for name in self.get_parameter('joint_names').value]
        self.home_joint_positions_deg = [
            float(value) for value in self.get_parameter('home_joint_positions_deg').value
        ]
        self.move_duration_sec = float(self.get_parameter('move_duration_sec').value)
        self.trajectory_dt_sec = float(self.get_parameter('trajectory_dt_sec').value)
        self.trajectory_start_delay_sec = float(
            self.get_parameter('trajectory_start_delay_sec').value
        )
        self.wait_for_joint_state_sec = float(self.get_parameter('wait_for_joint_state_sec').value)
        self.wait_for_server_sec = float(self.get_parameter('wait_for_server_sec').value)

        if len(self.joint_names) != len(self.home_joint_positions_deg):
            raise ValueError('joint_names and home_joint_positions_deg must have the same length')

        self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)
        self.client = ActionClient(self, FollowJointTrajectory, self.trajectory_action)

    def joint_state_callback(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in self.joint_names):
            self.latest_joint_positions = {
                name: float(positions[name]) for name in self.joint_names
            }

    def run(self) -> int:
        self.get_logger().info(
            'Waiting for trajectory action server '
            f'{self.trajectory_action} ...'
        )
        if not self.client.wait_for_server(timeout_sec=self.wait_for_server_sec):
            self.get_logger().error(f'Action server not available: {self.trajectory_action}')
            return 1
        start_positions = self._wait_for_current_joint_positions()
        if start_positions is None:
            self.get_logger().error(
                f'No complete joint state received on {self.joint_states_topic}; '
                'cannot build a smooth home trajectory.'
            )
            return 1

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.header.stamp = (
            self.get_clock().now() + Duration(
                seconds=max(0.0, self.trajectory_start_delay_sec)
            )
        ).to_msg()
        goal.trajectory.joint_names = list(self.joint_names)
        goal.trajectory.points = self._make_smooth_points(start_positions)

        self.get_logger().info(
            f'Sending smooth home trajectory ({len(goal.trajectory.points)} points, '
            f'duration={self.move_duration_sec:.2f}s): '
            + ', '.join(
                f'{name}={deg:.1f}deg' for name, deg in zip(self.joint_names, self.home_joint_positions_deg)
            )
        )
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Home goal was rejected.')
            return 1

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Home motion finished successfully.')
            return 0

        self.get_logger().error(
            f'Home motion failed: error_code={result.error_code}, error_string={result.error_string}'
        )
        return 1

    def _wait_for_current_joint_positions(self) -> Optional[Dict[str, float]]:
        deadline = time.monotonic() + max(0.0, self.wait_for_joint_state_sec)
        while time.monotonic() <= deadline:
            if self.latest_joint_positions is not None:
                return dict(self.latest_joint_positions)
            rclpy.spin_once(self, timeout_sec=0.05)
        return None

    def _make_smooth_points(self, start_positions: Dict[str, float]) -> List[JointTrajectoryPoint]:
        goal_positions = {
            name: math.radians(value)
            for name, value in zip(self.joint_names, self.home_joint_positions_deg)
        }
        duration = max(0.5, self.move_duration_sec)
        dt = min(max(0.01, self.trajectory_dt_sec), duration)
        steps = max(2, int(math.ceil(duration / dt)))

        points: List[JointTrajectoryPoint] = []
        for index in range(steps + 1):
            elapsed = min(duration, duration * index / steps)
            u = 0.0 if duration <= 0.0 else elapsed / duration
            blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            blend_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
            blend_ddot = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (duration * duration)

            point = JointTrajectoryPoint()
            point.positions = [
                start_positions[name] + (goal_positions[name] - start_positions[name]) * blend
                for name in self.joint_names
            ]
            point.velocities = [
                (goal_positions[name] - start_positions[name]) * blend_dot
                for name in self.joint_names
            ]
            point.accelerations = [
                (goal_positions[name] - start_positions[name]) * blend_ddot
                for name in self.joint_names
            ]
            point.time_from_start = self._duration_msg(elapsed)
            points.append(point)
        return points

    @staticmethod
    def _duration_msg(seconds: float) -> DurationMsg:
        seconds = max(0.0, float(seconds))
        whole = int(math.floor(seconds))
        nanosec = int(round((seconds - whole) * 1e9))
        if nanosec >= 1_000_000_000:
            whole += 1
            nanosec -= 1_000_000_000
        msg = DurationMsg()
        msg.sec = whole
        msg.nanosec = nanosec
        return msg

    def _make_point(self) -> JointTrajectoryPoint:
        point = JointTrajectoryPoint()
        point.positions = [math.radians(value) for value in self.home_joint_positions_deg]
        point.time_from_start = self._duration_msg(self.move_duration_sec)
        return point


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MoveHome()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
