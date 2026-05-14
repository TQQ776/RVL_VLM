#!/usr/bin/env python3

import sys
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class StartMoveItServo(Node):
    def __init__(self) -> None:
        super().__init__('start_moveit_servo')
        self.declare_parameter('start_service', '/servo_node/start_servo')
        self.declare_parameter('timeout_sec', 20.0)
        self.start_service = str(self.get_parameter('start_service').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)

    def run(self) -> int:
        services = [self.start_service]
        for service_name in ('/servo_node/start_servo', '/servo_node/start'):
            if service_name not in services:
                services.append(service_name)

        deadline = time.monotonic() + self.timeout_sec
        client = None
        service_name = ''
        self.get_logger().info(
            'Waiting for MoveIt Servo start service: ' + ', '.join(services)
        )
        while time.monotonic() < deadline and rclpy.ok():
            for candidate in services:
                candidate_client = self.create_client(Trigger, candidate)
                if candidate_client.wait_for_service(timeout_sec=0.5):
                    client = candidate_client
                    service_name = candidate
                    break
                self.destroy_client(candidate_client)
            if client is not None:
                break

        if client is None:
            self.get_logger().error('MoveIt Servo start service was not available.')
            return 1

        future = client.call_async(Trigger.Request())
        self.get_logger().info(f'Calling MoveIt Servo start service: {service_name}')
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=max(0.5, deadline - time.monotonic())
        )
        if future.result() is None:
            self.get_logger().error('MoveIt Servo start service call timed out or failed.')
            return 1

        response = future.result()
        if response.success:
            self.get_logger().info(f'MoveIt Servo started: {response.message}')
            return 0

        self.get_logger().error(f'MoveIt Servo failed to start: {response.message}')
        return 1


def main() -> None:
    rclpy.init()
    node = StartMoveItServo()
    exit_code = 1
    try:
        exit_code = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
