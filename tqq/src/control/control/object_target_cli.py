import json
import sys
import threading
import time
from typing import Dict, List

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


class ObjectTargetCli(Node):
    """Small terminal UI for choosing which detected YOLO object to grab."""

    def __init__(self) -> None:
        super().__init__('object_target_cli')
        self.declare_parameter('detected_objects_topic', '/object_target_controller/detected_objects')
        self.declare_parameter('target_command_topic', '/object_target_controller/target_class_name')
        self.declare_parameter('refresh_interval_sec', 1.0)

        self.detected_objects_topic = str(self.get_parameter('detected_objects_topic').value)
        self.target_command_topic = str(self.get_parameter('target_command_topic').value)
        self.refresh_interval_sec = float(self.get_parameter('refresh_interval_sec').value)

        self.objects: List[Dict] = []
        self.target_class_name = ''
        self.last_printed = ''
        self.last_print_time = 0.0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.create_subscription(String, self.detected_objects_topic, self.detected_objects_callback, 10)
        self.target_pub = self.create_publisher(String, self.target_command_topic, 10)

        self.get_logger().info(
            f'Listening on {self.detected_objects_topic}; '
            f'publishing choices to {self.target_command_topic}'
        )

    def detected_objects_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        with self.lock:
            self.objects = list(payload.get('objects', []))
            self.target_class_name = str(payload.get('target_class_name', ''))

        self.print_objects(force=False)

    def print_objects(self, force: bool) -> None:
        now = time.monotonic()
        with self.lock:
            objects_text = self.format_objects()
            target = self.target_class_name

        line = f'I can see: {objects_text}'
        if target:
            line += f' | selected: {target}'

        if not force and line == self.last_printed:
            return
        if not force and now - self.last_print_time < self.refresh_interval_sec:
            return

        print(f'\n{line}', flush=True)
        self.last_printed = line
        self.last_print_time = now

    def format_objects(self) -> str:
        if not self.objects:
            return 'nothing yet'
        return ', '.join(
            f'{item.get("class_name", "")} x{item.get("count", 1)}'
            for item in self.objects
        )

    def object_names(self) -> List[str]:
        with self.lock:
            return [str(item.get('class_name', '')) for item in self.objects]

    def publish_choice(self, choice: str) -> None:
        self.target_pub.publish(String(data=choice))
        if choice:
            names = [name.lower() for name in self.object_names()]
            if choice.lower() in names:
                print(f'Selected {choice}. I will grab it when the next matching detection is ready.', flush=True)
            else:
                print(f'Selected {choice}. I do not see it right now, so I will wait for it.', flush=True)
        else:
            print('Selection cleared.', flush=True)

    def input_loop(self) -> None:
        print('Type an object name to grab it, for example: apple')
        print('Type "clear" to clear the target, or "q" to quit.')
        while not self.stop_event.is_set():
            try:
                choice = input('grab> ').strip()
            except EOFError:
                self.stop_event.set()
                break
            except KeyboardInterrupt:
                self.stop_event.set()
                break

            if not choice:
                continue
            if choice.lower() in ('q', 'quit', 'exit'):
                self.stop_event.set()
                break
            if choice.lower() in ('clear', 'none', 'cancel'):
                self.publish_choice('')
                continue

            self.publish_choice(choice)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectTargetCli()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.input_loop()
    finally:
        node.stop_event.set()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        sys.exit(0)


if __name__ == '__main__':
    main()
