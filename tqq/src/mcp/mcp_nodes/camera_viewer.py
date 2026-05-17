import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraViewer(Node):
    def __init__(self) -> None:
        super().__init__('camera_viewer')
        self.declare_parameter('image_topic', '/d435/d435/color/image_raw')
        self.declare_parameter('window_name', '全局相机视野')
        self.declare_parameter('window_width', 960)
        self.declare_parameter('window_height', 720)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.window_width = int(self.get_parameter('window_width').value)
        self.window_height = int(self.get_parameter('window_height').value)

        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_image_time = 0.0
        self._shutdown = threading.Event()

        self.create_subscription(Image, self.image_topic, self._image_callback, 10)
        self._ui_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._ui_thread.start()

        self.get_logger().info(
            f'Camera viewer ready: topic={self.image_topic}, window={self.window_name}'
        )

    def _image_callback(self, msg: Image) -> None:
        try:
            frame = self._ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f'Could not convert camera image: {exc}')
            return
        with self._lock:
            self._latest_image = frame
            self._latest_image_time = time.monotonic()

    def _display_loop(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            self.get_logger().error(
                'camera viewer requires python3-opencv and python3-numpy. '
                f'Details: {exc}'
            )
            return

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
            while rclpy.ok() and not self._shutdown.is_set():
                with self._lock:
                    frame = None if self._latest_image is None else self._latest_image.copy()
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    self._shutdown.set()
                    break
            cv2.destroyWindow(self.window_name)
        except Exception as exc:
            self.get_logger().error(f'camera viewer window failed: {exc}')

    @staticmethod
    def _ros_image_to_bgr(msg: Image):
        import cv2
        import numpy as np

        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding or '').lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding in ('bgr8', '8uc3'):
            return data.reshape((height, width, 3)).copy()
        if encoding == 'rgb8':
            rgb = data.reshape((height, width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if encoding in ('bgra8', 'rgba8'):
            image = data.reshape((height, width, 4))
            code = cv2.COLOR_BGRA2BGR if encoding == 'bgra8' else cv2.COLOR_RGBA2BGR
            return cv2.cvtColor(image, code)
        if encoding in ('mono8', '8uc1'):
            mono = data.reshape((height, width))
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        if encoding in ('yuyv', 'yuyv422'):
            yuyv = data.reshape((height, width, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        raise ValueError(f'unsupported image encoding: {msg.encoding}')

    def destroy_node(self) -> bool:
        self._shutdown.set()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
