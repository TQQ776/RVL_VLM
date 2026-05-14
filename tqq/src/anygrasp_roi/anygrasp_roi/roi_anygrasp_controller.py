import copy
from dataclasses import dataclass
import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from cv_bridge import CvBridge, CvBridgeError
from franka_msgs.action import Grasp as GripperGrasp
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

import tf2_geometry_msgs  # noqa: F401  Registers geometry message transforms.


@dataclass
class RoiSelection:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class RoiAnyGraspTarget:
    roi: RoiSelection
    grasp_score: float
    grasp_width: float
    grasp_depth: float
    camera_pose: PoseStamped
    execution_camera_pose: PoseStamped
    base_pose: PoseStamped
    roi_points: np.ndarray
    roi_colors: np.ndarray
    orientation_source: str


class AnyGraspBackend:
    """Use third_party/anygrasp_sdk inference while keeping ROS + MoveIt control."""

    def __init__(self, node) -> None:
        self.node = node
        self.inferencer = None
        self._load_error = ''

    def ready(self) -> Tuple[bool, str]:
        if self.inferencer is not None:
            return True, 'ready'
        if self._load_error:
            return False, self._load_error
        try:
            self._load()
            return True, 'ready'
        except Exception as exc:
            self._load_error = str(exc)
            return False, self._load_error

    def _load(self) -> None:
        root_dir = os.path.expanduser(self.node.anygrasp_sdk_dir)
        detection_dir = os.path.join(root_dir, 'grasp_detection')
        checkpoint_path = os.path.expanduser(self.node.anygrasp_checkpoint_path)
        if not os.path.isdir(root_dir):
            raise RuntimeError(f'anygrasp_sdk_dir does not exist: {root_dir}')
        if not os.path.isdir(detection_dir):
            raise RuntimeError(
                'AnyGrasp SDK is incomplete: missing grasp_detection directory. '
                f'Current anygrasp_sdk_dir: {root_dir}'
            )
        license_dir = os.path.join(detection_dir, 'license')
        if self.node.anygrasp_require_license and not os.path.isdir(license_dir):
            raise RuntimeError(
                'AnyGrasp SDK license is missing. Apply for a license with '
                'third_party/anygrasp_sdk/license_registration, then place the '
                f'license directory at: {license_dir}'
            )
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                'AnyGrasp checkpoint is missing. Download checkpoint_detection.tar '
                'from the AnyGrasp SDK instructions and set anygrasp_checkpoint_path. '
                f'Current value: {checkpoint_path or "<empty>"}'
            )
        local_lib_dir = os.path.join(root_dir, 'lib')
        if os.path.isdir(local_lib_dir):
            existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            ld_parts = [part for part in existing_ld_path.split(':') if part]
            if local_lib_dir not in ld_parts:
                os.environ['LD_LIBRARY_PATH'] = (
                    local_lib_dir if not existing_ld_path else f'{local_lib_dir}:{existing_ld_path}'
                )
            try:
                import ctypes

                ctypes.CDLL(os.path.join(local_lib_dir, 'libcrypto.so.1.1'))
            except Exception as exc:
                self.node.get_logger().warn(
                    f'Failed to preload AnyGrasp local libcrypto.so.1.1: {exc}'
                )
        self._ensure_binary_aliases(root_dir, detection_dir)
        for path in (
            root_dir,
            detection_dir,
            os.path.join(detection_dir, 'gsnet_versions'),
            os.path.join(root_dir, 'pointnet2'),
        ):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from gsnet import AnyGrasp
        except Exception as exc:
            raise RuntimeError(
                'Failed to import AnyGrasp SDK module gsnet. Make sure '
                'third_party/anygrasp_sdk is checked out, the Python version matches '
                'one gsnet_versions/*.so file, and AnyGrasp dependencies are installed. '
                f'Original error: {exc}'
            ) from exc

        cfg = type('AnyGraspConfig', (), {})()
        cfg.checkpoint_path = checkpoint_path
        cfg.max_gripper_width = self.node.anygrasp_max_gripper_width
        cfg.gripper_height = self.node.anygrasp_gripper_height
        cfg.top_down_grasp = self.node.anygrasp_top_down_grasp
        cfg.debug = self.node.anygrasp_debug
        self.inferencer = AnyGrasp(cfg)
        self.inferencer.load_net()
        self.node._publish_status(
            f'AnyGrasp backend loaded from {detection_dir}, checkpoint={checkpoint_path}'
        )

    def _ensure_binary_aliases(self, root_dir: str, detection_dir: str) -> None:
        gsnet_so = os.path.join(detection_dir, 'gsnet.so')
        lib_cxx_so = os.path.join(detection_dir, 'lib_cxx.so')
        if os.path.isfile(gsnet_so) and os.path.isfile(lib_cxx_so):
            return
        py_tag = f'cpython-{sys.version_info.major}{sys.version_info.minor}'
        gsnet_version = os.path.join(
            detection_dir,
            'gsnet_versions',
            f'gsnet.{py_tag}-x86_64-linux-gnu.so',
        )
        lib_cxx_version = os.path.join(
            root_dir,
            'license_registration',
            'lib_cxx_versions',
            f'lib_cxx.{py_tag}-x86_64-linux-gnu.so',
        )
        missing = []
        if not os.path.isfile(gsnet_so):
            missing.append(f'{gsnet_so} (copy from {gsnet_version})')
        if not os.path.isfile(lib_cxx_so):
            missing.append(f'{lib_cxx_so} (copy from {lib_cxx_version})')
        if missing:
            raise RuntimeError(
                'AnyGrasp binary aliases are missing for this Python version. '
                'Copy the matching SDK binaries before running:\n'
                + '\n'.join(f'- {item}' for item in missing)
            )

    def infer(self, points: np.ndarray, colors: np.ndarray):
        ok, message = self.ready()
        if not ok:
            raise RuntimeError(message)
        if points.shape[0] < self.node.anygrasp_min_points:
            raise RuntimeError(
                f'not enough ROI point-cloud points for AnyGrasp: '
                f'{points.shape[0]} < {self.node.anygrasp_min_points}'
            )

        points = points.astype(np.float32, copy=False)
        colors = np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False)
        lims = self.node._anygrasp_lims(points)
        gg, _cloud = self.inferencer.get_grasp(
            points.astype(np.float32, copy=False),
            colors,
            lims=lims,
            apply_object_mask=self.node.anygrasp_apply_object_mask,
            dense_grasp=self.node.anygrasp_dense_grasp,
            collision_detection=self.node.anygrasp_collision_detection,
        )
        if len(gg) == 0:
            raise RuntimeError('AnyGrasp produced no valid grasp')
        gg = gg.nms().sort_by_score()
        if len(gg) == 0:
            raise RuntimeError('AnyGrasp produced no valid grasp after NMS')
        return gg[0]

class RoiAnyGraspController(Node):
    """Select an ROI in the camera image, estimate an AnyGrasp 6D grasp, then execute it."""

    def __init__(self) -> None:
        super().__init__('roi_anygrasp_controller')
        self.callback_group = ReentrantCallbackGroup()

        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.backend = AnyGraspBackend(self)

        self.latest_color: Optional[np.ndarray] = None
        self.latest_color_header = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_header = None
        self.latest_depth_encoding = ''
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_joint_state: Optional[JointState] = None

        self.state_lock = threading.Lock()
        self.ui_lock = threading.Lock()
        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.selected_roi: Optional[RoiSelection] = None
        self.worker_running = False
        self.shutdown_event = threading.Event()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image,
            self.color_topic,
            self.color_callback,
            image_qos,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            image_qos,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            image_qos,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.gripper_6d_pose_camera_pub = self.create_publisher(
            PoseStamped,
            '~/gripper_6d_pose_camera',
            10,
        )
        self.gripper_6d_pose_base_pub = self.create_publisher(
            PoseStamped,
            '~/gripper_6d_pose_base',
            10,
        )
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_group_action,
            callback_group=self.callback_group,
        )
        self.ik_client = self.create_client(
            GetPositionIK,
            self.ik_service,
            callback_group=self.callback_group,
        )
        self.gripper_grasp_clients = [
            (
                action_name,
                ActionClient(
                    self,
                    GripperGrasp,
                    action_name,
                    callback_group=self.callback_group,
                ),
            )
            for action_name in self.gripper_grasp_actions
        ]

        self.ui_thread = threading.Thread(target=self._ui_loop, daemon=True)
        self.ui_thread.start()

        self.get_logger().info(
            'ROI AnyGrasp controller ready. '
            f'color={self.color_topic}, depth={self.depth_topic}, '
            f'base_frame={self.base_frame}, ee={self.end_effector_frame}'
        )
        self._publish_status(
            'ROI window ready: drag a rectangle, press Enter or g to run AnyGrasp, '
            'press c to clear, q to close the window.'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('use_latest_tf', True)
        self.declare_parameter('end_effector_frame', 'fr3_hand_tcp')
        self.declare_parameter('move_group_name', 'fr3_arm')
        self.declare_parameter('ik_service', '/compute_ik')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('plan_only', False)
        self.declare_parameter('avoid_collisions', True)
        self.declare_parameter('ik_timeout_sec', 0.5)
        self.declare_parameter('service_wait_timeout_sec', 5.0)
        self.declare_parameter('action_wait_timeout_sec', 10.0)
        self.declare_parameter('move_group_result_timeout_sec', 90.0)
        self.declare_parameter('goal_joint_tolerance', 0.01)
        self.declare_parameter('max_velocity_scaling', 0.01)
        self.declare_parameter('max_acceleration_scaling', 0.01)
        self.declare_parameter('num_planning_attempts', 5)
        self.declare_parameter('allowed_planning_time', 5.0)
        self.declare_parameter('roi_window_name', 'AnyGrasp ROI Selector')
        self.declare_parameter('roi_padding_px', 8)
        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('anygrasp_sdk_dir', '/home/tqq/TQQ_ws/third_party/anygrasp_sdk')
        self.declare_parameter(
            'anygrasp_checkpoint_path',
            '/home/tqq/TQQ_ws/third_party/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar',
        )
        self.declare_parameter('anygrasp_require_license', True)
        self.declare_parameter('anygrasp_min_points', 200)
        self.declare_parameter('anygrasp_max_gripper_width', 0.08)
        self.declare_parameter('anygrasp_gripper_height', 0.03)
        self.declare_parameter('anygrasp_top_down_grasp', False)
        self.declare_parameter('anygrasp_debug', False)
        self.declare_parameter('anygrasp_apply_object_mask', True)
        self.declare_parameter('anygrasp_dense_grasp', False)
        self.declare_parameter('anygrasp_collision_detection', True)
        self.declare_parameter('anygrasp_workspace_margin_m', 0.02)
        self.declare_parameter('anygrasp_orientation_mode', 'anygrasp')
        self.declare_parameter('anygrasp_fallback_to_current_orientation', False)
        self.declare_parameter('anygrasp_tcp_rotation_rpy_grasp', [0.0, 1.57079632679, 0.0])
        self.declare_parameter('anygrasp_tcp_offset_xyz_grasp', [0.0, 0.0, 0.0])
        self.declare_parameter('target_offset_xyz_base', [0.0, 0.0, 0.0])
        self.declare_parameter('min_grasp_z_m', 0.05)
        self.declare_parameter('popup_preview_before_execute', True)
        self.declare_parameter('popup_preview_max_points', 20000)
        self.declare_parameter('popup_preview_frame_size_m', 0.08)
        self.declare_parameter('popup_preview_window_title', 'AnyGrasp ROI grasp preview')
        self.declare_parameter('gripper_grasp_actions', [
            '/franka_gripper/grasp',
            '/fr3_gripper/grasp',
            '/left_fr3_gripper/grasp',
        ])
        self.declare_parameter('gripper_server_wait_timeout_sec', 1.0)
        self.declare_parameter('gripper_action_timeout_sec', 10.0)
        self.declare_parameter('gripper_close_width_m', 0.0)
        self.declare_parameter('gripper_close_speed_mps', 0.03)
        self.declare_parameter('gripper_close_force_n', 50.0)
        self.declare_parameter('gripper_grasp_epsilon_inner_m', 0.01)
        self.declare_parameter('gripper_grasp_epsilon_outer_m', 0.08)
        self.declare_parameter('arm_joints', [
            'fr3_joint1',
            'fr3_joint2',
            'fr3_joint3',
            'fr3_joint4',
            'fr3_joint5',
            'fr3_joint6',
            'fr3_joint7',
        ])

    def _read_parameters(self) -> None:
        self.color_topic = str(self.get_parameter('color_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.camera_frame_override = str(self.get_parameter('camera_frame').value)
        self.use_latest_tf = bool(self.get_parameter('use_latest_tf').value)
        self.end_effector_frame = str(self.get_parameter('end_effector_frame').value)
        self.move_group_name = str(self.get_parameter('move_group_name').value)
        self.ik_service = str(self.get_parameter('ik_service').value)
        self.move_group_action = str(self.get_parameter('move_group_action').value)
        self.plan_only = bool(self.get_parameter('plan_only').value)
        self.avoid_collisions = bool(self.get_parameter('avoid_collisions').value)
        self.ik_timeout_sec = float(self.get_parameter('ik_timeout_sec').value)
        self.service_wait_timeout_sec = float(self.get_parameter('service_wait_timeout_sec').value)
        self.action_wait_timeout_sec = float(self.get_parameter('action_wait_timeout_sec').value)
        self.move_group_result_timeout_sec = float(
            self.get_parameter('move_group_result_timeout_sec').value
        )
        self.goal_joint_tolerance = float(self.get_parameter('goal_joint_tolerance').value)
        self.max_velocity_scaling = float(self.get_parameter('max_velocity_scaling').value)
        self.max_acceleration_scaling = float(self.get_parameter('max_acceleration_scaling').value)
        self.num_planning_attempts = int(self.get_parameter('num_planning_attempts').value)
        self.allowed_planning_time = float(self.get_parameter('allowed_planning_time').value)
        self.roi_window_name = str(self.get_parameter('roi_window_name').value)
        self.roi_padding_px = max(0, int(self.get_parameter('roi_padding_px').value))
        self.depth_unit_scale = float(self.get_parameter('depth_unit_scale').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.anygrasp_sdk_dir = str(self.get_parameter('anygrasp_sdk_dir').value)
        self.anygrasp_checkpoint_path = str(self.get_parameter('anygrasp_checkpoint_path').value)
        self.anygrasp_require_license = bool(self.get_parameter('anygrasp_require_license').value)
        self.anygrasp_min_points = int(self.get_parameter('anygrasp_min_points').value)
        self.anygrasp_max_gripper_width = float(
            self.get_parameter('anygrasp_max_gripper_width').value
        )
        self.anygrasp_gripper_height = float(self.get_parameter('anygrasp_gripper_height').value)
        self.anygrasp_top_down_grasp = bool(self.get_parameter('anygrasp_top_down_grasp').value)
        self.anygrasp_debug = bool(self.get_parameter('anygrasp_debug').value)
        self.anygrasp_apply_object_mask = bool(
            self.get_parameter('anygrasp_apply_object_mask').value
        )
        self.anygrasp_dense_grasp = bool(self.get_parameter('anygrasp_dense_grasp').value)
        self.anygrasp_collision_detection = bool(
            self.get_parameter('anygrasp_collision_detection').value
        )
        self.anygrasp_workspace_margin_m = float(
            self.get_parameter('anygrasp_workspace_margin_m').value
        )
        self.anygrasp_orientation_mode = str(
            self.get_parameter('anygrasp_orientation_mode').value
        ).strip().lower()
        self.anygrasp_fallback_to_current_orientation = bool(
            self.get_parameter('anygrasp_fallback_to_current_orientation').value
        )
        self.anygrasp_tcp_rotation_rpy_grasp = self._float_list(
            self.get_parameter('anygrasp_tcp_rotation_rpy_grasp').value,
            3,
            'anygrasp_tcp_rotation_rpy_grasp',
        )
        self.anygrasp_tcp_rotation_matrix = self._rpy_to_rotation_matrix(
            self.anygrasp_tcp_rotation_rpy_grasp
        )
        self.anygrasp_tcp_offset_xyz_grasp = self._float_list(
            self.get_parameter('anygrasp_tcp_offset_xyz_grasp').value,
            3,
            'anygrasp_tcp_offset_xyz_grasp',
        )
        self.target_offset_xyz_base = self._float_list(
            self.get_parameter('target_offset_xyz_base').value,
            3,
            'target_offset_xyz_base',
        )
        self.min_grasp_z_m = float(self.get_parameter('min_grasp_z_m').value)
        self.popup_preview_before_execute = bool(
            self.get_parameter('popup_preview_before_execute').value
        )
        self.popup_preview_max_points = max(
            1,
            int(self.get_parameter('popup_preview_max_points').value),
        )
        self.popup_preview_frame_size_m = float(
            self.get_parameter('popup_preview_frame_size_m').value
        )
        self.popup_preview_window_title = str(
            self.get_parameter('popup_preview_window_title').value
        )
        self.gripper_grasp_actions = self._string_list(
            self.get_parameter('gripper_grasp_actions').value
        )
        self.gripper_server_wait_timeout_sec = float(
            self.get_parameter('gripper_server_wait_timeout_sec').value
        )
        self.gripper_action_timeout_sec = float(
            self.get_parameter('gripper_action_timeout_sec').value
        )
        self.gripper_close_width_m = float(self.get_parameter('gripper_close_width_m').value)
        self.gripper_close_speed_mps = float(self.get_parameter('gripper_close_speed_mps').value)
        self.gripper_close_force_n = float(self.get_parameter('gripper_close_force_n').value)
        self.gripper_grasp_epsilon_inner_m = float(
            self.get_parameter('gripper_grasp_epsilon_inner_m').value
        )
        self.gripper_grasp_epsilon_outer_m = float(
            self.get_parameter('gripper_grasp_epsilon_outer_m').value
        )
        self.arm_joints = [str(name) for name in self.get_parameter('arm_joints').value]
        if self.anygrasp_orientation_mode not in ('anygrasp', 'current'):
            raise ValueError('anygrasp_orientation_mode must be anygrasp or current')

    @staticmethod
    def _float_list(value: Sequence[float], expected_len: int, name: str) -> List[float]:
        result = [float(item) for item in value]
        if len(result) != expected_len:
            raise ValueError(f'{name} must contain {expected_len} values')
        return result

    @staticmethod
    def _string_list(value) -> List[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return [str(item).strip() for item in value or [] if str(item).strip()]

    def color_callback(self, msg: Image) -> None:
        try:
            color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as exc:
            self.get_logger().error(f'Failed to convert color image: {exc}')
            return
        with self.state_lock:
            self.latest_color = np.asarray(color)
            self.latest_color_header = copy.deepcopy(msg.header)

    def depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            self.get_logger().error(f'Failed to convert depth image: {exc}')
            return
        with self.state_lock:
            self.latest_depth = np.asarray(depth)
            self.latest_depth_header = copy.deepcopy(msg.header)
            self.latest_depth_encoding = msg.encoding

    def camera_info_callback(self, msg: CameraInfo) -> None:
        with self.state_lock:
            self.latest_camera_info = copy.deepcopy(msg)

    def joint_state_callback(self, msg: JointState) -> None:
        with self.state_lock:
            self.latest_joint_state = copy.deepcopy(msg)

    def _ui_loop(self) -> None:
        try:
            cv2.namedWindow(self.roi_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.roi_window_name, 960, 720)
            cv2.setMouseCallback(self.roi_window_name, self._mouse_callback)
        except cv2.error as exc:
            self.get_logger().error(f'Cannot open ROI window: {exc}')
            return

        while rclpy.ok() and not self.shutdown_event.is_set():
            with self.state_lock:
                color = None if self.latest_color is None else self.latest_color.copy()
            if color is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            display = self._draw_roi_overlay(frame)
            cv2.imshow(self.roi_window_name, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, ord('g'), ord('G')):
                self._start_selected_roi()
            elif key in (ord('c'), ord('C')):
                with self.ui_lock:
                    self.selected_roi = None
                    self.dragging = False
                    self.drag_start = None
                    self.drag_current = None
                self._publish_status('ROI selection cleared.')
            elif key in (ord('q'), ord('Q'), 27):
                self._publish_status('ROI window closed.')
                break
        cv2.destroyWindow(self.roi_window_name)

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        del flags, param
        with self.ui_lock:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.drag_start = (int(x), int(y))
                self.drag_current = (int(x), int(y))
                self.selected_roi = None
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                self.drag_current = (int(x), int(y))
            elif event == cv2.EVENT_LBUTTONUP and self.dragging:
                self.dragging = False
                if self.drag_start is None:
                    return
                self.drag_current = (int(x), int(y))
                roi = self._normalize_roi(self.drag_start, self.drag_current)
                if roi is not None:
                    self.selected_roi = roi
                    self._publish_status(
                        f'ROI selected: x={roi.x0}:{roi.x1}, y={roi.y0}:{roi.y1}. '
                        'Press Enter or g to run AnyGrasp.'
                    )

    def _draw_roi_overlay(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        with self.ui_lock:
            selected = copy.deepcopy(self.selected_roi)
            dragging = self.dragging
            drag_start = self.drag_start
            drag_current = self.drag_current
            worker_running = self.worker_running
        roi = selected
        color = (0, 255, 255)
        if dragging and drag_start is not None and drag_current is not None:
            roi = self._normalize_roi(drag_start, drag_current)
            color = (255, 200, 0)
        if roi is not None:
            cv2.rectangle(display, (roi.x0, roi.y0), (roi.x1, roi.y1), color, 2)
        status = 'running AnyGrasp...' if worker_running else 'drag ROI, Enter/g run, c clear, q close'
        cv2.putText(
            display,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if worker_running else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    @staticmethod
    def _normalize_roi(
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Optional[RoiSelection]:
        x0 = min(start[0], end[0])
        y0 = min(start[1], end[1])
        x1 = max(start[0], end[0])
        y1 = max(start[1], end[1])
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return RoiSelection(x0=x0, y0=y0, x1=x1, y1=y1)

    def _start_selected_roi(self) -> None:
        with self.ui_lock:
            if self.worker_running:
                self._publish_status('AnyGrasp ROI worker is already running.')
                return
            roi = copy.deepcopy(self.selected_roi)
            if roi is None:
                self._publish_status('No ROI selected yet.')
                return
            self.worker_running = True
        thread = threading.Thread(target=self._roi_worker, args=(roi,), daemon=True)
        thread.start()

    def _roi_worker(self, roi: RoiSelection) -> None:
        try:
            target = self._build_grasp_target(roi)
            if target is None:
                return
            self.gripper_6d_pose_camera_pub.publish(target.execution_camera_pose)
            self.gripper_6d_pose_base_pub.publish(target.base_pose)
            self._publish_status(
                f'ROI AnyGrasp grasp_score={target.grasp_score:.3f} '
                f'width={target.grasp_width:.3f}m '
                f'base=({target.base_pose.pose.position.x:.3f}, '
                f'{target.base_pose.pose.position.y:.3f}, {target.base_pose.pose.position.z:.3f})'
            )
            self._publish_gripper_6d_status(target)
            if not self._preview_and_confirm(target):
                self._publish_status('ROI grasp canceled before motion; gripper will not close.')
                return
            moved = self._execute_pose_motion(target.base_pose, 'roi anygrasp target')
            if moved:
                self._close_gripper_for_grasp()
            else:
                self._publish_status('ROI grasp motion did not report success; gripper will not close.')
        finally:
            with self.ui_lock:
                self.worker_running = False

    def _build_grasp_target(self, roi: RoiSelection) -> Optional[RoiAnyGraspTarget]:
        snapshot = self._snapshot_rgbd()
        if snapshot is None:
            return None
        color, depth, depth_encoding, depth_header, camera_info = snapshot
        points, colors = self._roi_point_cloud(color, depth, depth_encoding, camera_info, roi)
        if points is None or colors is None:
            return None
        try:
            grasp = self.backend.infer(points, colors)
        except Exception as exc:
            self.get_logger().error(f'AnyGrasp inference failed: {exc}')
            self._publish_status(f'AnyGrasp inference failed: {exc}')
            return None
        camera_pose = self._grasp_to_camera_pose(grasp, depth_header)
        if camera_pose is None:
            return None
        try:
            base_pose = self.tf_buffer.transform(
                camera_pose,
                self.base_frame,
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().error(
                f'Cannot transform AnyGrasp pose {camera_pose.header.frame_id} -> {self.base_frame}: {exc}'
            )
            return None
        base_pose.pose.position.x += self.target_offset_xyz_base[0]
        base_pose.pose.position.y += self.target_offset_xyz_base[1]
        base_pose.pose.position.z = max(
            self.min_grasp_z_m,
            base_pose.pose.position.z + self.target_offset_xyz_base[2],
        )
        orientation_source = 'anygrasp'
        if self.anygrasp_orientation_mode == 'current':
            current_orientation = self._current_end_effector_orientation()
            if current_orientation is None:
                return None
            base_pose.pose.orientation = current_orientation
            orientation_source = 'current_end_effector'
        execution_camera_pose = self._transform_pose_or_copy(
            base_pose,
            camera_pose.header.frame_id,
            camera_pose,
        )
        return RoiAnyGraspTarget(
            roi=roi,
            grasp_score=float(grasp.score),
            grasp_width=float(grasp.width),
            grasp_depth=float(grasp.depth),
            camera_pose=camera_pose,
            execution_camera_pose=execution_camera_pose,
            base_pose=base_pose,
            roi_points=points.astype(np.float32, copy=False),
            roi_colors=colors.astype(np.float32, copy=False),
            orientation_source=orientation_source,
        )

    def _preview_and_confirm(self, target: RoiAnyGraspTarget) -> bool:
        if self.popup_preview_before_execute:
            try:
                confirmed = self._show_open3d_preview(target)
            except Exception as exc:
                message = f'Failed to open AnyGrasp preview window: {exc}'
                self.get_logger().error(message)
                self._publish_status(message)
                return False
            if not confirmed:
                self._publish_status('AnyGrasp ROI preview canceled by operator.')
                return False
            self._publish_status('AnyGrasp ROI preview confirmed by operator.')
            return True
        self._publish_status('AnyGrasp ROI preview confirmed by operator.')
        return True

    def _show_open3d_preview(self, target: RoiAnyGraspTarget) -> bool:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                'open3d is not installed in this Python environment. '
                'Install it or run the ROS node with /usr/bin/python3.'
            ) from exc

        points = target.roi_points
        colors = target.roi_colors
        if len(points) > self.popup_preview_max_points:
            indexes = np.linspace(
                0,
                len(points) - 1,
                self.popup_preview_max_points,
                dtype=np.int64,
            )
            points = points[indexes]
            colors = colors[indexes]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))

        frame_size = max(0.01, self.popup_preview_frame_size_m)
        preview_pose = target.execution_camera_pose
        pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size,
            origin=[
                preview_pose.pose.position.x,
                preview_pose.pose.position.y,
                preview_pose.pose.position.z,
            ],
        )
        pose_frame.rotate(
            self._quaternion_to_rotation_matrix(preview_pose.pose.orientation),
            center=[
                preview_pose.pose.position.x,
                preview_pose.pose.position.y,
                preview_pose.pose.position.z,
            ],
        )
        gripper_geometry = self._create_open3d_gripper_geometry(o3d, preview_pose, target)

        title = (
            f'{self.popup_preview_window_title} - '
            f'score={target.grasp_score:.3f} width={target.grasp_width:.3f}m '
            f'pose={target.orientation_source}'
        )
        self._publish_status(
            'Showing ROI AnyGrasp popup preview. Focus the Open3D window: '
            'press Enter to execute, or press c/q/Esc to cancel.'
        )
        decision = {'confirmed': False}

        def confirm(vis):
            decision['confirmed'] = True
            vis.close()
            return False

        def cancel(vis):
            decision['confirmed'] = False
            vis.close()
            return False

        o3d.visualization.draw_geometries_with_key_callbacks(
            [cloud, pose_frame, gripper_geometry],
            {
                13: confirm,
                257: confirm,
                ord('C'): cancel,
                ord('Q'): cancel,
                256: cancel,
            },
            title,
            1280,
            720,
        )
        return decision['confirmed']

    def _create_open3d_gripper_geometry(self, o3d, pose: PoseStamped, target: RoiAnyGraspTarget):
        width = target.grasp_width if math.isfinite(target.grasp_width) else 0.06
        width = min(0.12, max(0.02, float(width)))
        depth = target.grasp_depth if math.isfinite(target.grasp_depth) else 0.06
        finger_length = min(0.12, max(0.04, float(depth)))
        palm_depth = 0.025
        wrist_length = 0.045
        finger_height = 0.018

        local_points = np.asarray([
            [0.0, -width / 2.0, 0.0],
            [0.0, width / 2.0, 0.0],
            [finger_length, -width / 2.0, 0.0],
            [finger_length, width / 2.0, 0.0],
            [0.0, -width / 2.0, finger_height],
            [0.0, width / 2.0, finger_height],
            [finger_length, -width / 2.0, finger_height],
            [finger_length, width / 2.0, finger_height],
            [-palm_depth, -width / 2.0, 0.0],
            [-palm_depth, width / 2.0, 0.0],
            [-palm_depth, 0.0, 0.0],
            [-palm_depth - wrist_length, 0.0, 0.0],
        ], dtype=np.float64)
        lines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [8, 9],
            [8, 0],
            [9, 1],
            [10, 11],
        ]
        colors = [[0.0, 1.0, 1.0] for _ in lines]
        rotation = self._quaternion_to_rotation_matrix(pose.pose.orientation)
        translation = np.asarray([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ], dtype=np.float64)
        points = local_points @ rotation.T + translation

        geometry = o3d.geometry.LineSet()
        geometry.points = o3d.utility.Vector3dVector(points)
        geometry.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        geometry.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        return geometry

    def _snapshot_rgbd(self):
        with self.state_lock:
            color = None if self.latest_color is None else self.latest_color.copy()
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            depth_header = copy.deepcopy(self.latest_depth_header)
            depth_encoding = self.latest_depth_encoding
            camera_info = copy.deepcopy(self.latest_camera_info)
        if color is None:
            self.get_logger().warn('No color image received yet.')
            return None
        if depth is None or depth_header is None:
            self.get_logger().warn('No aligned depth image received yet.')
            return None
        if camera_info is None:
            self.get_logger().warn('No camera info received yet.')
            return None
        return color, depth, depth_encoding, depth_header, camera_info

    def _roi_point_cloud(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        depth_encoding: str,
        camera_info: CameraInfo,
        roi: RoiSelection,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        height, width = depth.shape[:2]
        pad = self.roi_padding_px
        x0 = max(0, int(roi.x0) - pad)
        x1 = min(width, int(roi.x1) + pad)
        y0 = max(0, int(roi.y0) - pad)
        y1 = min(height, int(roi.y1) + pad)
        if x1 <= x0 or y1 <= y0:
            self.get_logger().warn(f'Invalid ROI: {(roi.x0, roi.y0, roi.x1, roi.y1)}')
            return None, None

        depth_m = depth.astype(np.float32)
        if self._depth_is_integer_millimeters(depth_encoding, depth.dtype):
            depth_m *= self.depth_unit_scale
        valid_depth = np.isfinite(depth_m)
        valid_depth &= depth_m >= self.min_depth_m
        valid_depth &= depth_m <= self.max_depth_m
        mask = np.zeros(depth_m.shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = True
        mask &= valid_depth
        ys, xs = np.nonzero(mask)
        if len(xs) < self.anygrasp_min_points:
            self.get_logger().warn(
                f'ROI point cloud too small for AnyGrasp: {len(xs)} points in roi {(x0, y0, x1, y1)}'
            )
            return None, None

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        zs = depth_m[ys, xs]
        points = np.stack([
            (xs.astype(np.float32) - cx) * zs / fx,
            (ys.astype(np.float32) - cy) * zs / fy,
            zs,
        ], axis=1).astype(np.float32)
        colors = color[ys, xs].astype(np.float32) / 255.0
        return points, colors

    def _anygrasp_lims(self, points: np.ndarray) -> List[float]:
        margin = max(0.0, float(self.anygrasp_workspace_margin_m))
        mins = np.nanmin(points, axis=0)
        maxs = np.nanmax(points, axis=0)
        return [
            float(mins[0] - margin),
            float(maxs[0] + margin),
            float(mins[1] - margin),
            float(maxs[1] + margin),
            float(max(self.min_depth_m, mins[2] - margin)),
            float(min(self.max_depth_m, maxs[2] + margin)),
        ]

    @staticmethod
    def _depth_is_integer_millimeters(encoding: str, dtype: np.dtype) -> bool:
        if encoding.upper() in ('16UC1', 'MONO16'):
            return True
        return np.issubdtype(dtype, np.integer)

    def _grasp_to_camera_pose(self, grasp, depth_header) -> Optional[PoseStamped]:
        pose = PoseStamped()
        pose.header = copy.deepcopy(depth_header)
        pose.header.frame_id = (
            self.camera_frame_override
            or depth_header.frame_id
            or pose.header.frame_id
        )
        if self.use_latest_tf:
            pose.header.stamp = Time().to_msg()

        translation = np.asarray(grasp.translation, dtype=np.float64).reshape(3)
        grasp_rotation = np.asarray(grasp.rotation_matrix, dtype=np.float64).reshape(3, 3)
        tcp_rotation = grasp_rotation.dot(self.anygrasp_tcp_rotation_matrix)
        offset = np.asarray(self.anygrasp_tcp_offset_xyz_grasp, dtype=np.float64).reshape(3)
        translation = translation + grasp_rotation.dot(offset)
        pose.pose.position.x = float(translation[0])
        pose.pose.position.y = float(translation[1])
        pose.pose.position.z = float(translation[2])
        pose.pose.orientation = self._rotation_matrix_to_quaternion(tcp_rotation)
        return pose

    def _current_end_effector_orientation(self) -> Optional[Quaternion]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                Time(),
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warn(f'Cannot get current end-effector orientation: {exc}')
            return None
        return copy.deepcopy(transform.transform.rotation)

    def _transform_pose_or_copy(
        self,
        pose: PoseStamped,
        target_frame: str,
        fallback_pose: PoseStamped,
    ) -> PoseStamped:
        try:
            return self.tf_buffer.transform(
                pose,
                target_frame,
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Cannot transform execution gripper pose {pose.header.frame_id} -> '
                f'{target_frame}; using raw AnyGrasp camera pose for preview/log: {exc}'
            )
            return copy.deepcopy(fallback_pose)

    def _execute_pose_motion(self, target_pose: PoseStamped, label: str) -> bool:
        motion_label = label
        joint_goal = self._compute_ik(target_pose, motion_label)
        if (
            joint_goal is None
            and self.anygrasp_fallback_to_current_orientation
            and self.anygrasp_orientation_mode == 'anygrasp'
        ):
            fallback_pose = copy.deepcopy(target_pose)
            current_orientation = self._current_end_effector_orientation()
            if current_orientation is not None:
                fallback_pose.pose.orientation = current_orientation
                fallback_label = f'{label} with current end-effector orientation'
                self._publish_status(
                    f'IK failed for {label}; retrying at the same position with current end-effector orientation.'
                )
                joint_goal = self._compute_ik(fallback_pose, fallback_label)
                if joint_goal is not None:
                    motion_label = fallback_label
        if joint_goal is None:
            return False
        return self._execute_with_move_group(joint_goal, motion_label)

    def _compute_ik(self, target_pose: PoseStamped, label: str) -> Optional[Dict[str, float]]:
        if not self.ik_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().error(f'IK service not available: {self.ik_service}')
            return None
        with self.state_lock:
            joint_state = copy.deepcopy(self.latest_joint_state)
        if joint_state is None:
            self.get_logger().error('No /joint_states received; cannot seed MoveIt IK.')
            return None

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.move_group_name
        request.ik_request.robot_state.joint_state = joint_state
        request.ik_request.robot_state.is_diff = True
        request.ik_request.avoid_collisions = self.avoid_collisions
        request.ik_request.ik_link_name = self.end_effector_frame
        request.ik_request.pose_stamped = target_pose
        request.ik_request.timeout = self._duration_msg(self.ik_timeout_sec)
        future = self.ik_client.call_async(request)
        response = self._wait_for_future(
            future,
            self.service_wait_timeout_sec + self.ik_timeout_sec + 1.0,
            'IK response',
        )
        if response is None:
            return None
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'MoveIt IK failed for {label} with code {response.error_code.val}'
            )
            return None
        positions = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        missing = [name for name in self.arm_joints if name not in positions]
        if missing:
            self.get_logger().error(f'IK solution missing joints: {missing}')
            return None
        joint_goal = {name: float(positions[name]) for name in self.arm_joints}
        self._publish_status(f'IK solved: {self._format_joint_goal(joint_goal)}')
        return joint_goal

    def _execute_with_move_group(self, joint_goal: Dict[str, float], label: str) -> bool:
        if not self.move_group_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            self.get_logger().error(f'MoveGroup action not available: {self.move_group_action}')
            return False
        with self.state_lock:
            joint_state = copy.deepcopy(self.latest_joint_state)
        goal = MoveGroup.Goal()
        goal.request.group_name = self.move_group_name
        goal.request.num_planning_attempts = self.num_planning_attempts
        goal.request.allowed_planning_time = self.allowed_planning_time
        goal.request.max_velocity_scaling_factor = self.max_velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.max_acceleration_scaling
        goal.request.start_state.joint_state = joint_state
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(self._joint_goal_constraints(joint_goal))
        goal.planning_options.plan_only = self.plan_only
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.replan_delay = 0.5
        goal.planning_options.planning_scene_diff.is_diff = True
        self._publish_status(
            f'MoveGroup request for {label}: velocity_scaling={self.max_velocity_scaling:.3f}, '
            f'acceleration_scaling={self.max_acceleration_scaling:.3f}, '
            f'result_timeout={self.move_group_result_timeout_sec:.1f}s'
        )
        send_future = self.move_group_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, 'MoveGroup goal')
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('MoveGroup goal failed or was rejected.')
            return False
        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.move_group_result_timeout_sec,
            'MoveGroup result',
        )
        if action_result is None:
            if self._joint_goal_reached(joint_goal):
                self._publish_status(
                    f'MoveGroup result timed out for {label}, but joint state is within tolerance.'
                )
                return True
            return False
        result = action_result.result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            mode = 'planned' if self.plan_only else 'planned and executed'
            self._publish_status(f'MoveGroup {mode} {label} successfully.')
            return True
        self.get_logger().error(f'MoveGroup failed for {label} with code {result.error_code.val}')
        return False

    def _close_gripper_for_grasp(self) -> bool:
        client_name, client = self._available_gripper_grasp_client()
        if client is None:
            self.get_logger().error('No gripper grasp action available.')
            return False
        goal = GripperGrasp.Goal()
        goal.width = self.gripper_close_width_m
        goal.speed = self.gripper_close_speed_mps
        goal.force = self.gripper_close_force_n
        goal.epsilon.inner = self.gripper_grasp_epsilon_inner_m
        goal.epsilon.outer = self.gripper_grasp_epsilon_outer_m
        self._publish_status(
            f'closing gripper via {client_name}: width={goal.width:.3f}m '
            f'speed={goal.speed:.3f}m/s force={goal.force:.1f}N'
        )
        send_future = client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.gripper_action_timeout_sec,
            'gripper grasp goal',
        )
        if goal_handle is None or not goal_handle.accepted:
            return False
        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.gripper_action_timeout_sec,
            'gripper grasp result',
        )
        if action_result is None:
            return False
        result = action_result.result
        if result.success:
            self._publish_status('gripper closed successfully.')
            return True
        self.get_logger().error(f'Gripper grasp failed: {result.error}')
        return False

    def _available_gripper_grasp_client(self):
        for action_name, client in self.gripper_grasp_clients:
            if client.wait_for_server(timeout_sec=self.gripper_server_wait_timeout_sec):
                return action_name, client
        return '', None

    def _joint_goal_reached(self, joint_goal: Dict[str, float]) -> bool:
        with self.state_lock:
            joint_state = copy.deepcopy(self.latest_joint_state)
        if joint_state is None:
            return False
        positions = dict(zip(joint_state.name, joint_state.position))
        missing = [name for name in self.arm_joints if name not in positions]
        if missing:
            return False
        tolerance = max(0.001, self.goal_joint_tolerance)
        return all(abs(float(positions[name]) - float(joint_goal[name])) <= tolerance for name in self.arm_joints)

    def _joint_goal_constraints(self, joint_goal: Dict[str, float]) -> Constraints:
        constraints = Constraints()
        constraints.name = 'roi_anygrasp_ik_joint_goal'
        for name in self.arm_joints:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = joint_goal[name]
            joint_constraint.tolerance_above = self.goal_joint_tolerance
            joint_constraint.tolerance_below = self.goal_joint_tolerance
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        return constraints

    def _publish_gripper_6d_status(self, target: RoiAnyGraspTarget) -> None:
        self._publish_status(
            'ROI AnyGrasp gripper 6D pose '
            f'orientation_source={target.orientation_source}; '
            f'camera[{self._format_pose_6d(target.execution_camera_pose)}]; '
            f'base[{self._format_pose_6d(target.base_pose)}]'
        )

    def _format_pose_6d(self, pose: PoseStamped) -> str:
        roll, pitch, yaw = self._quaternion_to_rpy(pose.pose.orientation)
        return (
            f'frame={pose.header.frame_id}, '
            f'xyz=({pose.pose.position.x:.3f}, '
            f'{pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})m, '
            f'rpy=({math.degrees(roll):.1f}, '
            f'{math.degrees(pitch):.1f}, {math.degrees(yaw):.1f})deg, '
            f'quat=({pose.pose.orientation.x:.3f}, '
            f'{pose.pose.orientation.y:.3f}, {pose.pose.orientation.z:.3f}, '
            f'{pose.pose.orientation.w:.3f})'
        )

    @staticmethod
    def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> Quaternion:
        m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        return Quaternion(x=qx / norm, y=qy / norm, z=qz / norm, w=qw / norm)

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion: Quaternion) -> np.ndarray:
        x = float(quaternion.x)
        y = float(quaternion.y)
        z = float(quaternion.z)
        w = float(quaternion.w)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 0.0:
            return np.eye(3, dtype=np.float64)
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        return np.asarray([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def _rpy_to_rotation_matrix(rpy: Sequence[float]) -> np.ndarray:
        roll, pitch, yaw = [float(value) for value in rpy]
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        return np.asarray([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)

    @staticmethod
    def _quaternion_to_rpy(quaternion: Quaternion) -> Tuple[float, float, float]:
        x = float(quaternion.x)
        y = float(quaternion.y)
        z = float(quaternion.z)
        w = float(quaternion.w)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 0.0:
            return 0.0, 0.0, 0.0
        x /= norm
        y /= norm
        z /= norm
        w /= norm

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _wait_for_future(self, future, timeout_sec: float, label: str):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=max(0.1, timeout_sec)):
            self.get_logger().error(f'Timed out waiting for {label}.')
            return None
        try:
            return future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed waiting for {label}: {exc}')
            return None

    @staticmethod
    def _duration_msg(seconds: float) -> DurationMsg:
        seconds = max(0.0, float(seconds))
        whole = int(math.floor(seconds))
        msg = DurationMsg()
        msg.sec = whole
        msg.nanosec = int(round((seconds - whole) * 1e9))
        return msg

    @staticmethod
    def _format_joint_goal(joint_goal: Dict[str, float]) -> str:
        return ', '.join(f'{name}={value:.3f}' for name, value in joint_goal.items())

    def _publish_status(self, message: str) -> None:
        self.status_pub.publish(String(data=message))
        self.get_logger().info(message)

    def destroy_node(self):
        self.shutdown_event.set()
        time.sleep(0.1)
        try:
            cv2.destroyWindow(self.roi_window_name)
        except cv2.error:
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoiAnyGraspController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
