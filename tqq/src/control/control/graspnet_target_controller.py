import copy
from dataclasses import dataclass
import json
import math
import os
import sys
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge, CvBridgeError
from franka_msgs.action import Grasp as GripperGrasp
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_geometry_msgs  # noqa: F401  Registers geometry message transforms.


@dataclass
class GraspNetCommand:
    target_name: str = ''
    motion_speed: Optional[float] = None
    active: bool = False


@dataclass
class GraspNetTarget:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    grasp_score: float
    grasp_width: float
    grasp_depth: float
    camera_pose: PoseStamped
    execution_camera_pose: PoseStamped
    base_pose: PoseStamped
    roi_points: np.ndarray
    roi_colors: np.ndarray
    orientation_source: str
    motion_speed: Optional[float] = None


class _TemporaryMotionSpeed:
    def __init__(self, node, speed: Optional[float]) -> None:
        self.node = node
        self.speed = speed
        self.previous_velocity = node.max_velocity_scaling
        self.previous_acceleration = node.max_acceleration_scaling

    def __enter__(self):
        if self.speed is None:
            return
        speed = min(1.0, max(0.0, float(self.speed)))
        with self.node.state_lock:
            self.node.max_velocity_scaling = speed
            self.node.max_acceleration_scaling = speed
        self.node._publish_status(
            f'using one-shot GraspNet speed={speed:.3f}; default speed will be restored after this motion'
        )

    def __exit__(self, exc_type, exc, tb):
        if self.speed is None:
            return False
        with self.node.state_lock:
            self.node.max_velocity_scaling = self.previous_velocity
            self.node.max_acceleration_scaling = self.previous_acceleration
        self.node._publish_status(
            f'restored default motion speed: velocity={self.previous_velocity:.3f}, '
            f'acceleration={self.previous_acceleration:.3f}'
        )
        return False


class FrankaGraspNetBackend:
    """Use third_party/franka-graspnet-master inference while keeping ROS + MoveIt control."""

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
        root_dir = os.path.expanduser(self.node.franka_graspnet_dir)
        checkpoint_path = os.path.expanduser(self.node.graspnet_checkpoint_path)
        if not os.path.isdir(root_dir):
            raise RuntimeError(f'franka_graspnet_dir does not exist: {root_dir}')
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                'Franka-GraspNet checkpoint is missing. Set graspnet_checkpoint_path. '
                f'Current value: {checkpoint_path or "<empty>"}'
            )
        for path in (
            root_dir,
            os.path.join(root_dir, 'models'),
            os.path.join(root_dir, 'dataset'),
            os.path.join(root_dir, 'utils'),
            os.path.join(root_dir, 'graspnetAPI'),
        ):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from franka_graspnet.graspnet_infer import GraspNetInfer
        except Exception as exc:
            raise RuntimeError(
                'Failed to import franka_graspnet.graspnet_infer. Install/compile this '
                f'library dependencies first. Original error: {exc}'
            ) from exc

        cfg = type('FrankaGraspNetConfig', (), {})()
        cfg.checkpoint_path = checkpoint_path
        cfg.num_view = self.node.graspnet_num_view
        cfg.num_point = self.node.graspnet_num_point
        cfg.voxel_size = self.node.graspnet_voxel_size
        cfg.collision_thresh = self.node.graspnet_collision_thresh
        cfg.angle_threshold_deg = self.node.franka_graspnet_angle_threshold_deg
        self.inferencer = GraspNetInfer(cfg)
        self.node._publish_status(
            f'Franka-GraspNet backend loaded checkpoint={checkpoint_path}, '
            f'angle_threshold={cfg.angle_threshold_deg:.1f}deg'
        )

    def infer(self, points: np.ndarray, colors: np.ndarray):
        ok, message = self.ready()
        if not ok:
            raise RuntimeError(message)
        if points.shape[0] < self.node.graspnet_min_points:
            raise RuntimeError(
                f'not enough ROI point-cloud points for Franka-GraspNet: '
                f'{points.shape[0]} < {self.node.graspnet_min_points}'
            )

        end_points, cloud_o3d = self.inferencer.process_fs_data(
            points.astype(np.float32, copy=False),
            np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False),
        )
        gg = self.inferencer.predict_grasps(end_points, cloud_o3d, return_best=True)
        if len(gg) == 0:
            raise RuntimeError('Franka-GraspNet produced no valid grasp')
        return gg[0]


class GraspNetTargetController(Node):
    """YOLO ROI + depth point cloud + GraspNet pose + MoveIt grasp execution."""

    def __init__(self) -> None:
        super().__init__('graspnet_target_controller')
        self.callback_group = ReentrantCallbackGroup()

        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.backend = FrankaGraspNetBackend(self)

        self.latest_color: Optional[np.ndarray] = None
        self.latest_color_header = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_header = None
        self.latest_depth_encoding = ''
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_joint_state: Optional[JointState] = None
        self.pending_command = GraspNetCommand()
        self.moving = False
        self.state_lock = threading.Lock()

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
            String,
            self.detections_topic,
            self.detections_callback,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            self.target_command_topic,
            self.target_command_callback,
            10,
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
        self.grasp_pose_camera_pub = self.create_publisher(PoseStamped, '~/grasp_pose_camera', 10)
        self.grasp_pose_base_pub = self.create_publisher(PoseStamped, '~/grasp_pose_base', 10)
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
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.trajectory_action,
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

        self.get_logger().info(
            'GraspNet target controller ready. '
            f'target_command={self.target_command_topic}, detections={self.detections_topic}, '
            f'checkpoint={self.graspnet_checkpoint_path or "<missing>"}, '
            f'base_frame={self.base_frame}, ee={self.end_effector_frame}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('detections_topic', '/yolo/detections_json')
        self.declare_parameter('target_command_topic', '/graspnet_target_controller/target_command')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('use_latest_tf', True)
        self.declare_parameter('end_effector_frame', 'fr3_hand_tcp')
        self.declare_parameter('move_group_name', 'fr3_arm')
        self.declare_parameter('ik_service', '/compute_ik')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('trajectory_action', '/fr3_arm_controller/follow_joint_trajectory')
        self.declare_parameter('execution_mode', 'move_group')
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
        self.declare_parameter('min_confidence', 0.30)
        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('roi_padding_px', 12)
        self.declare_parameter('franka_graspnet_dir', '/home/tqq/TQQ_ws/third_party/franka-graspnet-master')
        self.declare_parameter('franka_graspnet_angle_threshold_deg', 30.0)
        self.declare_parameter('graspnet_checkpoint_path', '')
        self.declare_parameter('graspnet_device', '')
        self.declare_parameter('graspnet_num_point', 20000)
        self.declare_parameter('graspnet_min_points', 200)
        self.declare_parameter('graspnet_num_view', 300)
        self.declare_parameter('graspnet_collision_thresh', 0.01)
        self.declare_parameter('graspnet_voxel_size', 0.01)
        self.declare_parameter('graspnet_approach_dist', 0.05)
        self.declare_parameter('graspnet_orientation_mode', 'graspnet')
        self.declare_parameter('graspnet_fallback_to_current_orientation', False)
        self.declare_parameter('popup_preview_before_execute', True)
        self.declare_parameter('popup_preview_max_points', 20000)
        self.declare_parameter('popup_preview_frame_size_m', 0.08)
        self.declare_parameter('popup_preview_window_title', 'GraspNet grasp preview')
        self.declare_parameter('graspnet_tcp_rotation_rpy_grasp', [0.0, 1.57079632679, 0.0])
        self.declare_parameter('graspnet_tcp_offset_xyz_grasp', [0.0, 0.0, 0.0])
        self.declare_parameter('target_offset_xyz_base', [0.0, 0.0, 0.0])
        self.declare_parameter('min_grasp_z_m', 0.05)
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
        self.detections_topic = str(self.get_parameter('detections_topic').value)
        self.target_command_topic = str(self.get_parameter('target_command_topic').value)
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.camera_frame_override = str(self.get_parameter('camera_frame').value)
        self.use_latest_tf = bool(self.get_parameter('use_latest_tf').value)
        self.end_effector_frame = str(self.get_parameter('end_effector_frame').value)
        self.move_group_name = str(self.get_parameter('move_group_name').value)
        self.ik_service = str(self.get_parameter('ik_service').value)
        self.move_group_action = str(self.get_parameter('move_group_action').value)
        self.trajectory_action = str(self.get_parameter('trajectory_action').value)
        self.execution_mode = str(self.get_parameter('execution_mode').value).strip().lower()
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
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.depth_unit_scale = float(self.get_parameter('depth_unit_scale').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.roi_padding_px = max(0, int(self.get_parameter('roi_padding_px').value))
        self.franka_graspnet_dir = str(self.get_parameter('franka_graspnet_dir').value)
        self.franka_graspnet_angle_threshold_deg = float(
            self.get_parameter('franka_graspnet_angle_threshold_deg').value
        )
        self.graspnet_checkpoint_path = str(self.get_parameter('graspnet_checkpoint_path').value)
        self.graspnet_device = str(self.get_parameter('graspnet_device').value).strip()
        self.graspnet_num_point = int(self.get_parameter('graspnet_num_point').value)
        self.graspnet_min_points = int(self.get_parameter('graspnet_min_points').value)
        self.graspnet_num_view = int(self.get_parameter('graspnet_num_view').value)
        self.graspnet_collision_thresh = float(
            self.get_parameter('graspnet_collision_thresh').value
        )
        self.graspnet_voxel_size = float(self.get_parameter('graspnet_voxel_size').value)
        self.graspnet_approach_dist = float(self.get_parameter('graspnet_approach_dist').value)
        self.graspnet_orientation_mode = str(
            self.get_parameter('graspnet_orientation_mode').value
        ).strip().lower()
        self.graspnet_fallback_to_current_orientation = bool(
            self.get_parameter('graspnet_fallback_to_current_orientation').value
        )
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
        self.graspnet_tcp_rotation_rpy_grasp = self._float_list(
            self.get_parameter('graspnet_tcp_rotation_rpy_grasp').value,
            3,
            'graspnet_tcp_rotation_rpy_grasp',
        )
        self.graspnet_tcp_rotation_matrix = self._rpy_to_rotation_matrix(
            self.graspnet_tcp_rotation_rpy_grasp
        )
        self.graspnet_tcp_offset_xyz_grasp = self._float_list(
            self.get_parameter('graspnet_tcp_offset_xyz_grasp').value,
            3,
            'graspnet_tcp_offset_xyz_grasp',
        )
        self.target_offset_xyz_base = self._float_list(
            self.get_parameter('target_offset_xyz_base').value,
            3,
            'target_offset_xyz_base',
        )
        self.min_grasp_z_m = float(self.get_parameter('min_grasp_z_m').value)
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
        if self.execution_mode not in ('move_group', 'trajectory', 'ik_only'):
            raise ValueError('execution_mode must be one of: move_group, trajectory, ik_only')
        if self.graspnet_orientation_mode not in ('graspnet', 'current'):
            raise ValueError('graspnet_orientation_mode must be graspnet or current')

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

    def target_command_callback(self, msg: String) -> None:
        command = self._parse_command(msg.data)
        with self.state_lock:
            self.pending_command = command
        if not command.active:
            self._publish_status('GraspNet target command cleared; waiting for a new command.')
        elif command.target_name:
            speed_text = (
                f', speed={command.motion_speed:.3f}' if command.motion_speed is not None else ''
            )
            self._publish_status(
                f'GraspNet target command received: {command.target_name or "best"}{speed_text}.'
            )
        else:
            self._publish_status('GraspNet target command received: best YOLO detection.')

    def _parse_command(self, raw_text: str) -> GraspNetCommand:
        text = str(raw_text or '').strip()
        if not text:
            return GraspNetCommand()
        if not text.startswith('{'):
            return GraspNetCommand(target_name=text, active=True)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return GraspNetCommand(target_name=text, active=True)
        if not isinstance(data, dict):
            return GraspNetCommand(target_name=text, active=True)
        return GraspNetCommand(
            target_name=str(data.get('name') or data.get('object_name') or data.get('target') or '').strip(),
            motion_speed=self._optional_motion_speed(data),
            active=True,
        )

    @staticmethod
    def _optional_motion_speed(data: Dict) -> Optional[float]:
        raw = data.get('motion_speed') if 'motion_speed' in data else data.get('speed')
        if raw is None or raw == '':
            return None
        try:
            return min(1.0, max(0.0, float(raw)))
        except (TypeError, ValueError):
            return None

    def detections_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Ignoring malformed YOLO detections JSON: {exc}')
            return
        with self.state_lock:
            command = copy.deepcopy(self.pending_command)
        if not command.active:
            return
        detection = self._select_detection(payload.get('detections', []), command.target_name)
        if detection is None:
            self._publish_status(
                f'GraspNet waiting for target={command.target_name or "best"}; no matching YOLO detection.'
            )
            return
        started, message = self._start_graspnet_thread(payload, detection, command)
        if not started:
            self.get_logger().debug(message)

    def _select_detection(self, detections: Sequence[Dict], target_name: str) -> Optional[Dict]:
        candidates = []
        selected = target_name.strip().lower()
        for detection in detections:
            confidence = float(detection.get('confidence', 0.0))
            if confidence < self.min_confidence:
                continue
            class_name = str(detection.get('class_name', '')).strip()
            if selected and class_name.lower() != selected:
                continue
            if len(detection.get('bbox_xyxy', [])) != 4:
                continue
            candidates.append(detection)
        if not candidates:
            return None
        return max(candidates, key=lambda item: float(item.get('confidence', 0.0)))

    def _start_graspnet_thread(
        self,
        payload: Dict,
        detection: Dict,
        command: GraspNetCommand,
    ) -> Tuple[bool, str]:
        with self.state_lock:
            if self.moving:
                return False, 'GraspNet motion already running'
            self.moving = True
            self.pending_command = GraspNetCommand()
        thread = threading.Thread(
            target=self._graspnet_worker,
            args=(copy.deepcopy(payload), copy.deepcopy(detection), copy.deepcopy(command)),
            daemon=True,
        )
        thread.start()
        return True, 'started GraspNet grasp thread'

    def _graspnet_worker(self, payload: Dict, detection: Dict, command: GraspNetCommand) -> None:
        try:
            target = self._build_graspnet_target(payload, detection, command)
            if target is None:
                return
            self._publish_status(
                f'GraspNet target {target.class_name} yolo_conf={target.confidence:.2f} '
                f'grasp_score={target.grasp_score:.3f} width={target.grasp_width:.3f}m '
                f'base=({target.base_pose.pose.position.x:.3f}, '
                f'{target.base_pose.pose.position.y:.3f}, {target.base_pose.pose.position.z:.3f})'
            )
            self._publish_gripper_6d_status(target)
            if not self._preview_and_confirm(target):
                self._publish_status('GraspNet grasp canceled before motion; gripper will not close.')
                return
            with self._temporary_motion_speed(target.motion_speed):
                moved = self._execute_pose_motion(target.base_pose, 'graspnet target')
            if moved:
                self._close_gripper_for_grasp()
            else:
                self._publish_status('GraspNet motion did not report success; gripper will not close.')
        finally:
            with self.state_lock:
                self.moving = False

    def _build_graspnet_target(
        self,
        payload: Dict,
        detection: Dict,
        command: GraspNetCommand,
    ) -> Optional[GraspNetTarget]:
        snapshot = self._snapshot_rgbd()
        if snapshot is None:
            return None
        color, depth, depth_encoding, depth_header, camera_info = snapshot
        points, colors = self._roi_point_cloud(color, depth, depth_encoding, camera_info, detection)
        if points is None or colors is None:
            return None
        try:
            grasp = self.backend.infer(points, colors)
        except Exception as exc:
            self.get_logger().error(f'GraspNet inference failed: {exc}')
            self._publish_status(f'GraspNet inference failed: {exc}')
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
                f'Cannot transform GraspNet pose {camera_pose.header.frame_id} -> {self.base_frame}: {exc}'
            )
            return None
        base_pose.pose.position.x += self.target_offset_xyz_base[0]
        base_pose.pose.position.y += self.target_offset_xyz_base[1]
        base_pose.pose.position.z = max(
            self.min_grasp_z_m,
            base_pose.pose.position.z + self.target_offset_xyz_base[2],
        )
        orientation_source = 'graspnet'
        if self.graspnet_orientation_mode == 'current':
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
        bbox = tuple(float(value) for value in detection['bbox_xyxy'])
        return GraspNetTarget(
            class_name=str(detection.get('class_name', '')),
            confidence=float(detection.get('confidence', 0.0)),
            bbox_xyxy=bbox,
            grasp_score=float(grasp.score),
            grasp_width=float(grasp.width),
            grasp_depth=float(grasp.depth),
            camera_pose=camera_pose,
            execution_camera_pose=execution_camera_pose,
            base_pose=base_pose,
            roi_points=points.astype(np.float32, copy=False),
            roi_colors=colors.astype(np.float32, copy=False),
            orientation_source=orientation_source,
            motion_speed=command.motion_speed,
        )

    def _preview_and_confirm(self, target: GraspNetTarget) -> bool:
        self.grasp_pose_camera_pub.publish(target.camera_pose)
        self.grasp_pose_base_pub.publish(target.base_pose)
        self.gripper_6d_pose_camera_pub.publish(target.execution_camera_pose)
        self.gripper_6d_pose_base_pub.publish(target.base_pose)
        if not self.popup_preview_before_execute:
            return True
        try:
            self._show_open3d_preview(target)
        except Exception as exc:
            message = f'Failed to open GraspNet preview window: {exc}'
            self.get_logger().error(message)
            self._publish_status(message)
            return False
        try:
            answer = input('GraspNet preview closed. Press Enter to execute, or type c then Enter to cancel: ')
        except EOFError:
            self._publish_status(
                'No stdin available for GraspNet preview confirmation; canceling motion.'
            )
            return False
        if answer.strip().lower() in ('c', 'cancel', 'n', 'no', 'q', 'quit', 'stop'):
            self._publish_status('GraspNet preview canceled by operator.')
            return False
        self._publish_status('GraspNet preview confirmed by operator.')
        return True

    def _show_open3d_preview(self, target: GraspNetTarget) -> None:
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
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size,
            origin=[
                preview_pose.pose.position.x,
                preview_pose.pose.position.y,
                preview_pose.pose.position.z,
            ],
        )
        camera_frame.rotate(
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
            f'{target.class_name} score={target.grasp_score:.3f} '
            f'width={target.grasp_width:.3f}m '
            f'pose={target.orientation_source}'
        )
        self._publish_status(
            'Showing GraspNet popup preview. Close the Open3D window, then press Enter '
            'in this terminal to execute or type c to cancel.'
        )
        o3d.visualization.draw_geometries(
            [cloud, camera_frame, gripper_geometry],
            window_name=title,
            width=1280,
            height=720,
        )

    def _create_open3d_gripper_geometry(self, o3d, pose: PoseStamped, target: GraspNetTarget):
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

    def _publish_gripper_6d_status(self, target: GraspNetTarget) -> None:
        self._publish_status(
            'GraspNet gripper 6D pose '
            f'orientation_source={target.orientation_source}; '
            f'camera[{self._format_pose_6d(target.execution_camera_pose)}]; '
            f'base[{self._format_pose_6d(target.base_pose)}]'
        )

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
                f'{target_frame}; using raw GraspNet camera pose for preview/log: {exc}'
            )
            return copy.deepcopy(fallback_pose)

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
        detection: Dict,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        height, width = depth.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in detection['bbox_xyxy']]
        pad = self.roi_padding_px
        x0 = max(0, int(math.floor(x1)) - pad)
        x3 = min(width, int(math.ceil(x2)) + pad)
        y0 = max(0, int(math.floor(y1)) - pad)
        y3 = min(height, int(math.ceil(y2)) + pad)
        if x3 <= x0 or y3 <= y0:
            self.get_logger().warn(f'Invalid YOLO ROI: {(x1, y1, x2, y2)}')
            return None, None

        depth_m = depth.astype(np.float32)
        if self._depth_is_integer_millimeters(depth_encoding, depth.dtype):
            depth_m *= self.depth_unit_scale
        valid_depth = np.isfinite(depth_m)
        valid_depth &= depth_m >= self.min_depth_m
        valid_depth &= depth_m <= self.max_depth_m
        mask = np.zeros(depth_m.shape[:2], dtype=bool)
        mask[y0:y3, x0:x3] = True
        mask &= valid_depth
        ys, xs = np.nonzero(mask)
        if len(xs) < self.graspnet_min_points:
            self.get_logger().warn(
                f'ROI point cloud too small for GraspNet: {len(xs)} points in bbox {(x0, y0, x3, y3)}'
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
        tcp_rotation = grasp_rotation.dot(self.graspnet_tcp_rotation_matrix)
        offset = np.asarray(self.graspnet_tcp_offset_xyz_grasp, dtype=np.float64).reshape(3)
        translation = translation + grasp_rotation.dot(offset)
        pose.pose.position.x = float(translation[0])
        pose.pose.position.y = float(translation[1])
        pose.pose.position.z = float(translation[2])

        pose.pose.orientation = self._rotation_matrix_to_quaternion(tcp_rotation)
        return pose

    def _pose_with_current_orientation(self, target_pose: PoseStamped) -> Optional[PoseStamped]:
        orientation = self._current_end_effector_orientation()
        if orientation is None:
            return None
        pose = copy.deepcopy(target_pose)
        pose.pose.orientation = orientation
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

    def _temporary_motion_speed(self, speed: Optional[float]):
        return _TemporaryMotionSpeed(self, speed)

    def _execute_pose_motion(self, target_pose: PoseStamped, label: str) -> bool:
        motion_label = label
        joint_goal = self._compute_ik(target_pose, motion_label)
        if (
            joint_goal is None
            and self.graspnet_fallback_to_current_orientation
            and self.graspnet_orientation_mode == 'graspnet'
        ):
            fallback_pose = self._pose_with_current_orientation(target_pose)
            if fallback_pose is not None:
                fallback_label = f'{label} with current end-effector orientation'
                self._publish_status(
                    f'IK failed for {label}; retrying at the same position with current end-effector orientation.'
                )
                joint_goal = self._compute_ik(fallback_pose, fallback_label)
                if joint_goal is not None:
                    motion_label = fallback_label
        if joint_goal is None:
            return False
        if self.execution_mode == 'ik_only':
            self._publish_status(f'IK solved for {motion_label}: {self._format_joint_goal(joint_goal)}')
            return False
        if self.execution_mode == 'move_group':
            return self._execute_with_move_group(joint_goal, motion_label)
        return self._execute_with_trajectory_action(joint_goal, motion_label)

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

    def _execute_with_trajectory_action(self, joint_goal: Dict[str, float], label: str) -> bool:
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            self.get_logger().error(f'Trajectory action not available: {self.trajectory_action}')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.trajectory.joint_names = list(self.arm_joints)
        point = JointTrajectoryPoint()
        point.positions = [joint_goal[name] for name in self.arm_joints]
        point.time_from_start = self._duration_msg(self.move_group_result_timeout_sec)
        goal.trajectory.points.append(point)
        goal.goal_time_tolerance = self._duration_msg(1.0)
        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, f'{label} trajectory goal')
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Trajectory controller rejected the goal.')
            return False
        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.move_group_result_timeout_sec + self.action_wait_timeout_sec,
            f'{label} trajectory result',
        )
        if action_result is None:
            return False
        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self._publish_status(f'Trajectory {label} executed successfully.')
            return True
        self.get_logger().error(f'Trajectory failed for {label}: {result.error_code} {result.error_string}')
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
        constraints.name = 'graspnet_ik_joint_goal'
        for name in self.arm_joints:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = joint_goal[name]
            joint_constraint.tolerance_above = self.goal_joint_tolerance
            joint_constraint.tolerance_below = self.goal_joint_tolerance
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        return constraints

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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspNetTargetController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
