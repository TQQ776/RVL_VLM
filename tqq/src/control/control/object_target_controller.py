import copy
from dataclasses import dataclass
import json
import math
import threading
import time
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
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_geometry_msgs  # noqa: F401  Registers geometry message transforms.


@dataclass
class TargetState:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    center_uv: Tuple[int, int]
    depth_m: float
    camera_point: PointStamped
    base_point: PointStamped
    target_pose: PoseStamped
    motion_speed_override: Optional[float] = None


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
            f'using one-shot grasp speed={speed:.3f}; default speed will be restored after this motion'
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


class ObjectTargetController(Node):
    """Convert image detections plus aligned depth into FR3 target motions."""

    def __init__(self) -> None:
        super().__init__('object_target_controller')
        self.callback_group = ReentrantCallbackGroup()

        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_header = None
        self.latest_depth_encoding = ''
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_joint_state: Optional[JointState] = None
        self.latest_target: Optional[TargetState] = None
        self.pending_motion_speed_override: Optional[float] = None
        self.latest_detected_objects: List[Dict] = []
        self.last_auto_execute_time = 0.0
        self.last_detected_objects_log_time = 0.0
        self.execute_requested_by_command = False
        self.moving = False
        self.state_lock = threading.Lock()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
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

        self.camera_point_pub = self.create_publisher(PointStamped, '~/target_point_camera', 10)
        self.base_point_pub = self.create_publisher(PointStamped, '~/target_point_base', 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, '~/target_pose_base', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.detected_objects_pub = self.create_publisher(String, self.detected_objects_topic, 10)

        self.move_to_target_srv = self.create_service(
            Trigger,
            '~/move_to_target',
            self.move_to_target_callback,
            callback_group=self.callback_group,
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            self.ik_service,
            callback_group=self.callback_group,
        )
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            self.cartesian_path_service,
            callback_group=self.callback_group,
        )
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_group_action,
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
            'Object target controller ready. '
            f'detections={self.detections_topic}, depth={self.depth_topic}, '
            f'target_command={self.target_command_topic}, '
            f'base_frame={self.base_frame}, ee={self.end_effector_frame}, '
            f'execution_mode={self.execution_mode}, auto_execute={self.auto_execute}, '
            f'final_approach_z={self.grasp_final_z_offset_m:.3f}m, '
            f'staged_grasp_motion={self.staged_grasp_motion}, '
            f'use_cartesian_staged_motion={self.use_cartesian_staged_motion}, '
            f'cartesian_staged_segments={self.cartesian_staged_segments}, '
            f'min_grasp_z={self.min_grasp_z_m:.3f}m, '
            f'max_staged_xy={self.max_staged_xy_distance_m:.3f}m, '
            f'max_staged_vertical_descend={self.max_staged_vertical_descend_m:.3f}m, '
            f'close_gripper_after_motion={self.close_gripper_after_motion}, '
            f'velocity_scaling={self.max_velocity_scaling:.3f}, '
            f'acceleration_scaling={self.max_acceleration_scaling:.3f}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('detections_topic', '/mcp_omni_client/api_detections_json')
        self.declare_parameter('detected_objects_topic', '/object_target_controller/detected_objects')
        self.declare_parameter('target_command_topic', '/object_target_controller/target_class_name')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('use_latest_tf', True)
        self.declare_parameter('end_effector_frame', 'fr3_hand_tcp')
        self.declare_parameter('move_group_name', 'fr3_arm')
        self.declare_parameter('ik_service', '/compute_ik')
        self.declare_parameter('cartesian_path_service', '/compute_cartesian_path')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('trajectory_action', '/fr3_arm_controller/follow_joint_trajectory')
        self.declare_parameter('execution_mode', 'move_group')
        self.declare_parameter('plan_only', False)
        self.declare_parameter('auto_execute', False)
        self.declare_parameter('auto_execute_cooldown_sec', 5.0)
        self.declare_parameter('ask_for_target', True)
        self.declare_parameter('execute_on_target_command', True)
        self.declare_parameter('detected_objects_log_interval_sec', 2.0)
        self.declare_parameter('target_class_name', '')
        self.declare_parameter('target_class_id', -1)
        self.declare_parameter('min_confidence', 0.30)
        self.declare_parameter('depth_window_radius_px', 3)
        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('max_detection_depth_time_delta_sec', 0.5)
        self.declare_parameter('target_offset_xyz_base', [0.0, 0.0, 0.0])
        self.declare_parameter('grasp_final_z_offset_m', -0.06)
        self.declare_parameter('staged_grasp_motion', False)
        self.declare_parameter('use_cartesian_staged_motion', False)
        self.declare_parameter('cartesian_staged_segments', ['vertical'])
        self.declare_parameter('allow_cartesian_staged_fallback', False)
        self.declare_parameter('cartesian_max_step_m', 0.01)
        self.declare_parameter('cartesian_jump_threshold', 0.0)
        self.declare_parameter('cartesian_min_fraction', 0.95)
        self.declare_parameter('min_grasp_z_m', 0.05)
        self.declare_parameter('max_staged_xy_distance_m', 0.40)
        self.declare_parameter('max_staged_vertical_descend_m', 0.32)
        self.declare_parameter('close_gripper_after_motion', True)
        self.declare_parameter('close_gripper_on_service_motion', False)
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
        self.declare_parameter('orientation_mode', 'current')
        self.declare_parameter('fixed_orientation_xyzw', [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('avoid_collisions', True)
        self.declare_parameter('ik_timeout_sec', 0.5)
        self.declare_parameter('service_wait_timeout_sec', 5.0)
        self.declare_parameter('action_wait_timeout_sec', 10.0)
        self.declare_parameter('motion_duration_sec', 4.0)
        self.declare_parameter('move_group_result_timeout_sec', 90.0)
        self.declare_parameter('goal_joint_tolerance', 0.01)
        self.declare_parameter('max_velocity_scaling', 0.05)
        self.declare_parameter('max_acceleration_scaling', 0.05)
        self.declare_parameter('num_planning_attempts', 5)
        self.declare_parameter('allowed_planning_time', 5.0)
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
        self.detections_topic = str(self.get_parameter('detections_topic').value)
        self.detected_objects_topic = str(self.get_parameter('detected_objects_topic').value)
        self.target_command_topic = str(self.get_parameter('target_command_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.camera_frame_override = str(self.get_parameter('camera_frame').value)
        self.use_latest_tf = bool(self.get_parameter('use_latest_tf').value)
        self.end_effector_frame = str(self.get_parameter('end_effector_frame').value)
        self.move_group_name = str(self.get_parameter('move_group_name').value)
        self.ik_service = str(self.get_parameter('ik_service').value)
        self.cartesian_path_service = str(self.get_parameter('cartesian_path_service').value)
        self.move_group_action = str(self.get_parameter('move_group_action').value)
        self.trajectory_action = str(self.get_parameter('trajectory_action').value)
        self.execution_mode = str(self.get_parameter('execution_mode').value).strip().lower()
        self.plan_only = bool(self.get_parameter('plan_only').value)
        self.auto_execute = bool(self.get_parameter('auto_execute').value)
        self.auto_execute_cooldown_sec = float(self.get_parameter('auto_execute_cooldown_sec').value)
        self.ask_for_target = bool(self.get_parameter('ask_for_target').value)
        self.execute_on_target_command = bool(self.get_parameter('execute_on_target_command').value)
        self.detected_objects_log_interval_sec = float(
            self.get_parameter('detected_objects_log_interval_sec').value
        )
        self.target_class_name = str(self.get_parameter('target_class_name').value)
        self.target_class_id = int(self.get_parameter('target_class_id').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.depth_window_radius_px = max(0, int(self.get_parameter('depth_window_radius_px').value))
        self.depth_unit_scale = float(self.get_parameter('depth_unit_scale').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.max_detection_depth_time_delta_sec = float(
            self.get_parameter('max_detection_depth_time_delta_sec').value
        )
        self.target_offset_xyz_base = self._float_list(
            self.get_parameter('target_offset_xyz_base').value,
            3,
            'target_offset_xyz_base',
        )
        self.grasp_final_z_offset_m = float(self.get_parameter('grasp_final_z_offset_m').value)
        self.staged_grasp_motion = bool(self.get_parameter('staged_grasp_motion').value)
        self.use_cartesian_staged_motion = bool(
            self.get_parameter('use_cartesian_staged_motion').value
        )
        self.cartesian_staged_segments = {
            str(item).strip().lower()
            for item in self.get_parameter('cartesian_staged_segments').value
            if str(item).strip()
        }
        self.allow_cartesian_staged_fallback = bool(
            self.get_parameter('allow_cartesian_staged_fallback').value
        )
        self.cartesian_max_step_m = max(0.001, float(self.get_parameter('cartesian_max_step_m').value))
        self.cartesian_jump_threshold = float(self.get_parameter('cartesian_jump_threshold').value)
        self.cartesian_min_fraction = min(
            1.0,
            max(0.0, float(self.get_parameter('cartesian_min_fraction').value)),
        )
        self.min_grasp_z_m = float(self.get_parameter('min_grasp_z_m').value)
        self.max_staged_xy_distance_m = max(
            0.0,
            float(self.get_parameter('max_staged_xy_distance_m').value),
        )
        self.max_staged_vertical_descend_m = max(
            0.0,
            float(self.get_parameter('max_staged_vertical_descend_m').value),
        )
        self.close_gripper_after_motion = bool(self.get_parameter('close_gripper_after_motion').value)
        self.close_gripper_on_service_motion = bool(
            self.get_parameter('close_gripper_on_service_motion').value
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
        self.orientation_mode = str(self.get_parameter('orientation_mode').value).strip().lower()
        self.fixed_orientation_xyzw = self._float_list(
            self.get_parameter('fixed_orientation_xyzw').value,
            4,
            'fixed_orientation_xyzw',
        )
        self.avoid_collisions = bool(self.get_parameter('avoid_collisions').value)
        self.ik_timeout_sec = float(self.get_parameter('ik_timeout_sec').value)
        self.service_wait_timeout_sec = float(self.get_parameter('service_wait_timeout_sec').value)
        self.action_wait_timeout_sec = float(self.get_parameter('action_wait_timeout_sec').value)
        self.motion_duration_sec = float(self.get_parameter('motion_duration_sec').value)
        self.move_group_result_timeout_sec = float(
            self.get_parameter('move_group_result_timeout_sec').value
        )
        self.goal_joint_tolerance = float(self.get_parameter('goal_joint_tolerance').value)
        self.max_velocity_scaling = float(self.get_parameter('max_velocity_scaling').value)
        self.max_acceleration_scaling = float(self.get_parameter('max_acceleration_scaling').value)
        self.num_planning_attempts = int(self.get_parameter('num_planning_attempts').value)
        self.allowed_planning_time = float(self.get_parameter('allowed_planning_time').value)
        self.arm_joints = [str(name) for name in self.get_parameter('arm_joints').value]

        if self.execution_mode not in ('move_group', 'trajectory', 'ik_only'):
            raise ValueError('execution_mode must be one of: move_group, trajectory, ik_only')
        if self.orientation_mode not in ('current', 'fixed'):
            raise ValueError('orientation_mode must be either current or fixed')
        if not self.gripper_grasp_actions:
            raise ValueError('gripper_grasp_actions must contain at least one action name')

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
        target_name, motion_speed = self._parse_target_command(msg.data)
        with self.state_lock:
            if target_name:
                self.target_class_name = target_name
                self.target_class_id = -1
                self.pending_motion_speed_override = motion_speed
                self.latest_target = None
                self.execute_requested_by_command = self.execute_on_target_command
            else:
                self.target_class_name = ''
                self.target_class_id = -1
                self.pending_motion_speed_override = None
                self.latest_target = None
                self.execute_requested_by_command = False

        if target_name:
            action = 'will move when it sees the target' if self.execute_on_target_command else 'target selected'
            speed_text = (
                f'; one-shot speed={motion_speed:.3f}'
                if motion_speed is not None
                else ''
            )
            self._publish_status(f'target command received: {target_name}{speed_text}; {action}.')
        else:
            self._publish_status('target command cleared; waiting for a new object name.')

    def _parse_target_command(self, raw_text: str) -> Tuple[str, Optional[float]]:
        text = str(raw_text or '').strip()
        if not text:
            return '', None
        if not text.startswith('{'):
            return text, None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text, None
        if not isinstance(data, dict):
            return text, None
        name = str(data.get('name') or data.get('object_name') or data.get('target') or '').strip()
        return name, self._optional_motion_speed(data)

    @staticmethod
    def _optional_motion_speed(data: Dict) -> Optional[float]:
        raw = data.get('motion_speed') if 'motion_speed' in data else data.get('speed')
        if raw is None or raw == '':
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, value))

    def detections_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Ignoring malformed detections JSON: {exc}')
            return

        detections = payload.get('detections', [])
        summary = self._publish_detected_objects(detections)
        if self.ask_for_target and not self._has_target_filter():
            with self.state_lock:
                self.execute_requested_by_command = False
            self._maybe_log_target_prompt(summary)
            return

        detection = self._select_detection(detections)
        if detection is None:
            self._maybe_log_waiting_for_target(summary)
            return

        target = self._build_target(payload, detection)
        if target is None:
            return

        with self.state_lock:
            motion_speed_override = self.pending_motion_speed_override
            self.latest_target = target
            self.latest_target.motion_speed_override = motion_speed_override
            self.pending_motion_speed_override = None

        self.camera_point_pub.publish(target.camera_point)
        self.base_point_pub.publish(target.base_point)
        self.target_pose_pub.publish(target.target_pose)
        self._publish_status(
            f'target {target.class_name} conf={target.confidence:.2f} '
            f'uv={target.center_uv} depth={target.depth_m:.3f}m '
            f'base=({target.base_point.point.x:.3f}, '
            f'{target.base_point.point.y:.3f}, {target.base_point.point.z:.3f})'
        )

        if self.auto_execute:
            now = time.monotonic()
            if now - self.last_auto_execute_time >= self.auto_execute_cooldown_sec:
                started, message = self._start_motion_thread('auto_execute')
                if started:
                    self.last_auto_execute_time = now
                else:
                    self.get_logger().debug(message)

        if self.execute_requested_by_command:
            started, message = self._start_motion_thread('target_command')
            if started:
                self.execute_requested_by_command = False
            else:
                self.get_logger().debug(message)

    def _publish_detected_objects(self, detections: Sequence[Dict]) -> List[Dict]:
        by_name: Dict[str, Dict] = {}
        for detection in detections:
            confidence = float(detection.get('confidence', 0.0))
            if confidence < self.min_confidence:
                continue

            class_id = int(detection.get('class_id', -1))
            class_name = str(detection.get('class_name', class_id))
            item = by_name.setdefault(class_name, {
                'class_name': class_name,
                'class_id': class_id,
                'count': 0,
                'best_confidence': 0.0,
            })
            item['count'] += 1
            item['best_confidence'] = max(float(item['best_confidence']), confidence)

        summary = sorted(
            by_name.values(),
            key=lambda item: (-int(item['count']), -float(item['best_confidence']), str(item['class_name'])),
        )
        with self.state_lock:
            self.latest_detected_objects = copy.deepcopy(summary)

        self.detected_objects_pub.publish(String(data=json.dumps({
            'objects': summary,
            'target_class_name': self.target_class_name,
        }, ensure_ascii=True)))
        return summary

    def _has_target_filter(self) -> bool:
        return bool(self.target_class_name.strip()) or self.target_class_id >= 0

    def _target_description(self) -> str:
        if self.target_class_name.strip():
            return self.target_class_name.strip()
        if self.target_class_id >= 0:
            return f'class_id={self.target_class_id}'
        return ''

    def _format_detected_objects(self, summary: Sequence[Dict]) -> str:
        if not summary:
            return 'none'
        return ', '.join(
            f'{item["class_name"]} x{item["count"]} ({float(item["best_confidence"]):.2f})'
            for item in summary
        )

    def _maybe_log_target_prompt(self, summary: Sequence[Dict]) -> None:
        now = time.monotonic()
        if now - self.last_detected_objects_log_time < self.detected_objects_log_interval_sec:
            return

        objects_text = self._format_detected_objects(summary)
        self._publish_status(
            f'detected objects: {objects_text}. Tell me what to grab with: '
            'ros2 topic pub --once /object_target_controller/target_class_name '
            'std_msgs/msg/String "{data: apple}"'
        )
        self.last_detected_objects_log_time = now

    def _maybe_log_waiting_for_target(self, summary: Sequence[Dict]) -> None:
        now = time.monotonic()
        if now - self.last_detected_objects_log_time < self.detected_objects_log_interval_sec:
            return

        self._publish_status(
            f'waiting for target {self._target_description()}; '
            f'detected objects: {self._format_detected_objects(summary)}'
        )
        self.last_detected_objects_log_time = now

    def _select_detection(self, detections: Sequence[Dict]) -> Optional[Dict]:
        candidates = []
        selected_name = self.target_class_name.strip()
        for detection in detections:
            confidence = float(detection.get('confidence', 0.0))
            if confidence < self.min_confidence:
                continue

            class_id = int(detection.get('class_id', -1))
            class_name = str(detection.get('class_name', ''))
            if self.target_class_id >= 0 and class_id != self.target_class_id:
                continue
            if selected_name and class_name.lower() != selected_name.lower():
                continue
            if len(detection.get('bbox_xyxy', [])) != 4:
                continue
            candidates.append(detection)

        if not candidates:
            return None
        return max(candidates, key=lambda item: float(item.get('confidence', 0.0)))

    def _build_target(self, payload: Dict, detection: Dict) -> Optional[TargetState]:
        bbox = tuple(float(value) for value in detection['bbox_xyxy'])
        u = int(round((bbox[0] + bbox[2]) * 0.5))
        v = int(round((bbox[1] + bbox[3]) * 0.5))

        with self.state_lock:
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            depth_header = copy.deepcopy(self.latest_depth_header)
            depth_encoding = self.latest_depth_encoding
            camera_info = copy.deepcopy(self.latest_camera_info)

        if depth is None or depth_header is None:
            self.get_logger().warn('No aligned depth image received yet.')
            return None
        if camera_info is None:
            self.get_logger().warn('No camera info received yet.')
            return None

        self._warn_if_depth_time_far_from_detection(payload, depth_header)
        depth_m = self._sample_depth(depth, depth_encoding, u, v)
        if depth_m is None:
            self.get_logger().warn(f'No valid depth around pixel ({u}, {v}).')
            return None

        camera_point = self._pixel_to_camera_point(u, v, depth_m, camera_info, depth_header)
        if camera_point is None:
            return None

        try:
            base_point = self.tf_buffer.transform(
                camera_point,
                self.base_frame,
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Cannot transform {camera_point.header.frame_id} -> {self.base_frame}: {exc}'
            )
            return None

        target_pose = self._make_target_pose(base_point)
        if target_pose is None:
            return None

        return TargetState(
            class_id=int(detection.get('class_id', -1)),
            class_name=str(detection.get('class_name', '')),
            confidence=float(detection.get('confidence', 0.0)),
            bbox_xyxy=bbox,
            center_uv=(u, v),
            depth_m=depth_m,
            camera_point=camera_point,
            base_point=base_point,
            target_pose=target_pose,
        )

    def _warn_if_depth_time_far_from_detection(self, payload: Dict, depth_header) -> None:
        detection_stamp = payload.get('header', {}).get('stamp', {})
        if 'sec' not in detection_stamp or 'nanosec' not in detection_stamp:
            return
        detection_time = float(detection_stamp['sec']) + float(detection_stamp['nanosec']) * 1e-9
        depth_time = float(depth_header.stamp.sec) + float(depth_header.stamp.nanosec) * 1e-9
        if detection_time <= 0.0 or depth_time <= 0.0:
            return
        delta = abs(detection_time - depth_time)
        if delta > self.max_detection_depth_time_delta_sec:
            self.get_logger().warn(
                f'Detection/depth stamps differ by {delta:.3f}s; target depth may be stale.'
            )

    def _sample_depth(
        self,
        depth: np.ndarray,
        encoding: str,
        u: int,
        v: int,
    ) -> Optional[float]:
        height, width = depth.shape[:2]
        if u < 0 or v < 0 or u >= width or v >= height:
            self.get_logger().warn(f'Target pixel ({u}, {v}) outside depth image {width}x{height}.')
            return None

        radius = self.depth_window_radius_px
        x0 = max(0, u - radius)
        x1 = min(width, u + radius + 1)
        y0 = max(0, v - radius)
        y1 = min(height, v + radius + 1)
        window = depth[y0:y1, x0:x1].astype(np.float64)

        if self._depth_is_integer_millimeters(encoding, depth.dtype):
            window *= self.depth_unit_scale

        valid = window[np.isfinite(window)]
        valid = valid[(valid >= self.min_depth_m) & (valid <= self.max_depth_m)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    def _depth_is_integer_millimeters(encoding: str, dtype: np.dtype) -> bool:
        if encoding.upper() in ('16UC1', 'MONO16'):
            return True
        return np.issubdtype(dtype, np.integer)

    def _pixel_to_camera_point(
        self,
        u: int,
        v: int,
        depth_m: float,
        camera_info: CameraInfo,
        depth_header,
    ) -> Optional[PointStamped]:
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            self.get_logger().warn('CameraInfo has invalid focal length.')
            return None

        point = PointStamped()
        point.header = copy.deepcopy(depth_header)
        point.header.frame_id = self.camera_frame_override or depth_header.frame_id or camera_info.header.frame_id
        if self.use_latest_tf:
            point.header.stamp = Time().to_msg()
        point.point.x = (float(u) - cx) * depth_m / fx
        point.point.y = (float(v) - cy) * depth_m / fy
        point.point.z = depth_m
        return point

    def _make_target_pose(self, base_point: PointStamped) -> Optional[PoseStamped]:
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = base_point.point.x + self.target_offset_xyz_base[0]
        pose.pose.position.y = base_point.point.y + self.target_offset_xyz_base[1]
        raw_z = (
            base_point.point.z
            + self.target_offset_xyz_base[2]
            + self.grasp_final_z_offset_m
        )
        pose.pose.position.z = max(self.min_grasp_z_m, raw_z)
        if pose.pose.position.z > raw_z:
            self.get_logger().warn(
                f'Clamped grasp z from {raw_z:.3f}m to min_grasp_z_m='
                f'{self.min_grasp_z_m:.3f}m.'
            )

        if self.orientation_mode == 'fixed':
            pose.pose.orientation = Quaternion(
                x=self.fixed_orientation_xyzw[0],
                y=self.fixed_orientation_xyzw[1],
                z=self.fixed_orientation_xyzw[2],
                w=self.fixed_orientation_xyzw[3],
            )
            return pose

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                Time(),
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Cannot get current end-effector orientation '
                f'{self.base_frame} -> {self.end_effector_frame}: {exc}'
            )
            return None

        pose.pose.orientation = transform.transform.rotation
        return pose

    def _make_pose_at_position(
        self,
        source_pose: PoseStamped,
        x: float,
        y: float,
        z: float,
    ) -> PoseStamped:
        pose = copy.deepcopy(source_pose)
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        return pose

    def _current_end_effector_pose(self) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                Time(),
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Cannot get current end-effector pose '
                f'{self.base_frame} -> {self.end_effector_frame}: {exc}'
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def move_to_target_callback(self, request, response):
        started, message = self._start_motion_thread('service')
        response.success = started
        response.message = message
        return response

    def _start_motion_thread(self, reason: str) -> Tuple[bool, str]:
        with self.state_lock:
            if self.moving:
                return False, 'motion already running'
            if self.latest_target is None:
                return False, 'no target available yet'
            target = copy.deepcopy(self.latest_target)
            self.moving = True

        thread = threading.Thread(
            target=self._motion_worker,
            args=(target, reason),
            daemon=True,
        )
        thread.start()
        return True, f'started motion for {target.class_name or target.class_id}'

    def _motion_worker(self, target: TargetState, reason: str) -> None:
        try:
            self._publish_status(f'motion requested by {reason}')
            with self._temporary_motion_speed(target.motion_speed_override):
                moved = self._execute_target_motion(target, reason)
                should_close = self._should_close_gripper(reason)

            if moved and should_close:
                self._close_gripper_for_grasp()
            elif not moved:
                self._publish_status(
                    f'motion for {target.class_name or target.class_id} did not report success; '
                    'gripper will not close.'
                )
            elif not should_close:
                self._publish_status(
                    f'motion finished for {target.class_name or target.class_id}; '
                    f'gripper close is disabled for reason={reason}.'
                )
        finally:
            with self.state_lock:
                self.moving = False

    def _temporary_motion_speed(self, speed: Optional[float]):
        return _TemporaryMotionSpeed(self, speed)

    def _execute_target_motion(self, target: TargetState, reason: str) -> bool:
        if self._should_use_staged_grasp(reason):
            return self._execute_staged_target_motion(target)
        return self._execute_pose_motion(target.target_pose, 'target')

    def _should_use_staged_grasp(self, reason: str) -> bool:
        return self.staged_grasp_motion and self._should_close_gripper(reason)

    def _execute_staged_target_motion(self, target: TargetState) -> bool:
        current_pose = self._current_end_effector_pose()
        if current_pose is None:
            return False

        final_pose = target.target_pose
        xy_distance = math.hypot(
            final_pose.pose.position.x - current_pose.pose.position.x,
            final_pose.pose.position.y - current_pose.pose.position.y,
        )
        if (
            self.max_staged_xy_distance_m > 0.0
            and xy_distance > self.max_staged_xy_distance_m
        ):
            message = (
                f'staged grasp rejected: XY move would be {xy_distance:.3f}m, '
                f'exceeding max_staged_xy_distance_m='
                f'{self.max_staged_xy_distance_m:.3f}m.'
            )
            self.get_logger().error(message)
            self._publish_status(message)
            return False

        # For the staged API/Yolo grasp, keep the approach posture steady.
        # The default "current" orientation should not introduce an extra
        # wrist rotation during the XY move, otherwise the first segment can
        # look much less smooth than the original one-shot MoveGroup motion.
        staged_orientation_pose = final_pose if self.orientation_mode == 'fixed' else current_pose

        xy_pose = self._make_pose_at_position(
            staged_orientation_pose,
            final_pose.pose.position.x,
            final_pose.pose.position.y,
            current_pose.pose.position.z,
        )

        self._publish_status(
            'staged grasp step 1/2: move above target in XY '
            f'at current z={xy_pose.pose.position.z:.3f}m; '
            f'xy_distance={xy_distance:.3f}m'
        )
        if not self._execute_staged_segment(current_pose, xy_pose, 'xy pre-grasp', 'xy'):
            return False

        after_xy_pose = self._current_end_effector_pose()
        if after_xy_pose is None:
            return False

        descend_orientation_pose = final_pose if self.orientation_mode == 'fixed' else after_xy_pose
        descend_pose = self._make_pose_at_position(
            descend_orientation_pose,
            after_xy_pose.pose.position.x,
            after_xy_pose.pose.position.y,
            final_pose.pose.position.z,
        )
        vertical_descend_m = after_xy_pose.pose.position.z - descend_pose.pose.position.z
        if vertical_descend_m < -0.001:
            message = (
                f'staged grasp rejected: final z={descend_pose.pose.position.z:.3f}m '
                f'is above pre-grasp z={after_xy_pose.pose.position.z:.3f}m.'
            )
            self.get_logger().error(message)
            self._publish_status(message)
            return False
        if (
            self.max_staged_vertical_descend_m > 0.0
            and vertical_descend_m > self.max_staged_vertical_descend_m
        ):
            message = (
                f'staged grasp rejected: vertical descend would be '
                f'{vertical_descend_m:.3f}m, exceeding '
                f'max_staged_vertical_descend_m='
                f'{self.max_staged_vertical_descend_m:.3f}m. '
                'Move closer above the object first or increase the limit after checking safety.'
            )
            self.get_logger().error(message)
            self._publish_status(message)
            return False
        self._publish_status(
            'staged grasp step 2/2: descend vertically to '
            f'z={descend_pose.pose.position.z:.3f}m; '
            f'descend={vertical_descend_m:.3f}m'
        )
        return self._execute_staged_segment(xy_pose, descend_pose, 'vertical grasp', 'vertical')

    def _execute_staged_segment(
        self,
        start_pose: PoseStamped,
        target_pose: PoseStamped,
        label: str,
        segment_kind: str,
    ) -> bool:
        use_cartesian = (
            self.use_cartesian_staged_motion
            and self.execution_mode == 'move_group'
            and segment_kind != 'xy'
            and (
                'all' in self.cartesian_staged_segments
                or segment_kind in self.cartesian_staged_segments
            )
        )
        if use_cartesian:
            if self._execute_cartesian_segment(start_pose, target_pose, label):
                return True
            if not self.allow_cartesian_staged_fallback:
                self.get_logger().error(
                    f'Cartesian {label} failed; refusing fallback to a non-Cartesian '
                    'point-to-point segment.'
                )
                return False
            self.get_logger().warn(f'Cartesian {label} failed; falling back to IK/MoveGroup.')
        return self._execute_pose_motion(target_pose, label)

    def _execute_cartesian_segment(
        self,
        start_pose: PoseStamped,
        target_pose: PoseStamped,
        label: str,
    ) -> bool:
        if self.plan_only:
            return False
        if not self.cartesian_path_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().warn(
                f'Cartesian path service not available: {self.cartesian_path_service}'
            )
            return False

        with self.state_lock:
            joint_state = copy.deepcopy(self.latest_joint_state)

        if joint_state is None:
            self.get_logger().error('No /joint_states received; cannot compute Cartesian path.')
            return False

        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state.joint_state = joint_state
        request.start_state.is_diff = True
        request.group_name = self.move_group_name
        request.link_name = self.end_effector_frame
        request.waypoints = [copy.deepcopy(target_pose.pose)]
        request.max_step = self.cartesian_max_step_m
        request.jump_threshold = self.cartesian_jump_threshold
        request.avoid_collisions = self.avoid_collisions

        self._publish_status(
            f'Cartesian path request for {label}: '
            f'from=({start_pose.pose.position.x:.3f}, {start_pose.pose.position.y:.3f}, '
            f'{start_pose.pose.position.z:.3f}) '
            f'to=({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, '
            f'{target_pose.pose.position.z:.3f})'
        )

        future = self.cartesian_path_client.call_async(request)
        response = self._wait_for_future(
            future,
            self.service_wait_timeout_sec + self.allowed_planning_time + 1.0,
            f'{label} Cartesian path',
        )
        if response is None:
            return False
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(
                f'Cartesian path failed for {label} with code {response.error_code.val}'
            )
            return False
        if response.fraction < self.cartesian_min_fraction:
            self.get_logger().warn(
                f'Cartesian path for {label} incomplete: '
                f'fraction={response.fraction:.3f}, required={self.cartesian_min_fraction:.3f}'
            )
            return False
        if not response.solution.joint_trajectory.points:
            self.get_logger().warn(f'Cartesian path for {label} returned an empty trajectory.')
            return False

        self._time_parameterize_cartesian_trajectory(
            response.solution.joint_trajectory,
            self.motion_duration_sec,
        )
        return self._execute_joint_trajectory(
            response.solution.joint_trajectory,
            label,
            self.motion_duration_sec + self.action_wait_timeout_sec,
        )

    def _execute_pose_motion(self, target_pose: PoseStamped, label: str) -> bool:
        joint_goal = self._compute_ik(target_pose)
        if joint_goal is None:
            return False

        if self.execution_mode == 'ik_only':
            self._publish_status(f'IK solved for {label}: {self._format_joint_goal(joint_goal)}')
            return False

        if self.execution_mode == 'move_group':
            return self._execute_with_move_group(joint_goal, label)
        return self._execute_with_trajectory_action(joint_goal, label)

    def _time_parameterize_cartesian_trajectory(
        self,
        trajectory: JointTrajectory,
        total_duration_sec: float,
    ) -> None:
        points = trajectory.points
        if not points:
            return

        duration = max(0.1, float(total_duration_sec))
        if len(points) == 1:
            points[0].time_from_start = self._duration_msg(duration)
            return

        step = duration / float(len(points) - 1)
        for index, point in enumerate(points):
            point.time_from_start = self._duration_msg(max(0.05, step * index))

    def _execute_joint_trajectory(
        self,
        trajectory: JointTrajectory,
        label: str,
        timeout_sec: float,
    ) -> bool:
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            self.get_logger().error(f'Trajectory action not available: {self.trajectory_action}')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = copy.deepcopy(trajectory)
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.goal_time_tolerance = self._duration_msg(1.0)

        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.action_wait_timeout_sec,
            f'{label} Cartesian trajectory goal',
        )
        if goal_handle is None:
            return False
        if not goal_handle.accepted:
            self.get_logger().error(f'Trajectory controller rejected {label}.')
            return False

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            timeout_sec,
            f'{label} Cartesian trajectory result',
        )
        if action_result is None:
            return False

        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self._publish_status(f'Cartesian trajectory {label} executed successfully.')
            return True

        self.get_logger().error(
            f'Cartesian trajectory failed for {label}: '
            f'{result.error_code} {result.error_string}'
        )
        return False

    def _should_close_gripper(self, reason: str) -> bool:
        if not self.close_gripper_after_motion:
            return False
        if reason == 'target_command':
            return True
        if reason in ('auto_execute', 'service'):
            return self.close_gripper_on_service_motion
        return False

    def _compute_ik(self, target_pose: PoseStamped) -> Optional[Dict[str, float]]:
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
            self.get_logger().error(f'MoveIt IK failed with code {response.error_code.val}')
            return None

        positions = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        missing = [name for name in self.arm_joints if name not in positions]
        if missing:
            self.get_logger().error(f'IK solution missing joints: {missing}')
            return None

        joint_goal = {name: float(positions[name]) for name in self.arm_joints}
        self._publish_status(f'IK solved: {self._format_joint_goal(joint_goal)}')
        return joint_goal

    def _execute_with_move_group(self, joint_goal: Dict[str, float], label: str = 'target') -> bool:
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
        self._publish_status(
            f'MoveGroup request for {label}: velocity_scaling={self.max_velocity_scaling:.3f}, '
            f'acceleration_scaling={self.max_acceleration_scaling:.3f}, '
            f'result_timeout={self.move_group_result_timeout_sec:.1f}s'
        )
        goal.request.start_state.joint_state = joint_state
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(self._joint_goal_constraints(joint_goal))

        goal.planning_options.plan_only = self.plan_only
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.replan_delay = 0.5
        goal.planning_options.planning_scene_diff.is_diff = True

        send_future = self.move_group_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, 'MoveGroup goal')
        if goal_handle is None:
            return False
        if not goal_handle.accepted:
            self.get_logger().error('MoveGroup rejected the goal.')
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
                    f'MoveGroup result timed out for {label}, but current joint state '
                    'is within goal tolerance; treating motion as successful.'
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

    def _joint_goal_reached(self, joint_goal: Dict[str, float]) -> bool:
        with self.state_lock:
            joint_state = copy.deepcopy(self.latest_joint_state)

        if joint_state is None:
            self.get_logger().warn('Cannot verify final joint pose: no /joint_states received.')
            return False

        positions = dict(zip(joint_state.name, joint_state.position))
        missing = [name for name in self.arm_joints if name not in positions]
        if missing:
            self.get_logger().warn(f'Cannot verify final joint pose; missing joints: {missing}')
            return False

        tolerance = max(0.001, self.goal_joint_tolerance)
        errors = {
            name: abs(float(positions[name]) - float(joint_goal[name]))
            for name in self.arm_joints
        }
        max_error = max(errors.values()) if errors else float('inf')
        reached = all(error <= tolerance for error in errors.values())
        if not reached:
            self.get_logger().warn(
                f'Final joint pose not reached: max_error={max_error:.4f}rad, '
                f'tolerance={tolerance:.4f}rad'
            )
        return reached

    def _execute_with_trajectory_action(self, joint_goal: Dict[str, float], label: str = 'target') -> bool:
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            self.get_logger().error(f'Trajectory action not available: {self.trajectory_action}')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.trajectory.joint_names = list(self.arm_joints)

        point = JointTrajectoryPoint()
        point.positions = [joint_goal[name] for name in self.arm_joints]
        point.time_from_start = self._duration_msg(self.motion_duration_sec)
        goal.trajectory.points.append(point)
        goal.goal_time_tolerance = self._duration_msg(1.0)

        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, f'{label} trajectory goal')
        if goal_handle is None:
            return False
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory controller rejected the goal.')
            return False

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.motion_duration_sec + self.action_wait_timeout_sec,
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
            self.get_logger().error(
                'No gripper grasp action available; tried '
                + ', '.join(self.gripper_grasp_actions)
            )
            return False

        goal = GripperGrasp.Goal()
        goal.width = self.gripper_close_width_m
        goal.speed = self.gripper_close_speed_mps
        goal.force = self.gripper_close_force_n
        goal.epsilon.inner = self.gripper_grasp_epsilon_inner_m
        goal.epsilon.outer = self.gripper_grasp_epsilon_outer_m

        self._publish_status(
            f'closing gripper via {client_name}: '
            f'width={goal.width:.3f}m speed={goal.speed:.3f}m/s force={goal.force:.1f}N'
        )
        send_future = client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.gripper_action_timeout_sec,
            'gripper grasp goal',
        )
        if goal_handle is None:
            return False
        if not goal_handle.accepted:
            self.get_logger().error(f'Gripper grasp rejected by {client_name}.')
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

        error = result.error
        if not error:
            error = (
                'grasp did not satisfy width/epsilon; increase '
                'gripper_grasp_epsilon_outer_m or set gripper_close_width_m '
                'near the object width'
            )
        self.get_logger().error(f'Gripper grasp failed: {error}')
        return False

    def _available_gripper_grasp_client(self):
        for action_name, client in self.gripper_grasp_clients:
            if client.wait_for_server(timeout_sec=self.gripper_server_wait_timeout_sec):
                return action_name, client
        return '', None

    def _joint_goal_constraints(self, joint_goal: Dict[str, float]) -> Constraints:
        constraints = Constraints()
        constraints.name = 'ik_joint_goal'
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
    node = ObjectTargetController()
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
