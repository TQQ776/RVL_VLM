import base64
import copy
import io
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Grasp as GripperGrasp
from franka_msgs.action import Move as GripperMove
from geometry_msgs.msg import PoseStamped, Quaternion
from mcp.srv import CallTool, ListTools, MoveAxis, ObjectName
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
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class McpServer(Node):
    """Expose robot motion tools as ROS services for an LLM client."""

    def __init__(self) -> None:
        super().__init__('mcp_server')
        self.callback_group = ReentrantCallbackGroup()
        self.motion_lock = threading.Lock()
        self.gripper_lock = threading.Lock()
        self.detected_objects_lock = threading.Lock()
        self.latest_joint_state: Optional[JointState] = None
        self.latest_detected_objects_payload: Optional[Dict] = None
        self.latest_detected_objects_time = 0.0
        self.grasp_results_lock = threading.Lock()
        self.grasp_result_events: Dict[str, threading.Event] = {}
        self.grasp_results: Dict[str, Dict] = {}
        self._latest_vision_image: Optional[RosImage] = None
        self._latest_vision_image_time = 0.0
        self._vision_image_lock = threading.Lock()
        self._latest_api_detection_image_header = None
        self._last_api_detection_raw_json = {}
        self._vision_window_shutdown = threading.Event()
        self._vision_display_lock = threading.Lock()
        self._vision_display_image = None
        self._vision_latest_display_image = None
        self._vision_display_status = 'waiting for camera image...'
        self._vision_display_hold_until = 0.0
        self._vision_display_thread = None

        self._declare_parameters()
        self._read_parameters()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.target_command_pub = self.create_publisher(String, self.target_command_topic, 10)
        self.api_detections_pub = self.create_publisher(String, self.api_detections_topic, 10)
        self.create_subscription(
            String,
            self.detected_objects_topic,
            self.detected_objects_callback,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            self.grasp_result_topic,
            self.grasp_result_callback,
            10,
            callback_group=self.callback_group,
        )
        self._vision_image_sub = None
        if self.vision_enabled:
            image_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._vision_image_sub = self.create_subscription(
                RosImage,
                self.vision_image_topic,
                self._vision_image_callback,
                image_qos,
                callback_group=self.callback_group,
            )
            if self.vision_show_window:
                self._vision_display_thread = threading.Thread(
                    target=self._vision_display_loop,
                    daemon=True,
                )
                self._vision_display_thread.start()
        self.yolo_detect_once_client = self.create_client(
            Trigger,
            self.yolo_detect_once_service,
            callback_group=self.callback_group,
        )

        self.go_home_srv = self.create_service(
            Trigger,
            '~/go_home',
            self.go_home_callback,
            callback_group=self.callback_group,
        )
        self.list_tools_srv = self.create_service(
            ListTools,
            '~/list_tools',
            self.list_tools_callback,
            callback_group=self.callback_group,
        )
        self.call_tool_srv = self.create_service(
            CallTool,
            '~/call_tool',
            self.call_tool_callback,
            callback_group=self.callback_group,
        )
        self.move_x_srv = self.create_service(
            MoveAxis,
            '~/move_x_cm',
            lambda request, response: self.move_axis_callback('x', request, response),
            callback_group=self.callback_group,
        )
        self.move_y_srv = self.create_service(
            MoveAxis,
            '~/move_y_cm',
            lambda request, response: self.move_axis_callback('y', request, response),
            callback_group=self.callback_group,
        )
        self.move_z_srv = self.create_service(
            MoveAxis,
            '~/move_z_cm',
            lambda request, response: self.move_axis_callback('z', request, response),
            callback_group=self.callback_group,
        )
        self.list_yolo_objects_srv = self.create_service(
            Trigger,
            '~/list_yolo_objects',
            self.list_yolo_objects_callback,
            callback_group=self.callback_group,
        )
        self.grab_object_srv = self.create_service(
            ObjectName,
            '~/grab_object',
            self.grab_object_callback,
            callback_group=self.callback_group,
        )
        self.open_gripper_srv = self.create_service(
            Trigger,
            '~/open_gripper',
            self.open_gripper_callback,
            callback_group=self.callback_group,
        )
        self.close_gripper_srv = self.create_service(
            Trigger,
            '~/close_gripper',
            self.close_gripper_callback,
            callback_group=self.callback_group,
        )

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.trajectory_action,
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
        self.gripper_move_clients = [
            (
                action_name,
                ActionClient(
                    self,
                    GripperMove,
                    action_name,
                    callback_group=self.callback_group,
                ),
            )
            for action_name in self.gripper_move_actions
        ]
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

        self._publish_status(
            'mcp_server ready. services='
            '~/list_tools, ~/call_tool, ~/go_home, ~/move_x_cm, ~/move_y_cm, ~/move_z_cm, '
            '~/list_yolo_objects, ~/grab_object, '
            '~/open_gripper, ~/close_gripper; '
            f'base_frame={self.base_frame}, ee={self.end_effector_frame}, '
            f'vision_topic={self.vision_image_topic}, api_detections={self.api_detections_topic}'
        )

    def destroy_node(self) -> bool:
        self._vision_window_shutdown.set()
        if self._vision_display_thread is not None:
            self._vision_display_thread.join(timeout=1.0)
        return super().destroy_node()

    def _declare_parameters(self) -> None:
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
        self.declare_parameter('home_joint_positions_deg', [74.0, -3.0, -7.0, -115.0, -1.0, 110.0, 22.0])
        self.declare_parameter('home_move_duration_sec', 4.0)
        self.declare_parameter('home_trajectory_dt_sec', 0.05)
        self.declare_parameter('home_trajectory_start_delay_sec', 0.1)
        self.declare_parameter('home_wait_for_joint_state_sec', 3.0)
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('end_effector_frame', 'fr3_hand_tcp')
        self.declare_parameter('move_group_name', 'fr3_arm')
        self.declare_parameter('ik_service', '/compute_ik')
        self.declare_parameter('cartesian_path_service', '/compute_cartesian_path')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('execution_mode', 'move_group')
        self.declare_parameter('axis_move_execution_mode', 'cartesian')
        self.declare_parameter('plan_only', False)
        self.declare_parameter('avoid_collisions', True)
        self.declare_parameter('ik_timeout_sec', 0.5)
        self.declare_parameter('service_wait_timeout_sec', 5.0)
        self.declare_parameter('action_wait_timeout_sec', 10.0)
        self.declare_parameter('motion_duration_sec', 4.0)
        self.declare_parameter('max_single_axis_move_cm', 10.0)
        self.declare_parameter('goal_joint_tolerance', 0.01)
        self.declare_parameter('max_velocity_scaling', 0.05)
        self.declare_parameter('max_acceleration_scaling', 0.05)
        self.declare_parameter('num_planning_attempts', 5)
        self.declare_parameter('allowed_planning_time', 5.0)
        self.declare_parameter('axis_cartesian_max_step_m', 0.002)
        self.declare_parameter('axis_cartesian_jump_threshold', 0.0)
        self.declare_parameter('axis_cartesian_min_fraction', 0.99)
        self.declare_parameter('axis_cartesian_duration_per_cm_sec', 0.6)
        self.declare_parameter('axis_cartesian_min_duration_sec', 2.0)
        self.declare_parameter('gripper_move_actions', [
            '/franka_gripper/move',
            '/fr3_gripper/move',
            '/left_fr3_gripper/move',
        ])
        self.declare_parameter('gripper_grasp_actions', [
            '/franka_gripper/grasp',
            '/fr3_gripper/grasp',
            '/left_fr3_gripper/grasp',
        ])
        self.declare_parameter('gripper_server_wait_timeout_sec', 1.0)
        self.declare_parameter('gripper_action_timeout_sec', 10.0)
        self.declare_parameter('gripper_open_width_m', 0.08)
        self.declare_parameter('gripper_open_speed_mps', 0.05)
        self.declare_parameter('gripper_close_width_m', 0.0)
        self.declare_parameter('gripper_close_speed_mps', 0.03)
        self.declare_parameter('gripper_close_force_n', 50.0)
        self.declare_parameter('gripper_grasp_epsilon_inner_m', 0.01)
        self.declare_parameter('gripper_grasp_epsilon_outer_m', 0.08)
        self.declare_parameter('detected_objects_topic', '/object_target_controller/detected_objects')
        self.declare_parameter('target_command_topic', '/object_target_controller/target_class_name')
        self.declare_parameter('api_detections_topic', '/mcp_omni_client/api_detections_json')
        self.declare_parameter('grasp_result_topic', '/economic_grasp_roi/grasp_result')
        self.declare_parameter('yolo_detect_once_service', '/yolo/detect_once')
        self.declare_parameter('yolo_detection_wait_timeout_sec', 5.0)
        self.declare_parameter('target_command_settle_sec', 0.3)
        self.declare_parameter('detected_objects_max_age_sec', 2.0)
        self.declare_parameter('require_detected_before_grab', True)
        self.declare_parameter('require_target_command_subscriber', True)

        self.declare_parameter('omni_api_key_env', 'DASHSCOPE_API_KEY')
        self.declare_parameter('omni_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.declare_parameter('omni_text_model', 'qwen3.5-omni-plus')
        self.declare_parameter('omni_timeout', 90.0)
        self.declare_parameter('omni_max_tokens', 1000)
        self.declare_parameter('vision_enabled', True)
        self.declare_parameter('vision_image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('vision_image_max_age_sec', 30.0)
        self.declare_parameter('vision_max_image_width', 640)
        self.declare_parameter('vision_jpeg_quality', 85)
        self.declare_parameter('vision_show_window', True)
        self.declare_parameter('vision_window_name', 'Qwen-Omni Vision Box')
        self.declare_parameter('vision_save_images', True)
        self.declare_parameter('vision_output_dir', '/home/tqq/TQQ_ws/omni_vision_outputs')
        self.declare_parameter('vision_result_hold_sec', 60.0)
        self.declare_parameter('api_detection_default_confidence', 0.90)
        self.declare_parameter('api_detection_publish_settle_sec', 0.25)
        self.declare_parameter('api_detection_republish_count', 3)
        self.declare_parameter('api_detection_republish_interval_sec', 0.15)
        self.declare_parameter('api_detection_box_coordinate_space', 'qwen_1000')
        self.declare_parameter('api_detection_box_reference_size', 1000.0)
        self.declare_parameter('grab_api_default_motion_speed', 0.05)
        self.declare_parameter('grab_api_wait_for_result', True)
        self.declare_parameter('grab_api_result_timeout_sec', 180.0)

    def _read_parameters(self) -> None:
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.trajectory_action = str(self.get_parameter('trajectory_action').value)
        self.joint_names = [str(name) for name in self.get_parameter('joint_names').value]
        self.home_joint_positions_deg = [
            float(value) for value in self.get_parameter('home_joint_positions_deg').value
        ]
        self.home_move_duration_sec = float(self.get_parameter('home_move_duration_sec').value)
        self.home_trajectory_dt_sec = float(self.get_parameter('home_trajectory_dt_sec').value)
        self.home_trajectory_start_delay_sec = float(
            self.get_parameter('home_trajectory_start_delay_sec').value
        )
        self.home_wait_for_joint_state_sec = float(
            self.get_parameter('home_wait_for_joint_state_sec').value
        )
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.end_effector_frame = str(self.get_parameter('end_effector_frame').value)
        self.move_group_name = str(self.get_parameter('move_group_name').value)
        self.ik_service = str(self.get_parameter('ik_service').value)
        self.cartesian_path_service = str(self.get_parameter('cartesian_path_service').value)
        self.move_group_action = str(self.get_parameter('move_group_action').value)
        self.execution_mode = str(self.get_parameter('execution_mode').value).strip().lower()
        self.axis_move_execution_mode = str(
            self.get_parameter('axis_move_execution_mode').value
        ).strip().lower()
        self.plan_only = self._as_bool(self.get_parameter('plan_only').value)
        self.avoid_collisions = self._as_bool(self.get_parameter('avoid_collisions').value)
        self.ik_timeout_sec = float(self.get_parameter('ik_timeout_sec').value)
        self.service_wait_timeout_sec = float(self.get_parameter('service_wait_timeout_sec').value)
        self.action_wait_timeout_sec = float(self.get_parameter('action_wait_timeout_sec').value)
        self.motion_duration_sec = float(self.get_parameter('motion_duration_sec').value)
        self.max_single_axis_move_cm = float(self.get_parameter('max_single_axis_move_cm').value)
        self.goal_joint_tolerance = float(self.get_parameter('goal_joint_tolerance').value)
        self.max_velocity_scaling = float(self.get_parameter('max_velocity_scaling').value)
        self.max_acceleration_scaling = float(self.get_parameter('max_acceleration_scaling').value)
        self.num_planning_attempts = int(self.get_parameter('num_planning_attempts').value)
        self.allowed_planning_time = float(self.get_parameter('allowed_planning_time').value)
        self.axis_cartesian_max_step_m = max(
            0.001,
            float(self.get_parameter('axis_cartesian_max_step_m').value),
        )
        self.axis_cartesian_jump_threshold = float(
            self.get_parameter('axis_cartesian_jump_threshold').value
        )
        self.axis_cartesian_min_fraction = min(
            1.0,
            max(0.0, float(self.get_parameter('axis_cartesian_min_fraction').value)),
        )
        self.axis_cartesian_duration_per_cm_sec = max(
            0.05,
            float(self.get_parameter('axis_cartesian_duration_per_cm_sec').value),
        )
        self.axis_cartesian_min_duration_sec = max(
            0.1,
            float(self.get_parameter('axis_cartesian_min_duration_sec').value),
        )
        self.gripper_move_actions = self._string_list(
            self.get_parameter('gripper_move_actions').value
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
        self.gripper_open_width_m = float(self.get_parameter('gripper_open_width_m').value)
        self.gripper_open_speed_mps = float(self.get_parameter('gripper_open_speed_mps').value)
        self.gripper_close_width_m = float(self.get_parameter('gripper_close_width_m').value)
        self.gripper_close_speed_mps = float(self.get_parameter('gripper_close_speed_mps').value)
        self.gripper_close_force_n = float(self.get_parameter('gripper_close_force_n').value)
        self.gripper_grasp_epsilon_inner_m = float(
            self.get_parameter('gripper_grasp_epsilon_inner_m').value
        )
        self.gripper_grasp_epsilon_outer_m = float(
            self.get_parameter('gripper_grasp_epsilon_outer_m').value
        )
        self.detected_objects_topic = str(self.get_parameter('detected_objects_topic').value)
        self.target_command_topic = str(self.get_parameter('target_command_topic').value)
        self.api_detections_topic = str(self.get_parameter('api_detections_topic').value).strip()
        self.grasp_result_topic = str(self.get_parameter('grasp_result_topic').value).strip()
        self.yolo_detect_once_service = str(self.get_parameter('yolo_detect_once_service').value)
        self.yolo_detection_wait_timeout_sec = float(
            self.get_parameter('yolo_detection_wait_timeout_sec').value
        )
        self.target_command_settle_sec = max(
            0.0,
            float(self.get_parameter('target_command_settle_sec').value),
        )
        self.detected_objects_max_age_sec = float(
            self.get_parameter('detected_objects_max_age_sec').value
        )
        self.require_detected_before_grab = self._as_bool(
            self.get_parameter('require_detected_before_grab').value
        )
        self.require_target_command_subscriber = self._as_bool(
            self.get_parameter('require_target_command_subscriber').value
        )
        self.omni_api_key_env = str(self.get_parameter('omni_api_key_env').value).strip()
        self.omni_base_url = str(self.get_parameter('omni_base_url').value).strip()
        self.omni_text_model = str(self.get_parameter('omni_text_model').value).strip()
        self.omni_timeout = float(self.get_parameter('omni_timeout').value)
        self.omni_max_tokens = int(self.get_parameter('omni_max_tokens').value)
        self.vision_enabled = self._as_bool(self.get_parameter('vision_enabled').value)
        self.vision_image_topic = str(self.get_parameter('vision_image_topic').value).strip()
        self.vision_image_max_age_sec = float(
            self.get_parameter('vision_image_max_age_sec').value
        )
        self.vision_max_image_width = int(self.get_parameter('vision_max_image_width').value)
        self.vision_jpeg_quality = int(self.get_parameter('vision_jpeg_quality').value)
        self.vision_show_window = self._as_bool(self.get_parameter('vision_show_window').value)
        self.vision_window_name = str(self.get_parameter('vision_window_name').value).strip()
        self.vision_save_images = self._as_bool(self.get_parameter('vision_save_images').value)
        self.vision_output_dir = Path(
            str(self.get_parameter('vision_output_dir').value).strip()
        ).expanduser()
        self.vision_result_hold_sec = float(self.get_parameter('vision_result_hold_sec').value)
        self.api_detection_default_confidence = float(
            self.get_parameter('api_detection_default_confidence').value
        )
        self.api_detection_publish_settle_sec = max(
            0.0,
            float(self.get_parameter('api_detection_publish_settle_sec').value),
        )
        self.api_detection_republish_count = max(
            1,
            int(self.get_parameter('api_detection_republish_count').value),
        )
        self.api_detection_republish_interval_sec = max(
            0.0,
            float(self.get_parameter('api_detection_republish_interval_sec').value),
        )
        self.api_detection_box_coordinate_space = str(
            self.get_parameter('api_detection_box_coordinate_space').value
        ).strip().lower()
        self.api_detection_box_reference_size = max(
            1.0,
            float(self.get_parameter('api_detection_box_reference_size').value),
        )
        self.grab_api_default_motion_speed = min(
            1.0,
            max(0.0, float(self.get_parameter('grab_api_default_motion_speed').value)),
        )
        self.grab_api_wait_for_result = self._as_bool(
            self.get_parameter('grab_api_wait_for_result').value
        )
        self.grab_api_result_timeout_sec = max(
            1.0,
            float(self.get_parameter('grab_api_result_timeout_sec').value),
        )

        if len(self.joint_names) != len(self.home_joint_positions_deg):
            raise ValueError('joint_names and home_joint_positions_deg must have the same length')
        if self.execution_mode not in ('move_group', 'trajectory', 'ik_only'):
            raise ValueError('execution_mode must be move_group, trajectory, or ik_only')
        if self.axis_move_execution_mode not in ('cartesian', 'move_group', 'trajectory', 'ik_only'):
            raise ValueError(
                'axis_move_execution_mode must be cartesian, move_group, trajectory, or ik_only'
            )
        if self.max_single_axis_move_cm <= 0.0:
            raise ValueError('max_single_axis_move_cm must be greater than 0')
        if not self.gripper_move_actions:
            raise ValueError('gripper_move_actions must contain at least one action name')
        if not self.gripper_grasp_actions:
            raise ValueError('gripper_grasp_actions must contain at least one action name')

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'on'}
        return bool(value)

    @staticmethod
    def _string_list(value) -> List[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return [str(item).strip() for item in value or [] if str(item).strip()]

    @staticmethod
    def _normalize_object_name(name: str) -> str:
        text = str(name or '').strip().strip(' "\'“”‘’。，,.;；:：')
        return ' '.join(text.split())

    def _parse_grab_request(self, raw_name: str) -> Tuple[str, Optional[float]]:
        text = str(raw_name or '').strip()
        if not text:
            return '', None
        if not text.startswith('{'):
            return self._normalize_object_name(text), None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._normalize_object_name(text), None
        if not isinstance(data, dict):
            return self._normalize_object_name(text), None

        name = self._normalize_object_name(
            data.get('name') or data.get('object_name') or data.get('target') or ''
        )
        speed = self._optional_motion_speed(data)
        return name, speed

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

    @staticmethod
    def _target_command_payload(target_name: str, motion_speed: Optional[float]) -> str:
        return McpServer._api_target_command_payload(target_name, motion_speed, '')

    @staticmethod
    def _api_target_command_payload(
        target_name: str,
        motion_speed: Optional[float],
        request_id: str,
    ) -> str:
        target_name = str(target_name or '').strip()
        if not target_name:
            return ''
        if motion_speed is None and not request_id:
            return target_name
        payload = {'name': target_name}
        if motion_speed is not None:
            payload['motion_speed'] = motion_speed
        if request_id:
            payload['request_id'] = request_id
        return json.dumps(
            payload,
            ensure_ascii=False,
        )

    def _latest_detected_objects(self) -> Tuple[Optional[Dict], float]:
        with self.detected_objects_lock:
            payload = copy.deepcopy(self.latest_detected_objects_payload)
            stamp = self.latest_detected_objects_time
        if payload is None:
            return None, math.inf
        return payload, time.monotonic() - stamp

    def _detected_objects_stale(self, age_sec: float) -> bool:
        return self.detected_objects_max_age_sec > 0.0 and age_sec > self.detected_objects_max_age_sec

    def _wait_for_detected_objects_update(
        self,
        since_time: float,
        timeout_sec: float,
    ) -> Tuple[Optional[Dict], float]:
        deadline = time.monotonic() + max(0.1, timeout_sec)
        while time.monotonic() < deadline:
            with self.detected_objects_lock:
                payload = copy.deepcopy(self.latest_detected_objects_payload)
                stamp = self.latest_detected_objects_time
            if payload is not None and stamp >= since_time:
                return payload, time.monotonic() - stamp
            time.sleep(0.02)
        return None, math.inf

    def _trigger_yolo_detection(self, label: str) -> Tuple[bool, str]:
        if not self.yolo_detect_once_client.wait_for_service(
            timeout_sec=self.service_wait_timeout_sec,
        ):
            return False, (
                f'{label}: YOLO detect-once service unavailable: '
                f'{self.yolo_detect_once_service}; start yolo_realsense first'
            )

        future = self.yolo_detect_once_client.call_async(Trigger.Request())
        response = self._wait_for_future(
            future,
            self.yolo_detection_wait_timeout_sec,
            f'{label} YOLO detect_once',
        )
        if response is None:
            return False, f'{label}: YOLO detect_once timed out'
        if not response.success:
            return False, f'{label}: YOLO detect_once failed: {response.message}'
        return True, str(response.message)

    def _detect_and_wait_for_summary(self, label: str) -> Tuple[bool, str, Optional[Dict], float]:
        start_time = time.monotonic()
        ok, message = self._trigger_yolo_detection(label)
        if not ok:
            return False, message, None, math.inf

        payload, age = self._wait_for_detected_objects_update(
            start_time,
            self.yolo_detection_wait_timeout_sec,
        )
        if payload is None:
            return False, (
                f'{label}: YOLO ran once, but no fresh detected-object summary arrived on '
                f'{self.detected_objects_topic}; start object_target_controller first'
            ), None, math.inf
        return True, message, payload, age

    @staticmethod
    def _object_names_from_payload(payload: Optional[Dict]) -> List[str]:
        if not payload:
            return []
        objects = payload.get('objects', [])
        names = []
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                name = str(item.get('class_name', '')).strip()
                if name:
                    names.append(name)
        return names

    @staticmethod
    def _object_items_from_payload(payload: Optional[Dict]) -> List[Dict]:
        if not payload:
            return []
        objects = payload.get('objects', [])
        return [item for item in objects if isinstance(item, dict)] if isinstance(objects, list) else []

    @staticmethod
    def _contains_object_name(object_names: Sequence[str], target_name: str) -> bool:
        target_lower = target_name.lower()
        return any(str(name).strip().lower() == target_lower for name in object_names)

    @staticmethod
    def _format_object_names(object_names: Sequence[str]) -> str:
        if not object_names:
            return 'none'
        return ', '.join(str(name) for name in object_names)

    def _format_detected_objects_message(self, payload: Dict, age_sec: float) -> str:
        objects = self._object_items_from_payload(payload)
        if not objects:
            return f'YOLO detected no objects. age={age_sec:.2f}s'

        parts = []
        for item in objects:
            name = str(item.get('class_name', '')).strip() or str(item.get('class_id', 'object'))
            count = int(item.get('count', 1))
            confidence = float(item.get('best_confidence', item.get('confidence', 0.0)))
            parts.append(f'{name} x{count} confidence={confidence:.2f}')
        target = str(payload.get('target_class_name', '')).strip()
        suffix = f'; current_target={target}' if target else ''
        return f'YOLO detected: {", ".join(parts)}. age={age_sec:.2f}s{suffix}'

    def _resolve_object_name(self, requested_name: str, payload: Optional[Dict]) -> str:
        name = self._normalize_object_name(requested_name)
        if not name:
            return ''

        object_names = self._object_names_from_payload(payload)
        for candidate in object_names:
            if candidate.lower() == name.lower():
                return candidate
        return name

    def joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_state = copy.deepcopy(msg)

    def detected_objects_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'objects': [], 'raw': msg.data}

        with self.detected_objects_lock:
            self.latest_detected_objects_payload = payload
            self.latest_detected_objects_time = time.monotonic()

    def grasp_result_callback(self, msg: String) -> None:
        try:
            payload = json.loads(str(msg.data or '{}'))
        except json.JSONDecodeError:
            payload = {'success': False, 'message': str(msg.data or '')}
        if not isinstance(payload, dict):
            payload = {'success': False, 'message': str(payload)}
        request_id = str(payload.get('request_id') or '').strip()
        if not request_id:
            return
        with self.grasp_results_lock:
            self.grasp_results[request_id] = payload
            event = self.grasp_result_events.get(request_id)
        if event is not None:
            event.set()

    def go_home_callback(self, request, response):
        if not self.motion_lock.acquire(blocking=False):
            response.success = False
            response.message = 'motion already running'
            return response
        try:
            response.success, response.message = self._go_home()
            return response
        finally:
            self.motion_lock.release()

    def list_tools_callback(self, request, response: ListTools.Response):
        del request
        tools = self._tool_schema()
        response.success = True
        response.message = f'{len(tools)} tools available from mcp_server'
        response.tools_json = json.dumps(tools, ensure_ascii=False)
        return response

    def call_tool_callback(self, request: CallTool.Request, response: CallTool.Response):
        name = str(request.name or '').strip()
        try:
            arguments = json.loads(str(request.arguments_json or '{}'))
        except json.JSONDecodeError as exc:
            response.success = False
            response.message = f'call_tool failed: invalid arguments_json: {exc}'
            response.result_json = '{}'
            return response
        if not isinstance(arguments, dict):
            arguments = {}

        self._publish_status(
            f'call_tool name={name} args={json.dumps(arguments, ensure_ascii=False)}'
        )
        try:
            success, message, result = self._call_tool_by_name(name, arguments)
        except Exception as exc:
            self.get_logger().exception(f'call_tool crashed while running {name}: {exc}')
            success = False
            message = f'{name or "call_tool"} failed: {exc}'
            result = {}

        response.success = bool(success)
        response.message = message
        response.result_json = json.dumps(result or {}, ensure_ascii=False)
        return response

    def _call_tool_by_name(self, name: str, arguments: Dict) -> Tuple[bool, str, Dict]:
        if name == 'look_camera':
            return self._tool_look_camera(arguments)
        if name == 'go_home':
            return self._run_locked_motion('go_home', self._go_home)
        if name in ('list_api_objects', 'list_yolo_objects'):
            if name == 'list_yolo_objects':
                return self._tool_list_yolo_objects()
            return self._tool_list_api_objects(arguments)
        if name == 'box_api_object':
            return self._tool_box_api_object(arguments)
        if name in ('grab_api_object', 'grab_yolo_object'):
            if name == 'grab_yolo_object':
                return self._tool_grab_yolo_object(arguments)
            return self._tool_grab_api_object(arguments)
        if name == 'open_gripper':
            return self._run_locked_gripper('open_gripper', self._open_gripper)
        if name == 'close_gripper':
            return self._run_locked_gripper('close_gripper', self._close_gripper)
        if name == 'move_x_cm':
            return self._tool_move_axis('x', arguments)
        if name == 'move_y_cm':
            return self._tool_move_axis('y', arguments)
        if name == 'move_z_cm':
            return self._tool_move_axis('z', arguments)
        return False, f'unsupported mcp tool: {name or "empty"}', {}

    def _run_locked_motion(self, label: str, function) -> Tuple[bool, str, Dict]:
        if not self.motion_lock.acquire(blocking=False):
            return False, f'{label} failed: motion already running', {}
        try:
            success, message = function()
            return bool(success), f'{label} {"success" if success else "failed"}: {message}', {}
        finally:
            self.motion_lock.release()

    def _run_locked_gripper(self, label: str, function) -> Tuple[bool, str, Dict]:
        if not self.gripper_lock.acquire(blocking=False):
            return False, f'{label} failed: gripper action already running', {}
        try:
            success, message = function()
            return bool(success), f'{label} {"success" if success else "failed"}: {message}', {}
        finally:
            self.gripper_lock.release()

    def _tool_move_axis(self, axis: str, arguments: Dict) -> Tuple[bool, str, Dict]:
        try:
            centimeters = float(arguments.get('centimeters', 0.0))
        except (TypeError, ValueError):
            return False, f'move_{axis}_cm failed: centimeters must be a number', {}
        if not self.motion_lock.acquire(blocking=False):
            return False, f'move_{axis}_cm failed: motion already running', {}
        try:
            success, message = self._move_axis(axis, centimeters)
            return (
                bool(success),
                f'move_{axis}_cm {"success" if success else "failed"}: {message}',
                {'axis': axis, 'centimeters': centimeters},
            )
        finally:
            self.motion_lock.release()

    def _tool_list_yolo_objects(self) -> Tuple[bool, str, Dict]:
        self._clear_target_commands_before_detection()
        ok, message, payload, age = self._detect_and_wait_for_summary('list_yolo_objects')
        if not ok:
            return False, f'list_yolo_objects failed: {message}', {}
        if payload is None:
            return False, (
                f'list_yolo_objects failed: no detected objects received yet on '
                f'{self.detected_objects_topic}; start RealSense, YOLO, and '
                'object_target_controller first'
            ), {}
        if self._detected_objects_stale(age):
            return False, (
                f'list_yolo_objects failed: detected objects are stale: '
                f'age={age:.2f}s exceeds {self.detected_objects_max_age_sec:.2f}s'
            ), {'payload': payload, 'age_sec': age}

        return True, (
            'list_yolo_objects success: ' + self._format_detected_objects_message(payload, age)
        ), {'payload': payload, 'age_sec': age}

    def _tool_grab_yolo_object(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        requested_name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        speed = self._optional_motion_speed(arguments)
        if speed is None:
            request_name = requested_name
        else:
            request_name = json.dumps(
                {'name': requested_name, 'motion_speed': speed},
                ensure_ascii=False,
            )
        request = ObjectName.Request()
        request.name = request_name
        response = ObjectName.Response()
        response = self.grab_object_callback(request, response)
        return (
            bool(response.success),
            f'grab_yolo_object {"success" if response.success else "failed"}: {response.message}',
            {'object_name': requested_name, 'motion_speed': speed},
        )

    def _tool_look_camera(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        if not self.vision_enabled:
            return False, 'look_camera failed: vision is disabled', {}
        question = str(arguments.get('question') or '').strip() or '请描述当前相机画面。'
        try:
            answer = self._ask_vision_model(question)
        except Exception as exc:
            return False, f'look_camera failed: {exc}', {}
        return True, f'look_camera success: {answer}', {'answer': answer}

    def _ask_vision_model(self, question: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        image_payload = self._latest_camera_jpeg_payload()
        prompt = (
            f'{question}\n\n'
            f'图像发送尺寸是 {image_payload["api_width"]}x{image_payload["api_height"]} 像素。'
            f'原始 ROS RGB 图像尺寸是 {image_payload["original_width"]}x'
            f'{image_payload["original_height"]} 像素。'
            '如果需要给出像素坐标，必须使用原始 ROS RGB 图像坐标，'
            'x 从左到右，y 从上到下，不要使用归一化坐标。'
        )
        self._publish_status(
            f'omni_vision_call model={self.omni_text_model} topic={self.vision_image_topic} '
            f'api_size={image_payload["api_width"]}x{image_payload["api_height"]} '
            f'original_size={image_payload["original_width"]}x{image_payload["original_height"]}'
        )
        client = OpenAI(
            api_key=self._api_key_from_env(self.omni_api_key_env),
            base_url=self.omni_base_url,
            timeout=self.omni_timeout,
        )
        response = client.chat.completions.create(
            model=self.omni_text_model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        '你是机器人相机视觉助手。只根据用户问题和当前图像回答，'
                        '不要调用机器人控制工具，不要编造抓取成功。'
                    ),
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_payload['data_url']}},
                    ],
                },
            ],
            modalities=['text'],
            stream=False,
            max_tokens=self.omni_max_tokens,
        )
        return str(response.choices[0].message.content or '').strip() or '没有得到视觉回复。'

    def _tool_list_api_objects(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        if not self.vision_enabled:
            return False, 'list_api_objects failed: vision is disabled', {}
        self._publish_api_target_command('', None)
        question = str(arguments.get('question') or '').strip()
        if not question:
            question = '请列出当前画面里清晰可见、适合机械臂抓取的物体。'
        try:
            detections = self._detect_objects_with_vision_api(question, target_name='', max_results=0)
        except Exception as exc:
            return False, f'list_api_objects failed: {exc}', {}
        if not detections:
            return True, 'list_api_objects success: 当前视觉 API 没有返回可抓取目标。', {
                'detections': [],
            }
        output_path = self._save_and_show_api_detection_result(
            detections,
            answer={'objects': detections},
            mode='list_api_objects',
            status='Qwen-Omni API vision result',
        )
        parts = []
        for item in detections:
            name = str(item.get('class_name', '')).strip() or 'object'
            confidence = float(item.get('confidence', 0.0))
            bbox = item.get('bbox_xyxy', [])
            if len(bbox) == 4:
                parts.append(
                    f'{name} confidence={confidence:.2f} '
                    f'bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]'
                )
            else:
                parts.append(f'{name} confidence={confidence:.2f}')
        return True, (
            'list_api_objects success: vision API detected: ' + ', '.join(parts)
        ), {'detections': detections, 'saved': output_path}

    def _tool_box_api_object(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        requested_name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        if not requested_name:
            return False, 'box_api_object failed: object name is empty', {}
        if not self.vision_enabled:
            return False, 'box_api_object failed: vision is disabled', {}

        is_all_objects = self._is_all_objects_box_request(requested_name)
        if is_all_objects:
            question = '请框出当前画面中所有清晰可见、适合机械臂抓取或识别的独立物体。'
            target_name = ''
        else:
            question = f'请框出画面中的“{requested_name}”。如果有多个符合的目标，请分别返回多个框。'
            target_name = requested_name

        try:
            detections = self._detect_objects_with_vision_api(
                question,
                target_name=target_name,
                max_results=0,
            )
        except Exception as exc:
            return False, f'box_api_object failed: {exc}', {}
        if not detections:
            return False, f'box_api_object failed: 视觉 API 没有找到“{requested_name}”。', {}

        output_path = self._save_and_show_api_detection_result(
            detections,
            answer={'objects': detections},
            mode='box_api_object',
            status='Qwen-Omni box result',
        )
        parts = []
        for item in detections:
            name = str(item.get('class_name', '')).strip() or requested_name
            bbox = item.get('bbox_xyxy', [])
            if len(bbox) == 4:
                parts.append(
                    f'{name} bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]'
                )
            else:
                parts.append(name)
        target_label = '所有可见物体' if is_all_objects else requested_name
        return True, (
            f'box_api_object success: 已在 Vision Box 框出 {len(detections)} 个'
            f'“{target_label}”：' + '，'.join(parts)
            + f'；saved={output_path or "disabled"}'
        ), {'detections': detections, 'saved': output_path}

    @staticmethod
    def _is_all_objects_box_request(text: str) -> bool:
        compact = ''.join(str(text or '').lower().split())
        if not compact:
            return False
        return compact in (
            'all',
            'allobjects',
            'everything',
            'objects',
            '所有',
            '全部',
            '所有物体',
            '全部物体',
            '所有目标',
            '全部目标',
            '所有东西',
            '全部东西',
            '可见物体',
            '画面中物体',
            '画面里的物体',
            '你看到的物体',
            '能看到的物体',
        )

    def _tool_grab_api_object(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        requested_name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        if not requested_name:
            return False, 'grab_api_object failed: object name is empty', {}
        if not self.vision_enabled:
            return False, 'grab_api_object failed: vision is disabled', {}
        if self.target_command_pub.get_subscription_count() < 1:
            return False, (
                'grab_api_object failed: no subscriber on '
                f'{self.target_command_topic}; start roi_economic_grasp_controller first'
            ), {}
        if self.api_detections_pub.get_subscription_count() < 1:
            return False, (
                'grab_api_object failed: no subscriber on '
                f'{self.api_detections_topic}; start roi_economic_grasp_controller first'
            ), {}

        try:
            detections = self._detect_objects_with_vision_api(
                f'请只定位最符合“{requested_name}”的一个目标。',
                target_name=requested_name,
            )
        except Exception as exc:
            return False, f'grab_api_object failed: {exc}', {}
        if not detections:
            return False, f'grab_api_object failed: 视觉 API 没有找到“{requested_name}”。', {}

        detection = detections[0]
        target_name = str(detection.get('class_name', '')).strip() or requested_name
        request_id = f'grab_{uuid.uuid4().hex}'
        speed = self._optional_motion_speed(arguments)
        if speed is None:
            speed = self.grab_api_default_motion_speed
        self._prepare_grasp_result_wait(request_id)
        self._publish_api_target_command('', None)
        if self.api_detection_publish_settle_sec > 0.0:
            time.sleep(self.api_detection_publish_settle_sec)
        self._publish_api_target_command(target_name, speed, request_id=request_id)
        if self.api_detection_publish_settle_sec > 0.0:
            time.sleep(self.api_detection_publish_settle_sec)
        payload = self._api_detection_payload([detection])
        self._publish_api_detection_payload(payload)
        output_path = self._save_and_show_api_detection_result(
            [detection],
            answer=payload,
            mode='grab_api_object',
            status=f'Qwen-Omni target: {self._ascii_for_cv_text(target_name) or "object"}',
        )

        bbox = detection.get('bbox_xyxy', [0.0, 0.0, 0.0, 0.0])
        message = (
            f'grab_api_object command sent: requested="{requested_name}", '
            f'target="{target_name}", motion_speed={speed if speed is not None else "default"}, '
            f'bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]. '
            f'published detections to {self.api_detections_topic}; '
            f'saved={output_path or "disabled"}. '
            'roi_economic_grasp_controller will use the API bbox as ROI, segment RGB-D points, '
            'run EconomicGrasp, then use MoveIt IK and execute the motion.'
        )
        self._publish_status(message)
        if not self.grab_api_wait_for_result:
            return True, f'grab_api_object success: {message}', {
                'request_id': request_id,
                'requested_name': requested_name,
                'target_name': target_name,
                'motion_speed': speed,
                'detection': detection,
                'saved': output_path,
            }

        result = self._wait_for_grasp_result(request_id, self.grab_api_result_timeout_sec)
        if result is None:
            self._forget_grasp_result_wait(request_id)
            return False, (
                f'grab_api_object failed: timed out waiting {self.grab_api_result_timeout_sec:.1f}s '
                f'for EconomicGrasp result request_id={request_id}'
            ), {
                'request_id': request_id,
                'requested_name': requested_name,
                'target_name': target_name,
                'motion_speed': speed,
                'detection': detection,
                'saved': output_path,
            }

        self._forget_grasp_result_wait(request_id)
        result_message = str(result.get('message') or '').strip()
        result_stage = str(result.get('stage') or '').strip()
        result_success = bool(result.get('success'))
        if not result_success:
            return False, (
                f'grab_api_object failed: EconomicGrasp result request_id={request_id} '
                f'stage={result_stage or "unknown"}: {result_message or "failed"}'
            ), {
                'request_id': request_id,
                'requested_name': requested_name,
                'target_name': target_name,
                'motion_speed': speed,
                'detection': detection,
                'saved': output_path,
                'grasp_result': result,
            }

        return True, (
            f'grab_api_object success: EconomicGrasp completed request_id={request_id} '
            f'target="{target_name}" stage={result_stage or "completed"}: '
            f'{result_message or "grasp completed"}'
        ), {
            'request_id': request_id,
            'requested_name': requested_name,
            'target_name': target_name,
            'motion_speed': speed,
            'detection': detection,
            'saved': output_path,
            'grasp_result': result,
        }

    def _detect_objects_with_vision_api(
        self,
        question: str,
        target_name: str = '',
        max_results: int = 1,
    ) -> List[Dict]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        image_payload = self._latest_camera_jpeg_payload()
        self._latest_api_detection_image_header = {
            'stamp': {
                'sec': int(image_payload.get('stamp_sec', 0)),
                'nanosec': int(image_payload.get('stamp_nanosec', 0)),
            },
            'frame_id': str(image_payload.get('frame_id', '')),
        }
        width = int(image_payload['original_width'])
        height = int(image_payload['original_height'])
        if target_name:
            if max_results > 0:
                task = (
                    f'目标物体是“{target_name}”。只返回最适合抓取的一个目标框；'
                    '如果没看到这个目标，objects 返回空数组。'
                )
            else:
                task = (
                    f'目标描述是“{target_name}”。返回所有符合这个描述的独立目标框；'
                    '如果没看到符合目标，objects 返回空数组。'
                )
        else:
            task = (
                '返回当前画面中清晰可见、适合机械臂抓取的物体列表。'
                '同类多个物体要分别给多个框，不要用一个大框包住多个物体。'
            )
        prompt = (
            f'{question}\n'
            f'{task}\n'
            f'原始图像尺寸是 width={width}, height={height}。'
            '如果返回 bbox_xyxy，请使用 0 到 1000 的视觉坐标系：'
            'xmin,ymin,xmax,ymax 都是相对于 1000x1000 图像的整数。'
            '请估计每个物体的轴对齐矩形框，尽量贴合物体外轮廓。'
            'class_name 必须是英文小写名词，例如 orange/apple/cup。'
            '只输出严格 JSON，不要 Markdown，不要解释，不要思考过程。'
            'JSON 格式：'
            '{"objects":[{"class_name":"english_label","label_zh":"中文名",'
            '"confidence":0.0到1.0,"bbox_xyxy":[xmin,ymin,xmax,ymax]}]}'
        )
        self._publish_status(
            f'omni_api_detection_call model={self.omni_text_model} target={target_name or "list"} '
            f'topic={self.vision_image_topic} size={width}x{height}'
        )
        client = OpenAI(
            api_key=self._api_key_from_env(self.omni_api_key_env),
            base_url=self.omni_base_url,
            timeout=self.omni_timeout,
        )
        response = client.chat.completions.create(
            model=self.omni_text_model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        '你是机器人抓取视觉检测器。你的唯一任务是根据图像返回物体检测框 JSON。'
                        '不要生成图片，不要描述过程，不要输出 JSON 之外的任何文字。'
                    ),
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_payload['data_url']}},
                    ],
                },
            ],
            modalities=['text'],
            stream=False,
            max_tokens=min(max(300, self.omni_max_tokens), 1200),
            response_format={'type': 'json_object'},
        )
        content = str(response.choices[0].message.content or '').strip()
        data = self._load_json_object(content)
        detections = self._detections_from_api_json(data, width, height)
        self._last_api_detection_raw_json = data
        if target_name and max_results > 0 and detections:
            return detections[:max_results]
        return detections

    def _load_json_object(self, content: str) -> Dict:
        text = str(content or '').strip()
        if not text:
            raise RuntimeError('vision API returned empty JSON')
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start < 0 or end <= start:
                raise RuntimeError(f'vision API did not return JSON: {text[:160]}')
            data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise RuntimeError('vision API JSON root is not an object')
        return data

    def _detections_from_api_json(self, data: Dict, width: int, height: int) -> List[Dict]:
        objects = data.get('objects', [])
        if isinstance(objects, dict):
            objects = [objects]
        if not isinstance(objects, list):
            objects = []

        detections = []
        for index, item in enumerate(objects):
            if not isinstance(item, dict):
                continue
            class_name = str(
                item.get('class_name')
                or item.get('label_en')
                or item.get('name_en')
                or item.get('label')
                or item.get('name')
                or 'object'
            ).strip().lower()
            if not class_name:
                class_name = 'object'
            bbox, normalized_bbox, raw_bbox, bbox_scale = self._bbox_from_api_item(item, width, height)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5
            confidence = self._api_confidence(item)
            detections.append({
                'class_id': index,
                'class_name': class_name,
                'label_zh': str(item.get('label_zh') or item.get('name_zh') or '').strip(),
                'confidence': confidence,
                'bbox_xyxy': [x1, y1, x2, y2],
                'bbox_normalized': normalized_bbox,
                'bbox_raw': raw_bbox,
                'bbox_scale': bbox_scale,
                'center_xy': [center_x, center_y],
                'center_xy_int': [int(round(center_x)), int(round(center_y))],
            })
        return detections

    def _bbox_from_api_item(
        self,
        item: Dict,
        width: int,
        height: int,
    ) -> Tuple[Optional[List[float]], List[float], List[float], str]:
        raw = item.get('bbox_xyxy') or item.get('xyxy')
        scale = self.api_detection_box_coordinate_space or 'qwen_1000'
        if raw is None:
            raw = item.get('bbox_percent') or item.get('box_percent')
            scale = 'percent'
        if raw is None:
            raw = item.get('bbox_normalized') or item.get('box_normalized')
            scale = 'normalized'
        if raw is None:
            raw = item.get('box') or item.get('bbox')
            scale = self._infer_bbox_scale(raw)
        if isinstance(raw, dict):
            raw = [
                raw.get('xmin', raw.get('x1')),
                raw.get('ymin', raw.get('y1')),
                raw.get('xmax', raw.get('x2')),
                raw.get('ymax', raw.get('y2')),
            ]
        if not isinstance(raw, list) or len(raw) != 4:
            return None, [], [], scale
        try:
            x1, y1, x2, y2 = [float(value) for value in raw]
        except (TypeError, ValueError):
            return None, [], [], scale
        raw_bbox = [x1, y1, x2, y2]

        if scale in ('qwen_1000', '1000', 'vlm_1000'):
            ref = self.api_detection_box_reference_size
            nx1 = x1 / ref
            ny1 = y1 / ref
            nx2 = x2 / ref
            ny2 = y2 / ref
            x1 = nx1 * width
            x2 = nx2 * width
            y1 = ny1 * height
            y2 = ny2 * height
        elif scale == 'pixel':
            nx1 = x1 / float(width)
            nx2 = x2 / float(width)
            ny1 = y1 / float(height)
            ny2 = y2 / float(height)
        elif scale == 'normalized':
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
        elif scale == 'percent':
            nx1 = x1 / 100.0
            nx2 = x2 / 100.0
            ny1 = y1 / 100.0
            ny2 = y2 / 100.0
            x1 = nx1 * width
            x2 = nx2 * width
            y1 = ny1 * height
            y2 = ny2 * height
        else:
            inferred = self._infer_bbox_scale(raw)
            return self._bbox_from_api_values(raw_bbox, inferred, width, height)

        normalized = [
            max(0.0, min(1.0, min(nx1, nx2))),
            max(0.0, min(1.0, min(ny1, ny2))),
            max(0.0, min(1.0, max(nx1, nx2))),
            max(0.0, min(1.0, max(ny1, ny2))),
        ]
        left = max(0.0, min(float(width - 1), min(x1, x2)))
        right = max(0.0, min(float(width - 1), max(x1, x2)))
        top = max(0.0, min(float(height - 1), min(y1, y2)))
        bottom = max(0.0, min(float(height - 1), max(y1, y2)))
        if right - left < 2.0 or bottom - top < 2.0:
            return None, normalized, raw_bbox, scale
        return [left, top, right, bottom], normalized, raw_bbox, scale

    def _bbox_from_api_values(
        self,
        raw_bbox: List[float],
        scale: str,
        width: int,
        height: int,
    ) -> Tuple[Optional[List[float]], List[float], List[float], str]:
        item = {'bbox_xyxy': raw_bbox}
        previous = self.api_detection_box_coordinate_space
        try:
            self.api_detection_box_coordinate_space = scale
            return self._bbox_from_api_item(item, width, height)
        finally:
            self.api_detection_box_coordinate_space = previous

    @staticmethod
    def _infer_bbox_scale(raw) -> str:
        values = []
        if isinstance(raw, dict):
            raw_values = [
                raw.get('xmin', raw.get('x1')),
                raw.get('ymin', raw.get('y1')),
                raw.get('xmax', raw.get('x2')),
                raw.get('ymax', raw.get('y2')),
            ]
        else:
            raw_values = raw if isinstance(raw, list) else []
        for value in raw_values:
            try:
                values.append(abs(float(value)))
            except (TypeError, ValueError):
                pass
        if len(values) == 4 and max(values) <= 1.5:
            return 'normalized'
        return 'pixel'

    def _api_confidence(self, item: Dict) -> float:
        raw = item.get('confidence', item.get('score', self.api_detection_default_confidence))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = self.api_detection_default_confidence
        return min(1.0, max(0.0, value))

    def _api_detection_payload(self, detections: List[Dict]) -> Dict:
        now = self.get_clock().now().to_msg()
        header = self._latest_api_detection_image_header
        if not isinstance(header, dict):
            header = {
                'stamp': {'sec': int(now.sec), 'nanosec': int(now.nanosec)},
                'frame_id': self.vision_image_topic,
            }
        return {
            'source': 'mcp_server_api',
            'header': header,
            'detections': detections,
        }

    def _publish_api_detection_payload(self, payload: Dict) -> None:
        text = json.dumps(payload, ensure_ascii=True)
        count = max(1, self.api_detection_republish_count)
        for index in range(count):
            self.api_detections_pub.publish(String(data=text))
            if index + 1 < count and self.api_detection_republish_interval_sec > 0.0:
                time.sleep(self.api_detection_republish_interval_sec)

    def _publish_api_target_command(
        self,
        target_name: str,
        motion_speed: Optional[float],
        request_id: str = '',
    ) -> None:
        target_name = str(target_name or '').strip()
        if not target_name:
            self.target_command_pub.publish(String(data=''))
            return
        payload = self._api_target_command_payload(target_name, motion_speed, request_id)
        self.target_command_pub.publish(String(data=payload))

    def _prepare_grasp_result_wait(self, request_id: str) -> None:
        if not request_id:
            return
        with self.grasp_results_lock:
            self.grasp_results.pop(request_id, None)
            self.grasp_result_events[request_id] = threading.Event()

    def _wait_for_grasp_result(self, request_id: str, timeout_sec: float) -> Optional[Dict]:
        if not request_id:
            return None
        with self.grasp_results_lock:
            event = self.grasp_result_events.get(request_id)
            existing = copy.deepcopy(self.grasp_results.get(request_id))
        if existing is not None:
            return existing
        if event is None:
            event = threading.Event()
            with self.grasp_results_lock:
                self.grasp_result_events[request_id] = event
        if not event.wait(timeout=max(0.1, timeout_sec)):
            return None
        with self.grasp_results_lock:
            return copy.deepcopy(self.grasp_results.get(request_id))

    def _forget_grasp_result_wait(self, request_id: str) -> None:
        if not request_id:
            return
        with self.grasp_results_lock:
            self.grasp_result_events.pop(request_id, None)
            self.grasp_results.pop(request_id, None)

    def _vision_image_callback(self, msg: RosImage) -> None:
        with self._vision_image_lock:
            self._latest_vision_image = msg
            self._latest_vision_image_time = time.monotonic()
        if self.vision_show_window:
            try:
                frame = self._ros_image_to_bgr_array(msg)
            except Exception as exc:
                self.get_logger().warn(f'Could not convert vision preview image: {exc}')
                return
            with self._vision_display_lock:
                self._vision_latest_display_image = frame
                if time.time() >= self._vision_display_hold_until:
                    self._vision_display_status = 'camera preview'

    def _vision_display_loop(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            self.get_logger().error(
                'vision_show_window is enabled but OpenCV/Numpy is not available. '
                f'Install python3-opencv python3-numpy. Details: {exc}'
            )
            return

        try:
            cv2.namedWindow(self.vision_window_name, cv2.WINDOW_NORMAL)
            while rclpy.ok() and not self._vision_window_shutdown.is_set():
                frame = self._vision_display_frame(np)
                cv2.imshow(self.vision_window_name, frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord('q')):
                    self._vision_window_shutdown.set()
                    break
            cv2.destroyWindow(self.vision_window_name)
        except Exception as exc:
            self.get_logger().error(f'Qwen-Omni vision display window failed: {exc}')

    def _vision_display_frame(self, np_module):
        now = time.time()
        with self._vision_display_lock:
            latest = (
                None
                if self._vision_latest_display_image is None
                else self._vision_latest_display_image.copy()
            )
            display = (
                None
                if self._vision_display_image is None
                else self._vision_display_image.copy()
            )
            status = self._vision_display_status
            hold_active = now < self._vision_display_hold_until

        if display is not None and hold_active:
            frame = display
        elif latest is not None:
            frame = latest
            status = status or 'camera preview'
        else:
            frame = np_module.zeros((480, 640, 3), dtype=np_module.uint8)
            status = status or 'waiting for camera image...'

        if status:
            self._draw_status_bar(frame, status)
        return frame

    def _set_vision_display_result(self, image, status: str) -> None:
        with self._vision_display_lock:
            self._vision_display_image = image.copy()
            self._vision_display_status = status
            self._vision_display_hold_until = time.time() + max(0.0, self.vision_result_hold_sec)

    @staticmethod
    def _ascii_for_cv_text(text: str, fallback: str = '') -> str:
        value = str(text or '')
        encoded = value.encode('ascii', errors='ignore').decode('ascii')
        encoded = ' '.join(encoded.split())
        return encoded or fallback

    @staticmethod
    def _draw_status_bar(image, text: str) -> None:
        try:
            import cv2
        except ImportError:
            return
        if image is None or getattr(image, 'size', 0) == 0:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        text = McpServer._ascii_for_cv_text(text, 'Qwen-Omni vision')
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 8
        cv2.rectangle(
            image,
            (0, 0),
            (min(image.shape[1], tw + pad * 2), th + baseline + pad * 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (pad, th + pad),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _save_and_show_api_detection_result(
        self,
        detections: List[Dict],
        answer,
        mode: str,
        status: str,
    ) -> str:
        try:
            image_msg = self._latest_camera_image_msg()
            bgr = self._ros_image_to_bgr_array(image_msg)
            result_bgr = self._draw_api_detections(bgr, detections)
            if self.vision_show_window:
                self._set_vision_display_result(result_bgr, status)
            return self._save_vision_outputs(
                image_payload=self._ros_image_to_jpeg_payload(image_msg),
                answer=answer,
                boxes=detections,
                mode=mode,
                annotated_bgr=result_bgr,
            )
        except Exception as exc:
            self.get_logger().warn(f'Could not save/show API detection result: {exc}')
            return ''

    def _save_vision_outputs(
        self,
        image_payload: Dict,
        answer,
        boxes: List[Dict],
        mode: str,
        annotated_bgr=None,
    ) -> str:
        if not self.vision_save_images:
            return ''
        try:
            self.vision_output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            prefix = self.vision_output_dir / f'{timestamp}_{uuid.uuid4().hex[:6]}'
            raw_path = prefix.with_name(prefix.name + '_raw.jpg')
            result_json_path = prefix.with_name(prefix.name + '_result.json')
            with open(raw_path, 'wb') as file_obj:
                file_obj.write(base64.b64decode(image_payload['jpeg_b64']))
            annotated_path = ''
            if annotated_bgr is not None:
                annotated_path = str(prefix.with_name(prefix.name + '_annotated.jpg'))
                self._write_bgr_image(annotated_path, annotated_bgr)
            result = {
                'mode': mode,
                'raw_image': str(raw_path),
                'annotated_image': annotated_path,
                'answer': answer,
                'boxes': boxes,
                'vision_image_topic': self.vision_image_topic,
                'image_width': image_payload.get('original_width'),
                'image_height': image_payload.get('original_height'),
                'api_width': image_payload.get('api_width'),
                'api_height': image_payload.get('api_height'),
                'stamp_sec': image_payload.get('stamp_sec'),
                'stamp_nanosec': image_payload.get('stamp_nanosec'),
                'frame_id': image_payload.get('frame_id'),
            }
            with open(result_json_path, 'w', encoding='utf-8') as file_obj:
                json.dump(result, file_obj, ensure_ascii=False, indent=2)
            return str(result_json_path)
        except Exception as exc:
            self.get_logger().warn(f'Could not save Omni vision outputs: {exc}')
            return ''

    def _draw_api_detections(self, bgr, detections: List[Dict]):
        try:
            import cv2
        except ImportError:
            return bgr
        image = bgr.copy()
        height, width = image.shape[:2]
        for index, detection in enumerate(detections):
            bbox = detection.get('bbox_xyxy', [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            color = self._box_color(index)
            label = str(detection.get('class_name') or 'object')
            confidence = detection.get('confidence', None)
            if confidence is not None:
                try:
                    label = f'{label} {float(confidence):.2f}'
                except (TypeError, ValueError):
                    pass
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
            label_y1 = max(0, y1 - th - baseline - 6)
            label_y2 = min(height - 1, label_y1 + th + baseline + 6)
            label_x2 = min(width - 1, x1 + tw + 8)
            cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 4, label_y2 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            center = detection.get('center_xy_int') or detection.get('center_xy')
            if center and len(center) == 2:
                cx = int(round(float(center[0])))
                cy = int(round(float(center[1])))
                if 0 <= cx < width and 0 <= cy < height:
                    cv2.drawMarker(
                        image,
                        (cx, cy),
                        (255, 255, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=24,
                        thickness=4,
                        line_type=cv2.LINE_AA,
                    )
                    cv2.drawMarker(
                        image,
                        (cx, cy),
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=24,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )
        return image

    @staticmethod
    def _box_color(index: int) -> Tuple[int, int, int]:
        palette = [
            (255, 56, 56),
            (255, 157, 151),
            (255, 112, 31),
            (255, 178, 29),
            (72, 249, 10),
            (0, 212, 187),
            (44, 153, 168),
            (100, 115, 255),
        ]
        return palette[index % len(palette)]

    def _latest_camera_jpeg_payload(self) -> Dict:
        image_msg = self._latest_camera_image_msg()
        return self._ros_image_to_jpeg_payload(image_msg)

    def _latest_camera_image_msg(self) -> RosImage:
        with self._vision_image_lock:
            image_msg = self._latest_vision_image
            image_time = self._latest_vision_image_time

        if image_msg is None:
            raise RuntimeError(f'No camera image received on {self.vision_image_topic}.')
        age = time.monotonic() - image_time
        if self.vision_image_max_age_sec > 0 and age > self.vision_image_max_age_sec:
            raise RuntimeError(
                f'Latest camera image is stale: {age:.1f}s old on {self.vision_image_topic}.'
            )
        return image_msg

    def _ros_image_to_jpeg_payload(self, msg: RosImage) -> Dict:
        image = self._ros_image_to_pil(msg)
        original_width = image.width
        original_height = image.height
        if self.vision_max_image_width > 0 and original_width > self.vision_max_image_width:
            ratio = self.vision_max_image_width / float(original_width)
            new_size = (self.vision_max_image_width, max(1, int(original_height * ratio)))
            from PIL import Image as PilImage
            resampling = getattr(getattr(PilImage, 'Resampling', PilImage), 'LANCZOS')
            image = image.resize(new_size, resampling)

        buffer = io.BytesIO()
        quality = max(1, min(95, self.vision_jpeg_quality))
        image.save(buffer, format='JPEG', quality=quality)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
        return {
            'jpeg_b64': image_b64,
            'data_url': f'data:image/jpeg;base64,{image_b64}',
            'api_width': image.width,
            'api_height': image.height,
            'original_width': original_width,
            'original_height': original_height,
            'stamp_sec': int(msg.header.stamp.sec),
            'stamp_nanosec': int(msg.header.stamp.nanosec),
            'frame_id': str(msg.header.frame_id),
        }

    def _ros_image_to_bgr_array(self, msg: RosImage):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                'OpenCV/Numpy is not installed. Install python3-opencv python3-numpy.'
            ) from exc

        encoding = str(msg.encoding).lower()
        width = int(msg.width)
        height = int(msg.height)
        channel_map = {
            'rgb8': 3,
            'bgr8': 3,
            'rgba8': 4,
            'bgra8': 4,
            'mono8': 1,
            '8uc1': 1,
            '8uc3': 3,
        }
        if encoding not in channel_map:
            raise RuntimeError(
                f'Unsupported camera image encoding for display: {msg.encoding}. '
                'Use a color topic such as /camera/camera/color/image_raw.'
            )
        channels = channel_map[encoding]
        expected_step = width * channels
        packed = self._pack_image_rows(bytes(msg.data), height, int(msg.step), expected_step)
        if channels == 1:
            image = np.frombuffer(packed, dtype=np.uint8).reshape((height, width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = np.frombuffer(packed, dtype=np.uint8).reshape((height, width, channels)).copy()
        if encoding in ('rgb8', '8uc3'):
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == 'bgr8':
            return image
        if encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image[:, :, :3]

    @staticmethod
    def _write_bgr_image(path: str, image) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('OpenCV is not installed; cannot save annotated image.') from exc
        if not cv2.imwrite(path, image):
            raise RuntimeError(f'cv2.imwrite failed: {path}')

    def _ros_image_to_pil(self, msg: RosImage):
        try:
            from PIL import Image as PilImage
        except ImportError as exc:
            raise RuntimeError(
                'Python package Pillow is not installed. Install it with: '
                '/usr/bin/python3 -m pip install --user -U pillow'
            ) from exc

        encoding = str(msg.encoding).lower()
        raw_modes = {
            'rgb8': ('RGB', 'RGB', 3),
            'bgr8': ('RGB', 'BGR', 3),
            'rgba8': ('RGBA', 'RGBA', 4),
            'bgra8': ('RGBA', 'BGRA', 4),
            'mono8': ('L', 'L', 1),
            '8uc1': ('L', 'L', 1),
            '8uc3': ('RGB', 'BGR', 3),
        }
        if encoding not in raw_modes:
            raise RuntimeError(
                f'Unsupported camera image encoding for vision: {msg.encoding}. '
                'Use a color topic such as /camera/camera/color/image_raw.'
            )

        mode, raw_mode, channels = raw_modes[encoding]
        width = int(msg.width)
        height = int(msg.height)
        expected_step = width * channels
        packed = self._pack_image_rows(bytes(msg.data), height, int(msg.step), expected_step)
        image = PilImage.frombytes(mode, (width, height), packed, 'raw', raw_mode)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    @staticmethod
    def _pack_image_rows(data: bytes, height: int, step: int, expected_step: int) -> bytes:
        if step == expected_step:
            return data[:height * expected_step]
        rows = []
        for row in range(height):
            start = row * step
            rows.append(data[start:start + expected_step])
        return b''.join(rows)

    def _api_key_from_env(self, env_name: str) -> str:
        api_key = os.environ.get(env_name, '').strip()
        if not api_key:
            raise RuntimeError(f'Environment variable {env_name} is not set.')
        try:
            api_key.encode('ascii')
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                f'Environment variable {env_name} contains non-ASCII text. '
                'Use the real API key, not a Chinese placeholder.'
            ) from exc
        return api_key

    @staticmethod
    def _tool_schema() -> List[Dict]:
        centimeter_property = {
            'type': 'object',
            'properties': {
                'centimeters': {
                    'type': 'number',
                    'description': (
                        'Distance in centimeters. Positive follows the named base-frame axis; '
                        'negative moves in the opposite direction.'
                    ),
                },
            },
            'required': ['centimeters'],
            'additionalProperties': False,
        }
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'look_camera',
                    'description': (
                        'Use the latest RealSense camera image to answer questions about '
                        'what is visible, object identity, color, relative position, and '
                        'visual distance estimates.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'question': {
                                'type': 'string',
                                'description': 'The user question to answer using the current camera image.',
                            },
                        },
                        'required': ['question'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'go_home',
                    'description': 'Move the robot arm back to the saved home joint pose.',
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_api_objects',
                    'description': (
                        'Use the latest camera image and vision API to return graspable object '
                        'candidates with boxes. Use this only when the user asks what objects '
                        'are available for grasping or what the robot can grab. Do not use it '
                        'for general scene description questions such as 你能看到什么.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'question': {
                                'type': 'string',
                                'description': 'Optional user question about visible/graspable objects.',
                            },
                        },
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'grab_api_object',
                    'description': (
                        'Grasp one object by using the vision API to detect its bounding box, '
                        'then using that box as an RGB-D ROI for EconomicGrasp pose generation, '
                        'TF to the robot base frame, and MoveIt IK. '
                        'Do not provide pixel coordinates; only provide the object name.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'object_name': {
                                'type': 'string',
                                'description': (
                                    'Raw user target description for the object to grab, for example '
                                    '橘子, 甜甜圈, 中间那个, orange, donut.'
                                ),
                            },
                            'motion_speed': {
                                'type': 'number',
                                'description': (
                                    'Optional one-shot motion speed scaling for this grasp only, '
                                    'from 0.0 to 1.0. Use this only when the user explicitly says '
                                    'a speed such as 用0.2的速度抓橘子. Omit it to use the default speed.'
                                ),
                            },
                        },
                        'required': ['object_name'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'box_api_object',
                    'description': (
                        'Draw boxes in the Qwen-Omni Vision Box for one requested object '
                        'or for multiple requested/visible objects. '
                        'Use this when the user asks to frame, box, mark, annotate, or '
                        'circle objects, but does not ask the robot to grasp them. '
                        'The object_name field may be a specific target such as apple, '
                        'a category phrase such as cups and bottles, or a broad phrase '
                        'such as all visible objects / 所有物体 / 全部物体.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'object_name': {
                                'type': 'string',
                                'description': (
                                    'Target description to draw boxes around. It can name one object, '
                                    'multiple objects, a category, or all visible objects.'
                                ),
                            },
                        },
                        'required': ['object_name'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'open_gripper',
                    'description': 'Open the Franka gripper to the configured open width.',
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'close_gripper',
                    'description': 'Close the Franka gripper with the configured grasp force.',
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'move_x_cm',
                    'description': 'Move the end effector along base-frame X. Right is positive; left is negative.',
                    'parameters': centimeter_property,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'move_y_cm',
                    'description': 'Move the end effector along base-frame Y. Forward is positive; backward is negative.',
                    'parameters': centimeter_property,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'move_z_cm',
                    'description': 'Move the end effector along base-frame Z. Up is positive; down is negative.',
                    'parameters': centimeter_property,
                },
            },
        ]

    def move_axis_callback(self, axis: str, request: MoveAxis.Request, response: MoveAxis.Response):
        if not self.motion_lock.acquire(blocking=False):
            response.success = False
            response.message = 'motion already running'
            return response
        try:
            response.success, response.message = self._move_axis(axis, float(request.centimeters))
            return response
        finally:
            self.motion_lock.release()

    def list_yolo_objects_callback(self, request, response):
        del request
        self._clear_target_commands_before_detection()
        ok, message, payload, age = self._detect_and_wait_for_summary('list_yolo_objects')
        if not ok:
            response.success = False
            response.message = message
            return response
        if payload is None:
            response.success = False
            response.message = (
                f'no detected objects received yet on {self.detected_objects_topic}; '
                'start RealSense, YOLO, and object_target_controller first'
            )
            return response
        if self._detected_objects_stale(age):
            response.success = False
            response.message = (
                f'detected objects are stale: age={age:.2f}s exceeds '
                f'{self.detected_objects_max_age_sec:.2f}s'
            )
            return response

        response.success = True
        response.message = self._format_detected_objects_message(payload, age)
        return response

    def _clear_target_commands_before_detection(self) -> None:
        self.target_command_pub.publish(String(data=''))
        if self.target_command_settle_sec > 0.0:
            time.sleep(self.target_command_settle_sec)

    def grab_object_callback(self, request: ObjectName.Request, response: ObjectName.Response):
        requested_name, motion_speed = self._parse_grab_request(request.name)
        if not requested_name:
            response.success = False
            response.message = 'grab_object rejected: object name is empty'
            return response

        self._clear_target_commands_before_detection()

        payload = None
        age = math.inf
        if self.require_detected_before_grab:
            ok, message, payload, age = self._detect_and_wait_for_summary('grab_object precheck')
            if not ok:
                response.success = False
                response.message = message
                return response
            if payload is None:
                response.success = False
                response.message = (
                    f'grab_object rejected: no detected objects received yet on '
                    f'{self.detected_objects_topic}; start YOLO and object_target_controller first'
                )
                return response
            if self._detected_objects_stale(age):
                response.success = False
                response.message = (
                    f'grab_object rejected: detected objects are stale '
                    f'age={age:.2f}s exceeds {self.detected_objects_max_age_sec:.2f}s'
                )
                return response
        else:
            payload, age = self._latest_detected_objects()

        target_name = self._resolve_object_name(requested_name, payload)
        if self.require_detected_before_grab and payload is not None:
            object_names = self._object_names_from_payload(payload)
            if not self._contains_object_name(object_names, target_name):
                response.success = False
                response.message = (
                    f'grab_object rejected: target "{requested_name}" resolved to '
                    f'"{target_name}", but current YOLO objects are: '
                    f'{self._format_object_names(object_names)}'
                )
                return response

        if self.require_target_command_subscriber and self.target_command_pub.get_subscription_count() < 1:
            response.success = False
            response.message = (
                f'grab_object rejected: no subscriber on {self.target_command_topic}; '
                'start object_target_controller first'
            )
            return response

        self.target_command_pub.publish(
            String(data=self._target_command_payload(target_name, motion_speed))
        )
        if self.target_command_settle_sec > 0.0:
            time.sleep(self.target_command_settle_sec)

        ok, detect_message = self._trigger_yolo_detection('grab_object execute')
        if not ok:
            self._clear_object_target_command_after_failure()
            response.success = False
            response.message = (
                f'grab_object rejected after selecting target "{target_name}": {detect_message}; '
                'target command was cleared to avoid a delayed unintended motion'
            )
            return response

        message = (
            f'grab_object command sent: requested="{requested_name}", target="{target_name}". '
            f'motion_speed={motion_speed if motion_speed is not None else "default"}. '
            f'fresh_detection="{detect_message}". '
            'object_target_controller will use YOLO center, aligned depth, TF to base, '
            'MoveIt IK, and execute the motion from this on-demand detection.'
        )
        self._publish_status(message)
        response.success = True
        response.message = message
        return response

    def _clear_object_target_command_after_failure(self) -> None:
        self.target_command_pub.publish(String(data=''))
        if self.target_command_settle_sec > 0.0:
            time.sleep(self.target_command_settle_sec)

    def open_gripper_callback(self, request, response):
        del request
        if not self.gripper_lock.acquire(blocking=False):
            response.success = False
            response.message = 'gripper action already running'
            return response
        try:
            response.success, response.message = self._open_gripper()
            return response
        finally:
            self.gripper_lock.release()

    def close_gripper_callback(self, request, response):
        del request
        if not self.gripper_lock.acquire(blocking=False):
            response.success = False
            response.message = 'gripper action already running'
            return response
        try:
            response.success, response.message = self._close_gripper()
            return response
        finally:
            self.gripper_lock.release()

    def _open_gripper(self) -> Tuple[bool, str]:
        client_name, client = self._available_action_client(
            self.gripper_move_clients,
            'gripper move',
        )
        if client is None:
            return False, (
                'no gripper move action available; tried '
                + ', '.join(self.gripper_move_actions)
            )

        goal = GripperMove.Goal()
        goal.width = self.gripper_open_width_m
        goal.speed = self.gripper_open_speed_mps
        ok, message = self._send_gripper_goal(client, goal, client_name, 'open_gripper')
        if ok:
            self._publish_status(message)
        return ok, message

    def _close_gripper(self) -> Tuple[bool, str]:
        client_name, client = self._available_action_client(
            self.gripper_grasp_clients,
            'gripper grasp',
        )
        if client is None:
            return False, (
                'no gripper grasp action available; tried '
                + ', '.join(self.gripper_grasp_actions)
            )

        goal = GripperGrasp.Goal()
        goal.width = self.gripper_close_width_m
        goal.speed = self.gripper_close_speed_mps
        goal.force = self.gripper_close_force_n
        goal.epsilon.inner = self.gripper_grasp_epsilon_inner_m
        goal.epsilon.outer = self.gripper_grasp_epsilon_outer_m
        ok, message = self._send_gripper_goal(client, goal, client_name, 'close_gripper')
        if ok:
            self._publish_status(message)
        return ok, message

    def _available_action_client(self, clients, label: str):
        for action_name, client in clients:
            if client.wait_for_server(timeout_sec=self.gripper_server_wait_timeout_sec):
                return action_name, client
        self.get_logger().warn(f'No available {label} action server.')
        return '', None

    def _send_gripper_goal(self, client, goal, action_name: str, label: str) -> Tuple[bool, str]:
        send_future = client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.gripper_action_timeout_sec,
            f'{label} goal',
        )
        if goal_handle is None:
            return False, f'{label} timed out sending goal to {action_name}'
        if not goal_handle.accepted:
            return False, f'{label} rejected by {action_name}'

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.gripper_action_timeout_sec,
            f'{label} result',
        )
        if action_result is None:
            return False, f'{label} timed out waiting for result from {action_name}'

        result = action_result.result
        if getattr(result, 'success', False):
            current_width = getattr(result, 'current_width', float('nan'))
            return True, f'{label} succeeded via {action_name}; current_width={current_width:.4f}m'

        error = getattr(result, 'error', '')
        if not error and label == 'close_gripper':
            error = (
                'grasp did not satisfy width/epsilon; increase '
                'gripper_grasp_epsilon_outer_m or set gripper_close_width_m '
                'near the object width'
            )
        return False, f'{label} failed via {action_name}: {error}'

    def _go_home(self):
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'action server not available: {self.trajectory_action}'
        start_positions = self._wait_for_current_joint_positions(self.home_wait_for_joint_state_sec)
        if start_positions is None:
            return (
                False,
                f'no complete joint state received on {self.joint_states_topic}; '
                'cannot build a smooth home trajectory',
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.header.stamp = (
            self.get_clock().now() + Duration(
                seconds=max(0.0, self.home_trajectory_start_delay_sec)
            )
        ).to_msg()
        goal.trajectory.joint_names = list(self.joint_names)
        goal.trajectory.points = self._make_smooth_home_points(start_positions)
        goal.goal_time_tolerance = self._duration_msg(1.0)

        self._publish_status(
            f'go_home requested: smooth trajectory with {len(goal.trajectory.points)} points, '
            f'duration={self.home_move_duration_sec:.2f}s; '
            + ', '.join(
                f'{name}={deg:.1f}deg' for name, deg in zip(self.joint_names, self.home_joint_positions_deg)
            )
        )
        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, 'home goal')
        if goal_handle is None:
            return False, 'timed out sending home goal'
        if not goal_handle.accepted:
            return False, 'home goal rejected'

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.home_move_duration_sec + self.action_wait_timeout_sec,
            'home result',
        )
        if action_result is None:
            return False, 'timed out waiting for home result'

        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            message = 'home motion finished successfully'
            self._publish_status(message)
            return True, message

        message = f'home motion failed: {result.error_code} {result.error_string}'
        self.get_logger().error(message)
        return False, message

    def _wait_for_current_joint_positions(self, timeout_sec: float) -> Optional[Dict[str, float]]:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while time.monotonic() <= deadline:
            if self.latest_joint_state is not None:
                positions = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
                if all(name in positions for name in self.joint_names):
                    return {
                        name: float(positions[name]) for name in self.joint_names
                    }
            time.sleep(0.02)
        return None

    def _make_smooth_home_points(self, start_positions: Dict[str, float]) -> List[JointTrajectoryPoint]:
        goal_positions = {
            name: math.radians(value)
            for name, value in zip(self.joint_names, self.home_joint_positions_deg)
        }
        duration = max(0.5, self.home_move_duration_sec)
        dt = min(max(0.01, self.home_trajectory_dt_sec), duration)
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

    def _move_axis(self, axis: str, centimeters: float):
        if abs(centimeters) > self.max_single_axis_move_cm:
            message = (
                f'move_{axis}_cm rejected: requested {centimeters:.2f} cm exceeds '
                f'single-step limit {self.max_single_axis_move_cm:.2f} cm'
            )
            self.get_logger().warn(message)
            self._publish_status(message)
            return False, message

        target_pose = self._make_offset_pose(axis, centimeters)
        if target_pose is None:
            return False, 'failed to build target pose from current end-effector transform'

        self._publish_status(
            f'move_{axis}_cm requested: {centimeters:.2f} cm; '
            f'target=({target_pose.pose.position.x:.3f}, '
            f'{target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f})'
        )

        if self.axis_move_execution_mode == 'cartesian':
            return self._execute_axis_cartesian_path(axis, centimeters, target_pose)

        joint_goal = self._compute_ik(target_pose)
        if joint_goal is None:
            return False, 'MoveIt IK failed'

        if self.axis_move_execution_mode == 'ik_only':
            message = f'IK solved without execution: {self._format_joint_goal(joint_goal)}'
            self._publish_status(message)
            return True, message
        if self.axis_move_execution_mode == 'move_group':
            return self._execute_with_move_group(joint_goal)
        return self._execute_with_trajectory_action(joint_goal)

    def _make_offset_pose(self, axis: str, centimeters: float) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                Time(),
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().error(
                f'Cannot get current transform {self.base_frame} -> {self.end_effector_frame}: {exc}'
            )
            return None

        delta_m = centimeters / 100.0
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        if axis == 'x':
            pose.pose.position.x += delta_m
        elif axis == 'y':
            pose.pose.position.y += delta_m
        elif axis == 'z':
            pose.pose.position.z += delta_m
        else:
            raise ValueError(f'unsupported axis: {axis}')
        pose.pose.orientation = Quaternion(
            x=transform.transform.rotation.x,
            y=transform.transform.rotation.y,
            z=transform.transform.rotation.z,
            w=transform.transform.rotation.w,
        )
        return pose

    def _compute_ik(self, target_pose: PoseStamped) -> Optional[Dict[str, float]]:
        if not self.ik_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().error(f'IK service not available: {self.ik_service}')
            return None
        if self.latest_joint_state is None:
            self.get_logger().error('No /joint_states received; cannot seed MoveIt IK.')
            return None

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.move_group_name
        request.ik_request.robot_state.joint_state = copy.deepcopy(self.latest_joint_state)
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
        missing = [name for name in self.joint_names if name not in positions]
        if missing:
            self.get_logger().error(f'IK solution missing joints: {missing}')
            return None

        joint_goal = {name: float(positions[name]) for name in self.joint_names}
        self._publish_status(f'IK solved: {self._format_joint_goal(joint_goal)}')
        return joint_goal

    def _execute_axis_cartesian_path(
        self,
        axis: str,
        centimeters: float,
        target_pose: PoseStamped,
    ) -> Tuple[bool, str]:
        if self.plan_only:
            return False, 'plan_only=true; Cartesian axis execution skipped'
        if not self.cartesian_path_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            return False, f'Cartesian path service not available: {self.cartesian_path_service}'
        if self.latest_joint_state is None:
            return False, 'No /joint_states received; cannot compute Cartesian axis path.'

        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state.joint_state = copy.deepcopy(self.latest_joint_state)
        request.start_state.is_diff = True
        request.group_name = self.move_group_name
        request.link_name = self.end_effector_frame
        request.waypoints = [copy.deepcopy(target_pose.pose)]
        request.max_step = self.axis_cartesian_max_step_m
        request.jump_threshold = self.axis_cartesian_jump_threshold
        request.avoid_collisions = self.avoid_collisions

        self._publish_status(
            f'Cartesian move_{axis}_cm request: {centimeters:.2f} cm, '
            f'max_step={self.axis_cartesian_max_step_m:.3f}m'
        )
        future = self.cartesian_path_client.call_async(request)
        response = self._wait_for_future(
            future,
            self.service_wait_timeout_sec + self.allowed_planning_time + 1.0,
            f'move_{axis}_cm Cartesian path',
        )
        if response is None:
            return False, f'move_{axis}_cm Cartesian path timed out'
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, f'move_{axis}_cm Cartesian path failed with code {response.error_code.val}'
        if response.fraction < self.axis_cartesian_min_fraction:
            return False, (
                f'move_{axis}_cm Cartesian path incomplete: '
                f'fraction={response.fraction:.3f}, required={self.axis_cartesian_min_fraction:.3f}'
            )

        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            return False, f'move_{axis}_cm Cartesian path returned an empty trajectory'

        duration = max(
            self.axis_cartesian_min_duration_sec,
            abs(float(centimeters)) * self.axis_cartesian_duration_per_cm_sec,
        )
        self._time_parameterize_cartesian_trajectory(trajectory, duration)
        ok, message = self._execute_joint_trajectory(
            trajectory,
            f'move_{axis}_cm Cartesian trajectory',
            duration + self.action_wait_timeout_sec,
        )
        if ok:
            message = (
                f'move_{axis}_cm Cartesian path executed successfully; '
                f'fraction={response.fraction:.3f}'
            )
            self._publish_status(message)
        return ok, message

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
            positions = list(points[0].positions)
            points[0].time_from_start = self._duration_msg(duration)
            points[0].velocities = [0.0 for _ in positions]
            points[0].accelerations = [0.0 for _ in positions]
            return

        min_step = 0.05
        last_positions = [float(value) for value in points[-1].positions]
        for index, point in enumerate(points):
            u = index / float(len(points) - 1)
            positions = [float(value) for value in point.positions]
            if index == 0:
                next_positions = [float(value) for value in points[index + 1].positions]
                dt = duration / float(len(points) - 1)
                velocities = [
                    (next_positions[joint_index] - positions[joint_index]) / dt
                    for joint_index in range(len(positions))
                ]
            elif index == len(points) - 1:
                velocities = [0.0 for _ in positions]
            else:
                previous_positions = [float(value) for value in points[index - 1].positions]
                next_positions = [float(value) for value in points[index + 1].positions]
                dt = 2.0 * duration / float(len(points) - 1)
                velocities = [
                    (next_positions[joint_index] - previous_positions[joint_index]) / dt
                    for joint_index in range(len(positions))
                ]
            if index == 0:
                velocities = [0.0 for _ in positions]
            point.positions = positions if index < len(points) - 1 else last_positions
            point.velocities = velocities
            point.accelerations = [0.0 for _ in positions]
            point.time_from_start = self._duration_msg(max(min_step, duration * u))

    def _execute_with_move_group(self, joint_goal: Dict[str, float]):
        if not self.move_group_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'MoveGroup action not available: {self.move_group_action}'
        if self.latest_joint_state is None:
            return False, 'No /joint_states received; cannot plan MoveGroup goal.'

        goal = MoveGroup.Goal()
        goal.request.group_name = self.move_group_name
        goal.request.num_planning_attempts = self.num_planning_attempts
        goal.request.allowed_planning_time = self.allowed_planning_time
        goal.request.max_velocity_scaling_factor = self.max_velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.max_acceleration_scaling
        goal.request.start_state.joint_state = copy.deepcopy(self.latest_joint_state)
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
            return False, 'timed out sending MoveGroup goal'
        if not goal_handle.accepted:
            return False, 'MoveGroup rejected the goal'

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.allowed_planning_time + self.action_wait_timeout_sec + self.motion_duration_sec,
            'MoveGroup result',
        )
        if action_result is None:
            return False, 'timed out waiting for MoveGroup result'

        result = action_result.result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            mode = 'planned' if self.plan_only else 'planned and executed'
            message = f'MoveGroup {mode} successfully'
            self._publish_status(message)
            return True, message

        message = f'MoveGroup failed with code {result.error_code.val}'
        self.get_logger().error(message)
        return False, message

    def _execute_with_trajectory_action(self, joint_goal: Dict[str, float]):
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'Trajectory action not available: {self.trajectory_action}'

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.trajectory.joint_names = list(self.joint_names)

        point = JointTrajectoryPoint()
        point.positions = [joint_goal[name] for name in self.joint_names]
        point.time_from_start = self._duration_msg(self.motion_duration_sec)
        goal.trajectory.points.append(point)
        goal.goal_time_tolerance = self._duration_msg(1.0)

        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, 'trajectory goal')
        if goal_handle is None:
            return False, 'timed out sending trajectory goal'
        if not goal_handle.accepted:
            return False, 'trajectory controller rejected the goal'

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.motion_duration_sec + self.action_wait_timeout_sec,
            'trajectory result',
        )
        if action_result is None:
            return False, 'timed out waiting for trajectory result'

        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            message = 'trajectory target executed successfully'
            self._publish_status(message)
            return True, message

        message = f'Trajectory failed: {result.error_code} {result.error_string}'
        self.get_logger().error(message)
        return False, message

    def _execute_joint_trajectory(
        self,
        trajectory: JointTrajectory,
        label: str,
        timeout_sec: float,
    ) -> Tuple[bool, str]:
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'Trajectory action not available: {self.trajectory_action}'

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = copy.deepcopy(trajectory)
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.goal_time_tolerance = self._duration_msg(1.0)

        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.action_wait_timeout_sec,
            f'{label} goal',
        )
        if goal_handle is None:
            return False, f'timed out sending {label}'
        if not goal_handle.accepted:
            return False, f'trajectory controller rejected {label}'

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(result_future, timeout_sec, f'{label} result')
        if action_result is None:
            return False, f'timed out waiting for {label}'

        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            message = f'{label} executed successfully'
            self._publish_status(message)
            return True, message

        message = f'{label} failed: {result.error_code} {result.error_string}'
        self.get_logger().error(message)
        return False, message

    def _joint_goal_constraints(self, joint_goal: Dict[str, float]) -> Constraints:
        constraints = Constraints()
        constraints.name = 'mcp_ik_joint_goal'
        for name in self.joint_names:
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
        nanosec = int(round((seconds - whole) * 1e9))
        if nanosec >= 1_000_000_000:
            whole += 1
            nanosec -= 1_000_000_000
        msg = DurationMsg()
        msg.sec = whole
        msg.nanosec = nanosec
        return msg

    @staticmethod
    def _format_joint_goal(joint_goal: Dict[str, float]) -> str:
        return ', '.join(f'{name}={value:.3f}' for name, value in joint_goal.items())

    def _publish_status(self, message: str) -> None:
        self.status_pub.publish(String(data=message))
        self.get_logger().info(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = McpServer()
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
