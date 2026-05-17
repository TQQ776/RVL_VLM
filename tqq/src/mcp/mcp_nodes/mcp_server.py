import base64
import copy
import io
import json
import math
import os
import struct
import threading
import time
import traceback
import uuid
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Grasp as GripperGrasp
from franka_msgs.action import Move as GripperMove
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from mcp.srv import CallTool, ListTools, MoveAxis, PlaceIntoContainer, PlaceRelative
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    ObjectColor,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
    RobotTrajectory,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from mcp_nodes.container_place import ContainerPlaceMixin
from mcp_nodes.gripper import GripperMixin
from mcp_nodes.mcp_shared import normalize_object_name
from mcp_nodes.planning import PlanningMixin
from mcp_nodes.scene_memory import SceneMemoryMixin
from mcp_nodes.vision import VisionMixin

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - fallback only for unusual ROS envs
    get_package_share_directory = None


class McpServer(VisionMixin, SceneMemoryMixin, PlanningMixin, ContainerPlaceMixin, GripperMixin, Node):
    """Expose robot motion tools as ROS services for an LLM client."""

    def __init__(self) -> None:
        super().__init__('mcp_server')
        self.callback_group = ReentrantCallbackGroup()
        self.motion_lock = threading.Lock()
        self.gripper_lock = threading.Lock()
        self.latest_joint_state: Optional[JointState] = None
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
        self._latest_depth_image: Optional[RosImage] = None
        self._latest_depth_image_time = 0.0
        self._depth_image_lock = threading.Lock()
        self._latest_camera_info: Optional[CameraInfo] = None
        self._latest_camera_info_time = 0.0
        self._camera_info_lock = threading.Lock()
        self.scene_memory_lock = threading.Lock()
        self.scene_memory: Dict[str, Dict] = {}
        self.held_object_lock = threading.Lock()
        self.current_attached_object_id = ''
        self.scene_collision_lock = threading.Lock()
        self.active_scene_collision_ids = set()
        self.scene_marker_lock = threading.Lock()
        self.active_scene_marker_ids = set()
        self.active_scene_visual_entries: Dict[str, Dict] = {}
        self.held_object_visual: Optional[Dict] = None
        self._mesh_cache: Dict[str, Mesh] = {}
        self._shutdown_requested = threading.Event()
        self._emergency_stop_requested = threading.Event()
        self._active_goal_lock = threading.Lock()
        self._active_goal_handles: Dict[str, object] = {}

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
        self.emergency_stop_pub = self.create_publisher(String, '/mcp/emergency_stop', 10)
        self.target_command_pub = self.create_publisher(String, self.target_command_topic, 10)
        self.api_detections_pub = self.create_publisher(String, self.api_detections_topic, 10)
        self.scene_markers_pub = self.create_publisher(
            MarkerArray,
            self.scene_markers_topic,
            10,
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
            self._vision_depth_sub = self.create_subscription(
                RosImage,
                self.vision_depth_topic,
                self._vision_depth_callback,
                image_qos,
                callback_group=self.callback_group,
            )
            self._vision_camera_info_sub = self.create_subscription(
                CameraInfo,
                self.vision_camera_info_topic,
                self._vision_camera_info_callback,
                10,
                callback_group=self.callback_group,
            )
            if self.vision_show_window:
                self._vision_display_thread = threading.Thread(
                    target=self._vision_display_loop,
                    daemon=True,
                )
                self._vision_display_thread.start()
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
        self.place_relative_srv = self.create_service(
            PlaceRelative,
            '~/place_relative_to_object',
            self.place_relative_callback,
            callback_group=self.callback_group,
        )
        self.place_into_container_srv = self.create_service(
            PlaceIntoContainer,
            '~/place_into_container',
            self.place_into_container_callback,
            callback_group=self.callback_group,
        )
        self.emergency_stop_srv = self.create_service(
            Trigger,
            '~/emergency_stop',
            self.emergency_stop_callback,
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
        self.apply_planning_scene_client = self.create_client(
            ApplyPlanningScene,
            self.apply_planning_scene_service,
            callback_group=self.callback_group,
        )
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_group_action,
            callback_group=self.callback_group,
        )
        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            self.execute_trajectory_action,
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
            '~/open_gripper, ~/close_gripper, ~/place_relative_to_object, ~/place_into_container, '
            '~/emergency_stop; '
            f'base_frame={self.base_frame}, ee={self.end_effector_frame}, '
            f'vision_topic={self.vision_image_topic}, depth_topic={self.vision_depth_topic}, '
            f'api_detections={self.api_detections_topic}'
        )

    def destroy_node(self) -> bool:
        self._shutdown_requested.set()
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
        self.declare_parameter('apply_planning_scene_service', '/apply_planning_scene')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('move_group_planner_id', '')
        self.declare_parameter('execution_mode', 'move_group')
        self.declare_parameter('axis_move_execution_mode', 'cartesian')
        self.declare_parameter('plan_only', False)
        self.declare_parameter('avoid_collisions', True)
        self.declare_parameter('ik_timeout_sec', 0.5)
        self.declare_parameter('service_wait_timeout_sec', 5.0)
        self.declare_parameter('action_wait_timeout_sec', 10.0)
        self.declare_parameter('move_group_result_timeout_scale', 2.0)
        self.declare_parameter('execute_trajectory_action', '/execute_trajectory')
        self.declare_parameter('move_group_execution_backend', 'controller')
        self.declare_parameter('motion_duration_sec', 4.0)
        self.declare_parameter('max_single_axis_move_cm', 10.0)
        self.declare_parameter('goal_joint_tolerance', 0.01)
        self.declare_parameter('max_moveit_joint_delta_rad', 1.0471975511965976)
        self.declare_parameter('max_moveit_total_joint_delta_rad', 4.1887902047863905)
        self.declare_parameter('moveit_guard_plan_retries', 3)
        self.declare_parameter('moveit_joint_delta_guard_ignore_joints', ['fr3_joint7'])
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
        self.declare_parameter('target_command_topic', '/economic_grasp_roi/target_class_name')
        self.declare_parameter('api_detections_topic', '/mcp_omni_client/api_detections_json')
        self.declare_parameter('grasp_result_topic', '/economic_grasp_roi/grasp_result')
        self.declare_parameter('require_target_command_subscriber', True)

        self.declare_parameter('omni_api_key_env', 'DASHSCOPE_API_KEY')
        self.declare_parameter('omni_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.declare_parameter('omni_text_model', 'qwen3.5-omni-plus')
        self.declare_parameter('omni_timeout', 90.0)
        self.declare_parameter('omni_max_tokens', 1000)
        self.declare_parameter('vision_enabled', True)
        self.declare_parameter('vision_image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter(
            'vision_depth_topic',
            '/camera/camera/aligned_depth_to_color/image_raw',
        )
        self.declare_parameter('vision_camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('vision_tf_wait_timeout_sec', 10.0)
        self.declare_parameter('vision_image_max_age_sec', 30.0)
        self.declare_parameter('vision_max_image_width', 640)
        self.declare_parameter('vision_jpeg_quality', 85)
        self.declare_parameter('vision_show_window', True)
        self.declare_parameter('vision_window_name', '千问视觉检测框')
        self.declare_parameter('vision_save_images', True)
        self.declare_parameter('vision_output_dir', '/home/tqq/TQQ_ws/omni_vision_outputs')
        self.declare_parameter('vision_result_hold_sec', 60.0)
        self.declare_parameter('api_detection_default_confidence', 0.90)
        self.declare_parameter('api_detection_publish_settle_sec', 0.25)
        self.declare_parameter('api_detection_republish_count', 1)
        self.declare_parameter('api_detection_republish_interval_sec', 0.15)
        self.declare_parameter('api_detection_box_coordinate_space', 'qwen_1000')
        self.declare_parameter('api_detection_box_reference_size', 1000.0)
        self.declare_parameter('grab_api_default_motion_speed', 0.05)
        self.declare_parameter('grab_api_wait_for_result', True)
        self.declare_parameter('grab_api_result_timeout_sec', 180.0)
        self.declare_parameter('scene_memory_max_age_sec', 1800.0)
        self.declare_parameter('scene_memory_depth_percentile', 35.0)
        self.declare_parameter('scene_memory_depth_margin_m', 0.08)
        self.declare_parameter('scene_memory_min_depth_m', 0.05)
        self.declare_parameter('scene_memory_max_depth_m', 3.0)
        self.declare_parameter('scene_memory_erode_px', 2)
        self.declare_parameter('place_lift_m', 0.03)
        self.declare_parameter('place_lift_avoid_collisions', False)
        self.declare_parameter('place_transfer_via_home', True)
        self.declare_parameter('place_pre_height_m', 0.10)
        self.declare_parameter('place_descend_m', 0.08)
        self.declare_parameter('place_descend_min_fraction', 0.95)
        self.declare_parameter('place_default_offset_cm', 5.0)
        self.declare_parameter('place_target_z_offset_m', 0.0)
        self.declare_parameter('place_move_duration_sec', 4.0)
        self.declare_parameter('place_max_velocity_scaling', 0.03)
        self.declare_parameter('place_max_acceleration_scaling', 0.03)
        self.declare_parameter('place_move_group_planner_id', 'RRTConnectkConfigDefault')
        self.declare_parameter('place_num_planning_attempts', 12)
        self.declare_parameter('place_allowed_planning_time_sec', 8.0)
        self.declare_parameter('place_goal_position_tolerance_m', 0.005)
        self.declare_parameter('place_goal_orientation_tolerance_rad', 0.05)
        self.declare_parameter('container_length_x_m', 0.30)
        self.declare_parameter('container_width_y_m', 0.22)
        self.declare_parameter('container_height_m', 0.14)
        self.declare_parameter('container_wall_thickness_m', 0.015)
        self.declare_parameter('container_bottom_thickness_m', 0.015)
        self.declare_parameter('container_collision_z_offset_m', 0.0)
        self.declare_parameter('container_planning_scene_wait_timeout_sec', 2.0)
        self.declare_parameter('scene_collision_auto_apply', True)
        self.declare_parameter('scene_collision_show_fruits', False)
        self.declare_parameter('scene_collision_show_containers', True)
        self.declare_parameter('scene_collision_show_other_objects', False)
        self.declare_parameter('scene_collision_show_static_right_arm', True)
        self.declare_parameter('static_right_arm_offset_xyz_m', [0.92, 0.0, 0.0])
        self.declare_parameter(
            'static_right_arm_joint_positions_deg',
            [74.0, -3.0, -7.0, -115.0, -1.0, 110.0, 22.0],
        )
        self.declare_parameter('static_right_arm_collision_padding_m', 0.03)
        self.declare_parameter('static_right_arm_marker_base_id', 1800000)
        self.declare_parameter('scene_collision_preview_sec', 0.75)
        self.declare_parameter('scene_collision_clear_previous', True)
        self.declare_parameter('scene_markers_topic', '~/scene_markers')
        self.declare_parameter('scene_markers_enabled', True)
        self.declare_parameter('object_collision_padding_m', 0.005)
        self.declare_parameter('object_collision_min_size_m', 0.03)
        self.declare_parameter('object_collision_max_size_m', 0.20)
        self.declare_parameter('held_object_collision_enabled', False)
        self.declare_parameter('held_object_link_name', 'fr3_hand_tcp')
        self.declare_parameter('held_object_touch_links', [
            'fr3_hand',
            'fr3_leftfinger',
            'fr3_rightfinger',
            'fr3_hand_tcp',
        ])
        self.declare_parameter('held_object_offset_xyz_tcp', [0.0, 0.0, 0.04])
        self.declare_parameter('held_object_padding_m', 0.0)
        self.declare_parameter('held_object_min_radius_m', 0.035)
        self.declare_parameter('held_object_max_radius_m', 0.06)
        self.declare_parameter('held_object_default_radius_m', 0.05)
        self.declare_parameter('fruit_collision_box_long_ratio', 1.8)

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
        self.apply_planning_scene_service = str(
            self.get_parameter('apply_planning_scene_service').value
        )
        self.move_group_action = str(self.get_parameter('move_group_action').value)
        self.execute_trajectory_action = str(
            self.get_parameter('execute_trajectory_action').value
        )
        self.move_group_execution_backend = str(
            self.get_parameter('move_group_execution_backend').value
        ).strip().lower()
        self.move_group_planner_id = str(self.get_parameter('move_group_planner_id').value).strip()
        self.execution_mode = str(self.get_parameter('execution_mode').value).strip().lower()
        self.axis_move_execution_mode = str(
            self.get_parameter('axis_move_execution_mode').value
        ).strip().lower()
        self.plan_only = self._as_bool(self.get_parameter('plan_only').value)
        self.avoid_collisions = self._as_bool(self.get_parameter('avoid_collisions').value)
        self.ik_timeout_sec = float(self.get_parameter('ik_timeout_sec').value)
        self.service_wait_timeout_sec = float(self.get_parameter('service_wait_timeout_sec').value)
        self.action_wait_timeout_sec = float(self.get_parameter('action_wait_timeout_sec').value)
        self.move_group_result_timeout_scale = max(
            1.0,
            float(self.get_parameter('move_group_result_timeout_scale').value),
        )
        self.motion_duration_sec = float(self.get_parameter('motion_duration_sec').value)
        self.max_single_axis_move_cm = float(self.get_parameter('max_single_axis_move_cm').value)
        self.goal_joint_tolerance = float(self.get_parameter('goal_joint_tolerance').value)
        self.max_moveit_joint_delta_rad = max(
            0.0,
            float(self.get_parameter('max_moveit_joint_delta_rad').value),
        )
        self.max_moveit_total_joint_delta_rad = max(
            0.0,
            float(self.get_parameter('max_moveit_total_joint_delta_rad').value),
        )
        self.moveit_guard_plan_retries = max(
            1,
            int(self.get_parameter('moveit_guard_plan_retries').value),
        )
        self.moveit_joint_delta_guard_ignore_joints = set(
            self._string_list(self.get_parameter('moveit_joint_delta_guard_ignore_joints').value)
        )
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
        self.target_command_topic = str(self.get_parameter('target_command_topic').value)
        self.api_detections_topic = str(self.get_parameter('api_detections_topic').value).strip()
        self.grasp_result_topic = str(self.get_parameter('grasp_result_topic').value).strip()
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
        self.vision_depth_topic = str(self.get_parameter('vision_depth_topic').value).strip()
        self.vision_camera_info_topic = str(
            self.get_parameter('vision_camera_info_topic').value
        ).strip()
        self.vision_tf_wait_timeout_sec = max(
            self.ik_timeout_sec,
            float(self.get_parameter('vision_tf_wait_timeout_sec').value),
        )
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
        self.scene_memory_max_age_sec = max(
            0.0,
            float(self.get_parameter('scene_memory_max_age_sec').value),
        )
        self.scene_memory_depth_percentile = min(
            95.0,
            max(1.0, float(self.get_parameter('scene_memory_depth_percentile').value)),
        )
        self.scene_memory_depth_margin_m = max(
            0.0,
            float(self.get_parameter('scene_memory_depth_margin_m').value),
        )
        self.scene_memory_min_depth_m = max(
            0.0,
            float(self.get_parameter('scene_memory_min_depth_m').value),
        )
        self.scene_memory_max_depth_m = max(
            self.scene_memory_min_depth_m + 0.01,
            float(self.get_parameter('scene_memory_max_depth_m').value),
        )
        self.scene_memory_erode_px = max(0, int(self.get_parameter('scene_memory_erode_px').value))
        self.place_lift_m = max(0.0, float(self.get_parameter('place_lift_m').value))
        self.place_lift_avoid_collisions = self._as_bool(
            self.get_parameter('place_lift_avoid_collisions').value
        )
        self.place_transfer_via_home = self._as_bool(
            self.get_parameter('place_transfer_via_home').value
        )
        self.place_pre_height_m = max(0.0, float(self.get_parameter('place_pre_height_m').value))
        self.place_descend_m = max(0.0, float(self.get_parameter('place_descend_m').value))
        self.place_descend_min_fraction = min(
            1.0,
            max(0.0, float(self.get_parameter('place_descend_min_fraction').value)),
        )
        self.place_default_offset_cm = float(self.get_parameter('place_default_offset_cm').value)
        self.place_target_z_offset_m = float(self.get_parameter('place_target_z_offset_m').value)
        self.place_move_duration_sec = max(
            0.5,
            float(self.get_parameter('place_move_duration_sec').value),
        )
        self.place_max_velocity_scaling = min(
            1.0,
            max(0.001, float(self.get_parameter('place_max_velocity_scaling').value)),
        )
        self.place_max_acceleration_scaling = min(
            1.0,
            max(0.001, float(self.get_parameter('place_max_acceleration_scaling').value)),
        )
        self.place_move_group_planner_id = str(
            self.get_parameter('place_move_group_planner_id').value
        ).strip()
        self.place_num_planning_attempts = max(
            1,
            int(self.get_parameter('place_num_planning_attempts').value),
        )
        self.place_allowed_planning_time_sec = max(
            0.5,
            float(self.get_parameter('place_allowed_planning_time_sec').value),
        )
        self.place_goal_position_tolerance_m = max(
            0.001,
            float(self.get_parameter('place_goal_position_tolerance_m').value),
        )
        self.place_goal_orientation_tolerance_rad = max(
            0.001,
            float(self.get_parameter('place_goal_orientation_tolerance_rad').value),
        )
        self.container_length_x_m = max(0.05, float(self.get_parameter('container_length_x_m').value))
        self.container_width_y_m = max(0.05, float(self.get_parameter('container_width_y_m').value))
        self.container_height_m = max(0.02, float(self.get_parameter('container_height_m').value))
        self.container_wall_thickness_m = max(
            0.002,
            float(self.get_parameter('container_wall_thickness_m').value),
        )
        self.container_bottom_thickness_m = max(
            0.002,
            float(self.get_parameter('container_bottom_thickness_m').value),
        )
        self.container_collision_z_offset_m = float(
            self.get_parameter('container_collision_z_offset_m').value
        )
        self.container_planning_scene_wait_timeout_sec = max(
            0.1,
            float(self.get_parameter('container_planning_scene_wait_timeout_sec').value),
        )
        self.scene_collision_auto_apply = self._as_bool(
            self.get_parameter('scene_collision_auto_apply').value
        )
        self.scene_collision_show_fruits = self._as_bool(
            self.get_parameter('scene_collision_show_fruits').value
        )
        self.scene_collision_show_containers = self._as_bool(
            self.get_parameter('scene_collision_show_containers').value
        )
        self.scene_collision_show_other_objects = self._as_bool(
            self.get_parameter('scene_collision_show_other_objects').value
        )
        self.scene_collision_show_static_right_arm = self._as_bool(
            self.get_parameter('scene_collision_show_static_right_arm').value
        )
        self.static_right_arm_offset_xyz_m = self._float_list(
            self.get_parameter('static_right_arm_offset_xyz_m').value,
            3,
            [0.92, 0.0, 0.0],
        )
        static_right_arm_joint_positions_deg = self._float_list(
            self.get_parameter('static_right_arm_joint_positions_deg').value,
            len(self.joint_names),
            self.home_joint_positions_deg,
        )
        self.static_right_arm_joint_positions_deg = static_right_arm_joint_positions_deg
        self.static_right_arm_collision_padding_m = max(
            0.0,
            float(self.get_parameter('static_right_arm_collision_padding_m').value),
        )
        self.static_right_arm_marker_base_id = int(
            self.get_parameter('static_right_arm_marker_base_id').value
        )
        self.scene_collision_preview_sec = max(
            0.0,
            float(self.get_parameter('scene_collision_preview_sec').value),
        )
        self.scene_collision_clear_previous = self._as_bool(
            self.get_parameter('scene_collision_clear_previous').value
        )
        self.scene_markers_topic = str(self.get_parameter('scene_markers_topic').value).strip()
        self.scene_markers_enabled = self._as_bool(
            self.get_parameter('scene_markers_enabled').value
        )
        self.object_collision_padding_m = max(
            0.0,
            float(self.get_parameter('object_collision_padding_m').value),
        )
        self.object_collision_min_size_m = max(
            0.001,
            float(self.get_parameter('object_collision_min_size_m').value),
        )
        self.object_collision_max_size_m = max(
            self.object_collision_min_size_m,
            float(self.get_parameter('object_collision_max_size_m').value),
        )
        self.held_object_collision_enabled = self._as_bool(
            self.get_parameter('held_object_collision_enabled').value
        )
        self.held_object_link_name = str(self.get_parameter('held_object_link_name').value).strip()
        self.held_object_touch_links = self._string_list(
            self.get_parameter('held_object_touch_links').value
        )
        self.held_object_offset_xyz_tcp = self._float_list(
            self.get_parameter('held_object_offset_xyz_tcp').value,
            3,
            [0.0, 0.0, 0.04],
        )
        self.held_object_padding_m = max(
            0.0,
            float(self.get_parameter('held_object_padding_m').value),
        )
        self.held_object_min_radius_m = max(
            0.001,
            float(self.get_parameter('held_object_min_radius_m').value),
        )
        self.held_object_max_radius_m = max(
            self.held_object_min_radius_m,
            float(self.get_parameter('held_object_max_radius_m').value),
        )
        self.held_object_default_radius_m = min(
            self.held_object_max_radius_m,
            max(
                self.held_object_min_radius_m,
                float(self.get_parameter('held_object_default_radius_m').value),
            ),
        )
        self.fruit_collision_box_long_ratio = max(
            1.0,
            float(self.get_parameter('fruit_collision_box_long_ratio').value),
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
    def _float_list(value, expected_len: int, default: List[float]) -> List[float]:
        if isinstance(value, str):
            raw_items = value.replace('[', '').replace(']', '').split(',')
        else:
            raw_items = list(value or [])
        values = []
        for item in raw_items:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                pass
        if len(values) != expected_len:
            return list(default)
        return values

    @staticmethod
    def _normalize_object_name(name: str) -> str:
        text = str(name or '').strip().strip(' "\'“”‘’。，,.;；:：')
        return ' '.join(text.split())

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
            self.get_logger().error(
                f'call_tool crashed while running {name}: {exc}\n{traceback.format_exc()}'
            )
            success = False
            message = f'{name or "call_tool"} failed: {exc}'
            result = {}

        response.success = bool(success)
        response.message = message
        response.result_json = json.dumps(result or {}, ensure_ascii=False)
        return response

    def _call_tool_by_name(self, name: str, arguments: Dict) -> Tuple[bool, str, Dict]:
        if name == 'emergency_stop':
            return self._tool_emergency_stop(arguments)
        self._clear_emergency_stop_for_new_command()
        if name == 'look_camera':
            return self._tool_look_camera(arguments)
        if name == 'go_home':
            return self._run_locked_motion('go_home', self._go_home)
        if name == 'observe_scene':
            return self._tool_observe_scene(arguments)
        if name == 'list_api_objects':
            return self._tool_list_api_objects(arguments)
        if name == 'box_api_object':
            return self._tool_box_api_object(arguments)
        if name == 'grab_api_object':
            return self._tool_grab_api_object(arguments)
        if name == 'place_relative_to_object':
            return self._tool_place_relative_to_object(arguments)
        if name == 'pick_and_place_relative':
            return self._tool_pick_and_place_relative(arguments)
        if name == 'place_into_container':
            return self._tool_place_into_container(arguments)
        if name == 'pick_and_place_into_container':
            return self._tool_pick_and_place_into_container(arguments)
        if name == 'pick_all_fruits_into_container':
            return self._tool_pick_all_fruits_into_container(arguments)
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

    def emergency_stop_callback(self, request, response):
        del request
        success, message, _ = self._tool_emergency_stop({})
        response.success = bool(success)
        response.message = message
        return response

    def _tool_emergency_stop(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        del arguments
        message, result = self._request_emergency_stop()
        return True, message, result

    def _clear_emergency_stop_for_new_command(self) -> None:
        if self._emergency_stop_requested.is_set():
            self._emergency_stop_requested.clear()
            self._publish_status('EMERGENCY_STOP reset by new command')

    def _run_locked_motion(self, label: str, function) -> Tuple[bool, str, Dict]:
        self._clear_emergency_stop_for_new_command()
        if not self.motion_lock.acquire(blocking=False):
            return False, f'{label} failed: motion already running', {}
        try:
            success, message = function()
            return bool(success), f'{label} {"success" if success else "failed"}: {message}', {}
        finally:
            self.motion_lock.release()

    def _run_locked_gripper(self, label: str, function) -> Tuple[bool, str, Dict]:
        self._clear_emergency_stop_for_new_command()
        if not self.gripper_lock.acquire(blocking=False):
            return False, f'{label} failed: gripper action already running', {}
        try:
            success, message = function()
            return bool(success), f'{label} {"success" if success else "failed"}: {message}', {}
        finally:
            self.gripper_lock.release()

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
                    'name': 'emergency_stop',
                    'description': (
                        'Immediately request a software emergency stop: cancel active robot '
                        'motion goals, hold the current arm joint position, interrupt gripper '
                        'actions when possible, and notify grasp controllers to stop. Use only '
                        'when the user explicitly asks to stop/急停/停止机械臂.'
                    ),
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
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
                    'name': 'observe_scene',
                    'description': (
                        'Capture the current camera view, detect visible objects with boxes, '
                        'and save scene memory with each object bbox plus an estimated '
                        '3D base-frame position from aligned depth. The server also immediately '
                        'publishes fruit and container collision models to MoveIt planning scene, '
                        'so RViz shows the models before any robot motion starts. Use this before '
                        'a pick-and-place sequence when a later place reference may leave the camera view.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'object_names': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': (
                                    'Optional object names to remember, for example ["orange", "apple"]. '
                                    'Omit or pass an empty list to remember all visible objects.'
                                ),
                            },
                            'question': {
                                'type': 'string',
                                'description': 'Optional visual detection instruction for the current scene.',
                            },
                        },
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_api_objects',
                    'description': (
                        'Use the latest camera image and vision API to return graspable object '
                        'candidates with boxes, and refresh scene memory for later place actions. '
                        'Use this only when the user asks what objects are available for grasping '
                        'or what the robot can grab. Do not use it '
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
                        'The detection also refreshes scene memory for this object. '
                        'When called as a standalone tool, after a successful grasp it returns '
                        'to the saved home joint pose at half of the normal go_home speed. '
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
                        'Draw boxes in the 千问视觉检测框 for one requested object '
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
                    'name': 'place_relative_to_object',
                    'description': (
                        'Place the currently held object relative to a remembered reference object. '
                        'The reference must already be in scene memory from observe_scene, '
                        'list_api_objects, box_api_object, or a previous API grasp detection. '
                        'Execution preserves the current gripper orientation, lifts along base Z, '
                        'plans above the target, descends along base Z, opens the gripper, then lifts away. '
                        'When called directly, after successful placement it returns to the saved '
                        'home joint pose at half of the normal go_home speed.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'held_object_name': {
                                'type': 'string',
                                'description': 'Name of the object currently held by the gripper.',
                            },
                            'reference_object_name': {
                                'type': 'string',
                                'description': 'Name of the remembered object used as the placement reference.',
                            },
                            'direction': {
                                'type': 'string',
                                'description': (
                                    'Relative direction in the robot base frame: left/right/front/back/up/down, '
                                    'or 中文 左/右/前/后/上/下.'
                                ),
                            },
                            'distance_cm': {
                                'type': 'number',
                                'description': (
                                    'Offset distance from the reference object in centimeters. '
                                    'Omit only when the user did not specify a distance; the server default is used.'
                                ),
                            },
                        },
                        'required': ['reference_object_name', 'direction'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'pick_and_place_relative',
                    'description': (
                        'Complete one long sequence: first observe and remember the scene, then grasp '
                        'object_name with EconomicGrasp, then place it relative to reference_object_name. '
                        'The observe step applies visible MoveIt/RViz models before motion; after '
                        'grasping, the held fruit is attached to the gripper model for collision avoidance. '
                        'If the gripper already holds object_name, skip observe/grasp and directly place '
                        'using the remembered reference object. '
                        'The sequence is complete only after placement and a half-speed return home. '
                        'Use this for requests like 把橘子放到苹果右边5厘米, because the reference may '
                        'not remain visible after grasping.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'object_name': {
                                'type': 'string',
                                'description': 'Object to grasp first.',
                            },
                            'reference_object_name': {
                                'type': 'string',
                                'description': 'Reference object to place near.',
                            },
                            'direction': {
                                'type': 'string',
                                'description': 'Relative direction: left/right/front/back/up/down or 左/右/前/后/上/下.',
                            },
                            'distance_cm': {
                                'type': 'number',
                                'description': 'Offset distance from the reference object in centimeters.',
                            },
                            'motion_speed': {
                                'type': 'number',
                                'description': 'Optional one-shot grasp speed scaling from 0.0 to 1.0.',
                            },
                        },
                        'required': ['object_name', 'reference_object_name', 'direction'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'place_into_container',
                    'description': (
                        'Place the currently held object into an open-top container such as 箱子/box. '
                        'The container must already be in scene memory. The server adds a hollow '
                        'container collision model to MoveIt using five boxes: left wall, right wall, '
                        'front wall, back wall, and bottom. It preserves the current gripper '
                        'orientation, plans directly from the current pose to above the container '
                        'opening with MoveIt collision avoidance, descends vertically into the box, '
                        'opens the gripper, and lifts away. When called directly, after successful '
                        'placement it returns to the saved home joint pose at half of the normal '
                        'go_home speed.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'held_object_name': {
                                'type': 'string',
                                'description': 'Name of the object currently held by the gripper.',
                            },
                            'container_name': {
                                'type': 'string',
                                'description': 'Name of the remembered open container, for example box or 箱子.',
                            },
                        },
                        'required': ['container_name'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'pick_and_place_into_container',
                    'description': (
                        'Complete one long sequence for requests like 把苹果放到箱子里: first observe '
                        'and remember the object and box, grasp object_name with EconomicGrasp, add '
                        'visible MoveIt/RViz models before motion, attach the held fruit model after '
                        'grasping, plan to above the box opening with collision avoidance, descend, '
                        'open the gripper, detach the held fruit model, and lift away. If the gripper '
                        'already holds object_name, skip observe/grasp and directly place it into the '
                        'remembered container. The sequence is complete only after placement and a '
                        'half-speed return home.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'object_name': {
                                'type': 'string',
                                'description': 'Object to grasp first.',
                            },
                            'container_name': {
                                'type': 'string',
                                'description': 'Open container name, for example box or 箱子.',
                            },
                            'motion_speed': {
                                'type': 'number',
                                'description': 'Optional one-shot grasp speed scaling from 0.0 to 1.0.',
                            },
                        },
                        'required': ['object_name', 'container_name'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'pick_all_fruits_into_container',
                    'description': (
                        'Loop until the vision model confirms that every visible object matching the original '
                        'collection task has been placed into the open container. For each iteration the server '
                        'observes the full scene from home, asks the vision model to interpret task_request '
                        'semantically against the current image, picks one still-outside target selected by the '
                        'model, places it into container_name, returns home at half the normal go_home speed, '
                        'then observes again. Use this for requests like 把所有水果放进箱子里 or put all fruits '
                        'into the box; pass the original user request as task_request so the server-side vision '
                        'model decides what counts as a task target. For "all fruits"/所有水果 tasks, fruit means '
                        'daily edible fruit, not botanical fruit; peppers/chili, tomatoes, cucumbers and eggplants '
                        'are excluded unless the user explicitly names them.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'container_name': {
                                'type': 'string',
                                'description': 'Open container name, for example box or 箱子.',
                            },
                            'max_items': {
                                'type': 'integer',
                                'description': 'Safety cap for the number of fruits to process. Default is 10.',
                            },
                            'motion_speed': {
                                'type': 'number',
                                'description': 'Optional one-shot grasp speed scaling from 0.0 to 1.0.',
                            },
                            'task_request': {
                                'type': 'string',
                                'description': (
                                    'Original user request, for example 把所有水果放进篮子里. '
                                    'The server-side vision model uses this exact wording to decide '
                                    'which visible objects are task targets and when the task is complete. '
                                    'For 所有水果/all fruits, use the daily edible-fruit meaning and exclude '
                                    'peppers/chili, tomatoes, cucumbers and eggplants unless explicitly named.'
                                ),
                            },
                        },
                        'required': ['container_name'],
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
