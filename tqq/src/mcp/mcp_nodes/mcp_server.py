import copy
import json
import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Grasp as GripperGrasp
from franka_msgs.action import Move as GripperMove
from geometry_msgs.msg import PoseStamped, Quaternion
from mcp.srv import MoveAxis, ObjectName
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
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
        self.create_subscription(
            String,
            self.detected_objects_topic,
            self.detected_objects_callback,
            10,
            callback_group=self.callback_group,
        )
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
            '~/go_home, ~/move_x_cm, ~/move_y_cm, ~/move_z_cm, '
            '~/list_yolo_objects, ~/grab_object, '
            '~/open_gripper, ~/close_gripper; '
            f'base_frame={self.base_frame}, ee={self.end_effector_frame}'
        )

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
        self.declare_parameter('yolo_detect_once_service', '/yolo/detect_once')
        self.declare_parameter('yolo_detection_wait_timeout_sec', 5.0)
        self.declare_parameter('target_command_settle_sec', 0.3)
        self.declare_parameter('detected_objects_max_age_sec', 2.0)
        self.declare_parameter('require_detected_before_grab', True)
        self.declare_parameter('require_target_command_subscriber', True)

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
        if motion_speed is None:
            return target_name
        return json.dumps(
            {
                'name': target_name,
                'motion_speed': motion_speed,
            },
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
