import copy
from dataclasses import dataclass
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge, CvBridgeError
from franka_msgs.action import Grasp as GripperGrasp
from geometry_msgs.msg import PoseStamped, Quaternion
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
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory

import tf2_geometry_msgs  # noqa: F401  Registers geometry message transforms.


@dataclass
class RoiSelection:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class RoiEconomicGraspTarget:
    roi: RoiSelection
    grasp_score: float
    grasp_width: float
    grasp_height: float
    grasp_depth: float
    camera_pose: PoseStamped
    preview_camera_pose: PoseStamped
    execution_camera_pose: PoseStamped
    base_pose: PoseStamped
    roi_points: np.ndarray
    roi_colors: np.ndarray
    orientation_source: str


class EconomicGraspPrediction:
    def __init__(
        self,
        score: float,
        width: float,
        height: float,
        depth: float,
        rotation_matrix: np.ndarray,
        translation: np.ndarray,
        grasp_array: Optional[np.ndarray] = None,
    ) -> None:
        self.score = float(score)
        self.width = float(width)
        self.height = float(height)
        self.depth = float(depth)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)
        if grasp_array is None:
            grasp_array = np.concatenate([
                np.asarray([self.score, self.width, self.height, self.depth], dtype=np.float64),
                self.rotation_matrix.reshape(-1),
                self.translation,
                np.asarray([-1.0], dtype=np.float64),
            ])
        self.grasp_array = np.asarray(grasp_array, dtype=np.float64).reshape(17)


class EconomicGraspGroup:
    def __init__(self, grasp_array: np.ndarray) -> None:
        array = np.asarray(grasp_array, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] < 17:
            raise ValueError(f'EconomicGrasp prediction array must have at least 17 columns, got {array.shape}')
        self.array = array[:, :17]

    def __len__(self) -> int:
        return int(self.array.shape[0])

    @property
    def scores(self) -> np.ndarray:
        return self.array[:, 0]

    @property
    def widths(self) -> np.ndarray:
        return self.array[:, 1]

    @property
    def heights(self) -> np.ndarray:
        return self.array[:, 2]

    @property
    def depths(self) -> np.ndarray:
        return self.array[:, 3]

    @property
    def rotation_matrices(self) -> np.ndarray:
        return self.array[:, 4:13].reshape((-1, 3, 3))

    @property
    def translations(self) -> np.ndarray:
        return self.array[:, 13:16]

    def filtered(self, keep_mask: np.ndarray) -> 'EconomicGraspGroup':
        return EconomicGraspGroup(self.array[np.asarray(keep_mask, dtype=bool)])

    def sort_by_score(self) -> 'EconomicGraspGroup':
        if len(self) == 0:
            return self
        order = np.argsort(self.scores)[::-1]
        self.array = self.array[order]
        return self

    def nms(self, translation_thresh: float, rotation_thresh_rad: float) -> 'EconomicGraspGroup':
        if len(self) <= 1:
            return self
        order = np.argsort(self.scores)[::-1]
        keep = []
        rotations = self.rotation_matrices
        translations = self.translations
        for index in order:
            should_keep = True
            for kept in keep:
                if np.linalg.norm(translations[index] - translations[kept]) > translation_thresh:
                    continue
                delta = rotations[index].T.dot(rotations[kept])
                cos_angle = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
                angle = math.acos(float(cos_angle))
                if angle <= rotation_thresh_rad:
                    should_keep = False
                    break
            if should_keep:
                keep.append(int(index))
        return EconomicGraspGroup(self.array[keep])

    def best(self) -> EconomicGraspPrediction:
        if len(self) == 0:
            raise RuntimeError('no grasps available')
        row = self.array[0]
        return EconomicGraspPrediction(
            score=row[0],
            width=row[1],
            height=row[2],
            depth=row[3],
            rotation_matrix=row[4:13].reshape(3, 3),
            translation=row[13:16],
            grasp_array=row.copy(),
        )


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


class EconomicCollisionDetector:
    def __init__(self, scene_points: np.ndarray, voxel_size: float = 0.005) -> None:
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = float(voxel_size)
        points = np.asarray(scene_points, dtype=np.float32).reshape((-1, 3))
        if len(points) == 0:
            self.scene_points = points
            return
        coords = np.round(points / max(self.voxel_size, 1e-6)).astype(np.int64)
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        self.scene_points = points[np.sort(unique_idx)]

    def detect(self, grasp_group: EconomicGraspGroup, approach_dist: float, collision_thresh: float) -> np.ndarray:
        if len(grasp_group) == 0 or len(self.scene_points) == 0:
            return np.zeros((len(grasp_group),), dtype=bool)
        approach_dist = max(float(approach_dist), self.finger_width)
        translations = grasp_group.translations
        rotations = grasp_group.rotation_matrices
        heights = grasp_group.heights[:, np.newaxis]
        depths = grasp_group.depths[:, np.newaxis]
        widths = grasp_group.widths[:, np.newaxis]
        targets = self.scene_points[np.newaxis, :, :] - translations[:, np.newaxis, :]
        targets = np.matmul(targets, rotations)

        height_mask = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        finger_x_mask = (targets[:, :, 0] > depths - self.finger_length) & (targets[:, :, 0] < depths)
        left_y_mask = (targets[:, :, 1] > -(widths / 2 + self.finger_width)) & (targets[:, :, 1] < -widths / 2)
        right_y_mask = (targets[:, :, 1] < (widths / 2 + self.finger_width)) & (targets[:, :, 1] > widths / 2)
        inner_y_mask = (targets[:, :, 1] > -(widths / 2 + self.finger_width)) & (targets[:, :, 1] < (widths / 2 + self.finger_width))
        bottom_x_mask = (
            (targets[:, :, 0] <= depths - self.finger_length)
            & (targets[:, :, 0] > depths - self.finger_length - self.finger_width)
        )
        shifting_x_mask = (
            (targets[:, :, 0] <= depths - self.finger_length - self.finger_width)
            & (targets[:, :, 0] > depths - self.finger_length - self.finger_width - approach_dist)
        )

        left_mask = height_mask & finger_x_mask & left_y_mask
        right_mask = height_mask & finger_x_mask & right_y_mask
        bottom_mask = height_mask & inner_y_mask & bottom_x_mask
        shifting_mask = height_mask & inner_y_mask & shifting_x_mask
        global_mask = left_mask | right_mask | bottom_mask | shifting_mask

        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size ** 3)).reshape(-1)
        bottom_volume = (heights * (widths + 2 * self.finger_width) * self.finger_width / (self.voxel_size ** 3)).reshape(-1)
        shifting_volume = (heights * (widths + 2 * self.finger_width) * approach_dist / (self.voxel_size ** 3)).reshape(-1)
        volume = left_right_volume * 2 + bottom_volume + shifting_volume
        global_iou = global_mask.sum(axis=1) / (volume + 1e-6)
        return global_iou > float(collision_thresh)


class EconomicGraspBackend:
    """Use third_party/EconomicGrasp inference while keeping ROS + MoveIt control."""

    def __init__(self, node) -> None:
        self.node = node
        self.net = None
        self.device = None
        self._load_error = ''

    def ready(self) -> Tuple[bool, str]:
        if self.net is not None:
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
        repo_dir = os.path.expanduser(self.node.economic_grasp_repo_dir)
        checkpoint_path = os.path.expanduser(self.node.economic_grasp_checkpoint_path)
        if not os.path.isdir(repo_dir):
            raise RuntimeError(f'economic_grasp_repo_dir does not exist: {repo_dir}')
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                'EconomicGrasp checkpoint is missing. Download economicgrasp_realsense.tar '
                'from the EconomicGrasp release page and set economic_grasp_checkpoint_path. '
                f'Current value: {checkpoint_path or "<empty>"}'
            )

        for path in (
            repo_dir,
            os.path.join(repo_dir, 'models'),
            os.path.join(repo_dir, 'dataset'),
            os.path.join(repo_dir, 'utils'),
            os.path.join(repo_dir, 'libs', 'pointnet2'),
            os.path.join(repo_dir, 'libs', 'knn'),
        ):
            if path and path not in sys.path:
                sys.path.insert(0, path)

        old_argv = sys.argv[:]
        sys.argv = [
            'economic_grasp_roi_backend',
            '--dataset_root',
            repo_dir,
            '--camera',
            'realsense',
            '--checkpoint_path',
            checkpoint_path,
            '--save_dir',
            os.path.join(repo_dir, 'tmp_roi_outputs'),
            '--test_mode',
            'seen',
            '--num_point',
            str(self.node.economic_grasp_num_point),
            '--m_point',
            str(self.node.economic_grasp_m_point),
            '--num_view',
            str(self.node.economic_grasp_num_view),
            '--num_angle',
            str(self.node.economic_grasp_num_angle),
            '--num_depth',
            str(self.node.economic_grasp_num_depth),
            '--grasp_max_width',
            str(self.node.economic_grasp_grasp_max_width),
            '--graspness_threshold',
            str(self.node.economic_grasp_graspness_threshold),
            '--voxel_size',
            str(self.node.economic_grasp_voxel_size),
            '--collision_thresh',
            str(self.node.economic_grasp_collision_thresh),
        ]
        try:
            import torch
            from models.economicgrasp import economicgrasp, pred_decode
        except Exception as exc:
            raise RuntimeError(
                'Failed to import EconomicGrasp. Install its Python dependencies first: '
                'torch, MinkowskiEngine, pointnet2, knn, scipy, Pillow. '
                f'Original error: {exc}'
            ) from exc
        finally:
            sys.argv = old_argv

        self.torch = torch
        self.pred_decode = pred_decode
        requested_device = self.node.economic_grasp_device.strip()
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            self.node.get_logger().warn(
                f'Requested EconomicGrasp device {requested_device}, but CUDA is not available; using cpu.'
            )
            requested_device = 'cpu'
        self.device = torch.device(requested_device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))

        net = economicgrasp(
            seed_feat_dim=512,
            is_training=False,
            voxel_size=self.node.economic_grasp_voxel_size,
        )
        net.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        net.load_state_dict(state_dict)
        net.eval()
        self.net = net
        epoch = checkpoint.get('epoch', -1) if isinstance(checkpoint, dict) else -1
        self.node._publish_status(
            f'EconomicGrasp backend loaded from {repo_dir}, checkpoint={checkpoint_path}, '
            f'epoch={epoch}, device={self.device}'
        )

    def infer(self, points: np.ndarray, colors: np.ndarray):
        ok, message = self.ready()
        if not ok:
            raise RuntimeError(message)
        if points.shape[0] < self.node.economic_grasp_min_points:
            raise RuntimeError(
                f'not enough ROI point-cloud points for EconomicGrasp: '
                f'{points.shape[0]} < {self.node.economic_grasp_min_points}'
            )

        sampled_points, sampled_colors = self._sample_points(points, colors)
        torch = self.torch
        batch_data = {
            'point_clouds': torch.from_numpy(sampled_points[np.newaxis].astype(np.float32)).to(self.device),
            'cloud_colors': torch.from_numpy(sampled_colors[np.newaxis].astype(np.float32)).to(self.device),
            'coordinates_for_voxel': [
                torch.from_numpy(
                    (sampled_points.astype(np.float32) / self.node.economic_grasp_voxel_size)
                ).to(self.device)
            ],
        }
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = self.pred_decode(end_points)
        gg = EconomicGraspGroup(grasp_preds[0].detach().cpu().numpy())
        if len(gg) == 0:
            raise RuntimeError('EconomicGrasp produced no valid grasp')

        if self.node.economic_grasp_collision_thresh > 0.0:
            gg = self._collision_filter(gg, points)
            if len(gg) == 0:
                raise RuntimeError('EconomicGrasp produced no grasp after collision filtering')

        if self.node.economic_grasp_use_nms:
            gg = gg.nms(
                self.node.economic_grasp_nms_translation_thresh,
                math.radians(self.node.economic_grasp_nms_rotation_thresh_deg),
            )
        gg = self._center_filter(gg, points)
        gg = gg.sort_by_score()
        if len(gg) == 0:
            raise RuntimeError('EconomicGrasp produced no valid grasp after NMS')
        return gg.best()

    def _sample_points(self, points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points = points.astype(np.float32, copy=False)
        colors = np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False)
        num_point = int(self.node.economic_grasp_num_point)
        if len(points) >= num_point:
            indexes = np.random.choice(len(points), num_point, replace=False)
        else:
            base_indexes = np.arange(len(points))
            extra_indexes = np.random.choice(len(points), num_point - len(points), replace=True)
            indexes = np.concatenate([base_indexes, extra_indexes], axis=0)
        return points[indexes], colors[indexes]

    def _collision_filter(self, gg: EconomicGraspGroup, points: np.ndarray) -> EconomicGraspGroup:
        try:
            detector = EconomicCollisionDetector(
                points.astype(np.float32, copy=False),
                voxel_size=self.node.economic_grasp_voxel_size,
            )
            collision_mask = detector.detect(
                gg,
                approach_dist=self.node.economic_grasp_collision_approach_dist,
                collision_thresh=self.node.economic_grasp_collision_thresh,
            )
            return gg.filtered(~collision_mask)
        except Exception as exc:
            self.node.get_logger().warn(f'EconomicGrasp collision filtering failed; using raw grasps: {exc}')
            return gg

    def _center_filter(self, gg: EconomicGraspGroup, points: np.ndarray) -> EconomicGraspGroup:
        if not self.node.grasp_center_filter_enabled or len(gg) == 0 or len(points) == 0:
            return gg

        max_offset = float(self.node.grasp_center_max_xy_offset_m)
        if max_offset <= 0.0:
            return gg

        center = np.median(np.asarray(points, dtype=np.float64).reshape((-1, 3)), axis=0)
        translations = gg.translations
        offsets = np.linalg.norm(translations[:, :2] - center[:2], axis=1)
        keep_mask = offsets <= max_offset
        kept_count = int(np.count_nonzero(keep_mask))
        if kept_count == 0:
            best_offset = float(np.min(offsets)) if offsets.size else float('nan')
            self.node.get_logger().warn(
                'EconomicGrasp center filter would remove all candidates; '
                f'keeping raw candidates. center_xy=({center[0]:.3f}, {center[1]:.3f}), '
                f'best_offset={best_offset:.3f}m, max_offset={max_offset:.3f}m'
            )
            return gg

        self.node._publish_status(
            'EconomicGrasp center filter kept '
            f'{kept_count}/{len(gg)} candidates; '
            f'center_xy=({center[0]:.3f}, {center[1]:.3f}), '
            f'max_offset={max_offset:.3f}m'
        )
        return gg.filtered(keep_mask)

class RoiEconomicGraspController(Node):
    """Select an ROI in the camera image, estimate an EconomicGrasp 6D grasp, then execute it."""

    def __init__(self) -> None:
        super().__init__('roi_economic_grasp_controller')
        self.callback_group = ReentrantCallbackGroup()

        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.backend = EconomicGraspBackend(self)

        self.latest_color: Optional[np.ndarray] = None
        self.latest_color_header = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_header = None
        self.latest_depth_encoding = ''
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_joint_state: Optional[JointState] = None
        self.target_class_name = ''
        self.pending_motion_speed_override: Optional[float] = None
        self.last_target_command_time = 0.0

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
        self.create_subscription(
            String,
            self.target_command_topic,
            self.target_command_callback,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            self.api_detections_topic,
            self.api_detections_callback,
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
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            self.cartesian_path_service,
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

        self.ui_thread = threading.Thread(target=self._ui_loop, daemon=True)
        self.ui_thread.start()

        self.get_logger().info(
            'ROI EconomicGrasp controller ready. '
            f'color={self.color_topic}, depth={self.depth_topic}, '
            f'base_frame={self.base_frame}, ee={self.end_effector_frame}, '
            f'api_bbox={self.enable_api_bbox}'
        )
        self._publish_status(
            'ROI EconomicGrasp ready: drag a rectangle manually, or publish an API bbox '
            f'to {self.api_detections_topic} and target to {self.target_command_topic}.'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('api_detections_topic', '/mcp_omni_client/api_detections_json')
        self.declare_parameter('target_command_topic', '/economic_grasp_roi/target_class_name')
        self.declare_parameter('enable_api_bbox', True)
        self.declare_parameter('auto_execute_api_bbox', True)
        self.declare_parameter('api_bbox_target_timeout_sec', 5.0)
        self.declare_parameter('api_bbox_depth_percentile', 35.0)
        self.declare_parameter('api_bbox_depth_margin_m', 0.08)
        self.declare_parameter('api_bbox_erode_px', 1)
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('use_latest_tf', True)
        self.declare_parameter('end_effector_frame', 'fr3_hand_tcp')
        self.declare_parameter('move_group_name', 'fr3_arm')
        self.declare_parameter('ik_service', '/compute_ik')
        self.declare_parameter('cartesian_path_service', '/compute_cartesian_path')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('trajectory_action', '/fr3_arm_controller/follow_joint_trajectory')
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
        self.declare_parameter('staged_grasp_enabled', True)
        self.declare_parameter('pre_grasp_lift_m', 0.08)
        self.declare_parameter('cartesian_descend_m', 0.08)
        self.declare_parameter('cartesian_max_step_m', 0.005)
        self.declare_parameter('cartesian_jump_threshold', 0.0)
        self.declare_parameter('cartesian_min_fraction', 0.95)
        self.declare_parameter('cartesian_descend_duration_sec', 4.0)
        self.declare_parameter('roi_window_name', 'EconomicGrasp ROI Selector')
        self.declare_parameter('roi_padding_px', 8)
        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('economic_grasp_repo_dir', '/home/tqq/TQQ_ws/third_party/EconomicGrasp')
        self.declare_parameter(
            'economic_grasp_checkpoint_path',
            '/home/tqq/TQQ_ws/third_party/EconomicGrasp/checkpoints/economicgrasp_realsense.tar',
        )
        self.declare_parameter('economic_grasp_min_points', 200)
        self.declare_parameter('economic_grasp_num_point', 20000)
        self.declare_parameter('economic_grasp_m_point', 1024)
        self.declare_parameter('economic_grasp_num_view', 300)
        self.declare_parameter('economic_grasp_num_angle', 12)
        self.declare_parameter('economic_grasp_num_depth', 4)
        self.declare_parameter('economic_grasp_grasp_max_width', 0.1)
        self.declare_parameter('economic_grasp_graspness_threshold', 0.1)
        self.declare_parameter('economic_grasp_voxel_size', 0.005)
        self.declare_parameter('economic_grasp_collision_thresh', 0.01)
        self.declare_parameter('economic_grasp_collision_approach_dist', 0.05)
        self.declare_parameter('economic_grasp_use_nms', True)
        self.declare_parameter('economic_grasp_nms_translation_thresh', 0.03)
        self.declare_parameter('economic_grasp_nms_rotation_thresh_deg', 30.0)
        self.declare_parameter('grasp_center_filter_enabled', True)
        self.declare_parameter('grasp_center_max_xy_offset_m', 0.025)
        self.declare_parameter('economic_grasp_device', 'cuda:0')
        self.declare_parameter('economic_grasp_orientation_mode', 'economic_grasp')
        self.declare_parameter('economic_grasp_fallback_to_current_orientation', False)
        self.declare_parameter('economic_grasp_tcp_rotation_rpy_grasp', [0.0, 1.57079632679, 0.0])
        self.declare_parameter('economic_grasp_tcp_offset_xyz_grasp', [0.0, 0.0, 0.0])
        self.declare_parameter('target_offset_xyz_base', [0.0, 0.0, -0.02])
        self.declare_parameter('min_grasp_z_m', -10.0)
        self.declare_parameter('popup_preview_before_execute', True)
        self.declare_parameter('popup_preview_require_confirmation', False)
        self.declare_parameter('popup_preview_max_points', 20000)
        self.declare_parameter('popup_preview_frame_size_m', 0.08)
        self.declare_parameter('popup_preview_window_title', 'EconomicGrasp ROI grasp preview')
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
        self.api_detections_topic = str(self.get_parameter('api_detections_topic').value)
        self.target_command_topic = str(self.get_parameter('target_command_topic').value)
        self.enable_api_bbox = bool(self.get_parameter('enable_api_bbox').value)
        self.auto_execute_api_bbox = bool(self.get_parameter('auto_execute_api_bbox').value)
        self.api_bbox_target_timeout_sec = float(
            self.get_parameter('api_bbox_target_timeout_sec').value
        )
        self.api_bbox_depth_percentile = float(
            self.get_parameter('api_bbox_depth_percentile').value
        )
        self.api_bbox_depth_margin_m = float(
            self.get_parameter('api_bbox_depth_margin_m').value
        )
        self.api_bbox_erode_px = max(0, int(self.get_parameter('api_bbox_erode_px').value))
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.camera_frame_override = str(self.get_parameter('camera_frame').value)
        self.use_latest_tf = bool(self.get_parameter('use_latest_tf').value)
        self.end_effector_frame = str(self.get_parameter('end_effector_frame').value)
        self.move_group_name = str(self.get_parameter('move_group_name').value)
        self.ik_service = str(self.get_parameter('ik_service').value)
        self.cartesian_path_service = str(self.get_parameter('cartesian_path_service').value)
        self.move_group_action = str(self.get_parameter('move_group_action').value)
        self.trajectory_action = str(self.get_parameter('trajectory_action').value)
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
        self.staged_grasp_enabled = bool(self.get_parameter('staged_grasp_enabled').value)
        self.pre_grasp_lift_m = max(0.0, float(self.get_parameter('pre_grasp_lift_m').value))
        self.cartesian_descend_m = max(0.0, float(self.get_parameter('cartesian_descend_m').value))
        self.cartesian_max_step_m = max(0.001, float(self.get_parameter('cartesian_max_step_m').value))
        self.cartesian_jump_threshold = float(self.get_parameter('cartesian_jump_threshold').value)
        self.cartesian_min_fraction = min(
            1.0,
            max(0.0, float(self.get_parameter('cartesian_min_fraction').value)),
        )
        self.cartesian_descend_duration_sec = max(
            0.1,
            float(self.get_parameter('cartesian_descend_duration_sec').value),
        )
        self.roi_window_name = str(self.get_parameter('roi_window_name').value)
        self.roi_padding_px = max(0, int(self.get_parameter('roi_padding_px').value))
        self.depth_unit_scale = float(self.get_parameter('depth_unit_scale').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.economic_grasp_repo_dir = str(self.get_parameter('economic_grasp_repo_dir').value)
        self.economic_grasp_checkpoint_path = str(self.get_parameter('economic_grasp_checkpoint_path').value)
        self.economic_grasp_min_points = int(self.get_parameter('economic_grasp_min_points').value)
        self.economic_grasp_num_point = int(self.get_parameter('economic_grasp_num_point').value)
        self.economic_grasp_m_point = int(self.get_parameter('economic_grasp_m_point').value)
        self.economic_grasp_num_view = int(self.get_parameter('economic_grasp_num_view').value)
        self.economic_grasp_num_angle = int(self.get_parameter('economic_grasp_num_angle').value)
        self.economic_grasp_num_depth = int(self.get_parameter('economic_grasp_num_depth').value)
        self.economic_grasp_grasp_max_width = float(
            self.get_parameter('economic_grasp_grasp_max_width').value
        )
        self.economic_grasp_graspness_threshold = float(
            self.get_parameter('economic_grasp_graspness_threshold').value
        )
        self.economic_grasp_voxel_size = float(self.get_parameter('economic_grasp_voxel_size').value)
        self.economic_grasp_collision_thresh = float(
            self.get_parameter('economic_grasp_collision_thresh').value
        )
        self.economic_grasp_collision_approach_dist = float(
            self.get_parameter('economic_grasp_collision_approach_dist').value
        )
        self.economic_grasp_use_nms = bool(self.get_parameter('economic_grasp_use_nms').value)
        self.economic_grasp_nms_translation_thresh = float(
            self.get_parameter('economic_grasp_nms_translation_thresh').value
        )
        self.economic_grasp_nms_rotation_thresh_deg = float(
            self.get_parameter('economic_grasp_nms_rotation_thresh_deg').value
        )
        self.grasp_center_filter_enabled = bool(
            self.get_parameter('grasp_center_filter_enabled').value
        )
        self.grasp_center_max_xy_offset_m = float(
            self.get_parameter('grasp_center_max_xy_offset_m').value
        )
        self.economic_grasp_device = str(self.get_parameter('economic_grasp_device').value)
        self.economic_grasp_orientation_mode = str(
            self.get_parameter('economic_grasp_orientation_mode').value
        ).strip().lower()
        self.economic_grasp_fallback_to_current_orientation = bool(
            self.get_parameter('economic_grasp_fallback_to_current_orientation').value
        )
        self.economic_grasp_tcp_rotation_rpy_grasp = self._float_list(
            self.get_parameter('economic_grasp_tcp_rotation_rpy_grasp').value,
            3,
            'economic_grasp_tcp_rotation_rpy_grasp',
        )
        self.economic_grasp_tcp_rotation_matrix = self._rpy_to_rotation_matrix(
            self.economic_grasp_tcp_rotation_rpy_grasp
        )
        self.economic_grasp_tcp_offset_xyz_grasp = self._float_list(
            self.get_parameter('economic_grasp_tcp_offset_xyz_grasp').value,
            3,
            'economic_grasp_tcp_offset_xyz_grasp',
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
        self.popup_preview_require_confirmation = bool(
            self.get_parameter('popup_preview_require_confirmation').value
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
        if self.economic_grasp_orientation_mode not in ('economic_grasp', 'current', 'yaw_only'):
            raise ValueError(
                'economic_grasp_orientation_mode must be economic_grasp, current, or yaw_only'
            )

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
        target_name, motion_speed = self._parse_target_command(msg.data)
        with self.state_lock:
            self.target_class_name = target_name
            self.pending_motion_speed_override = motion_speed
            self.last_target_command_time = time.monotonic() if target_name else 0.0
        if target_name:
            speed_text = (
                f'; one-shot speed={motion_speed:.3f}'
                if motion_speed is not None
                else ''
            )
            self._publish_status(
                f'API bbox target command received: {target_name}{speed_text}; '
                'waiting for matching API detection.'
            )
        else:
            self._publish_status('API bbox target command cleared.')

    def api_detections_callback(self, msg: String) -> None:
        if not self.enable_api_bbox:
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Ignoring malformed API detections JSON: {exc}')
            return
        detections = payload.get('detections', [])
        if not isinstance(detections, list):
            return
        with self.state_lock:
            target_name = self.target_class_name.strip()
            target_time = self.last_target_command_time
            motion_speed = self.pending_motion_speed_override
        if not target_name:
            return
        if (
            self.api_bbox_target_timeout_sec > 0.0
            and target_time > 0.0
            and time.monotonic() - target_time > self.api_bbox_target_timeout_sec
        ):
            self._publish_status(
                f'API bbox for target "{target_name}" ignored: target command is stale.'
            )
            return
        detection = self._select_api_detection(detections, target_name)
        if detection is None:
            return
        roi = self._roi_from_api_detection(detection)
        if roi is None:
            return
        started = self._start_api_roi(roi, detection, motion_speed)
        if started:
            with self.state_lock:
                self.pending_motion_speed_override = None

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

    def _select_api_detection(self, detections: Sequence[Dict], target_name: str) -> Optional[Dict]:
        target_lower = target_name.strip().lower()
        candidates = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            bbox = detection.get('bbox_xyxy', [])
            if len(bbox) != 4:
                continue
            class_name = str(detection.get('class_name', '')).strip()
            if target_lower and class_name.lower() != target_lower:
                continue
            candidates.append(detection)
        if not candidates:
            return None
        return max(candidates, key=lambda item: float(item.get('confidence', 0.0)))

    def _roi_from_api_detection(self, detection: Dict) -> Optional[RoiSelection]:
        bbox = detection.get('bbox_xyxy', [])
        if len(bbox) != 4:
            return None
        x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
        roi = self._normalize_roi((x0, y0), (x1, y1))
        if roi is None:
            self.get_logger().warn(f'Ignoring invalid API bbox: {bbox}')
        return roi

    def _start_api_roi(
        self,
        roi: RoiSelection,
        detection: Dict,
        motion_speed: Optional[float],
    ) -> bool:
        with self.ui_lock:
            if self.worker_running:
                self._publish_status('API bbox ignored: EconomicGrasp worker is already running.')
                return False
            self.selected_roi = copy.deepcopy(roi)
            self.worker_running = True
        class_name = str(detection.get('class_name', '')).strip() or 'object'
        confidence = float(detection.get('confidence', 0.0))
        self._publish_status(
            f'API bbox selected for EconomicGrasp: {class_name} conf={confidence:.2f} '
            f'roi=({roi.x0},{roi.y0},{roi.x1},{roi.y1}).'
        )
        thread = threading.Thread(
            target=self._roi_worker,
            args=(roi, 'api_bbox', motion_speed),
            daemon=True,
        )
        thread.start()
        return True

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
                        'Press Enter or g to run EconomicGrasp.'
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
        status = 'running EconomicGrasp...' if worker_running else 'drag ROI, Enter/g run, c clear, q close'
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
                self._publish_status('EconomicGrasp ROI worker is already running.')
                return
            roi = copy.deepcopy(self.selected_roi)
            if roi is None:
                self._publish_status('No ROI selected yet.')
                return
            self.worker_running = True
        thread = threading.Thread(target=self._roi_worker, args=(roi,), daemon=True)
        thread.start()

    def _roi_worker(
        self,
        roi: RoiSelection,
        source: str = 'manual_roi',
        motion_speed: Optional[float] = None,
    ) -> None:
        try:
            target = self._build_grasp_target(roi, source=source)
            if target is None:
                return
            self.gripper_6d_pose_camera_pub.publish(target.preview_camera_pose)
            self.gripper_6d_pose_base_pub.publish(target.base_pose)
            self._publish_status(
                f'ROI EconomicGrasp source={source} grasp_score={target.grasp_score:.3f} '
                f'width={target.grasp_width:.3f}m '
                f'base=({target.base_pose.pose.position.x:.3f}, '
                f'{target.base_pose.pose.position.y:.3f}, {target.base_pose.pose.position.z:.3f})'
            )
            self._publish_gripper_6d_status(target)
            should_execute = self.auto_execute_api_bbox if source == 'api_bbox' else True
            if should_execute and not self._preview_and_confirm(target):
                self._publish_status('ROI grasp canceled before motion; gripper will not close.')
                return
            if not should_execute:
                self._publish_status('ROI EconomicGrasp target generated without execution.')
                return
            with self._temporary_motion_speed(motion_speed):
                moved = self._execute_grasp_motion(target.base_pose)
            if moved:
                self._close_gripper_for_grasp()
            else:
                self._publish_status('ROI grasp motion did not report success; gripper will not close.')
        finally:
            with self.ui_lock:
                self.worker_running = False

    def _build_grasp_target(
        self,
        roi: RoiSelection,
        source: str = 'manual_roi',
    ) -> Optional[RoiEconomicGraspTarget]:
        snapshot = self._snapshot_rgbd()
        if snapshot is None:
            return None
        color, depth, depth_encoding, depth_header, camera_info = snapshot
        points, colors = self._roi_point_cloud(
            color,
            depth,
            depth_encoding,
            camera_info,
            roi,
            filter_foreground=(source == 'api_bbox'),
        )
        if points is None or colors is None:
            return None
        try:
            grasp = self.backend.infer(points, colors)
        except Exception as exc:
            self.get_logger().error(f'EconomicGrasp inference failed: {exc}')
            self._publish_status(f'EconomicGrasp inference failed: {exc}')
            return None
        camera_pose = self._grasp_to_camera_pose(grasp, depth_header)
        if camera_pose is None:
            return None
        try:
            raw_base_pose = self.tf_buffer.transform(
                camera_pose,
                self.base_frame,
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().error(
                f'Cannot transform EconomicGrasp pose {camera_pose.header.frame_id} -> {self.base_frame}: {exc}'
            )
            return None
        base_pose = copy.deepcopy(raw_base_pose)
        base_pose.pose.position.x += self.target_offset_xyz_base[0]
        base_pose.pose.position.y += self.target_offset_xyz_base[1]
        base_pose.pose.position.z = max(
            self.min_grasp_z_m,
            base_pose.pose.position.z + self.target_offset_xyz_base[2],
        )
        orientation_source = 'economic_grasp'
        if self.economic_grasp_orientation_mode == 'current':
            current_orientation = self._current_end_effector_orientation()
            if current_orientation is None:
                return None
            base_pose.pose.orientation = current_orientation
            orientation_source = 'current_end_effector'
        elif self.economic_grasp_orientation_mode == 'yaw_only':
            yaw_only_orientation = self._yaw_only_orientation(base_pose.pose.orientation)
            if yaw_only_orientation is None:
                return None
            base_pose.pose.orientation = yaw_only_orientation
            orientation_source = 'economic_grasp_yaw_only'
        preview_base_pose = copy.deepcopy(raw_base_pose)
        preview_base_pose.pose.orientation = base_pose.pose.orientation
        preview_camera_pose = self._transform_pose_or_copy(
            preview_base_pose,
            camera_pose.header.frame_id,
            camera_pose,
        )
        execution_camera_pose = self._transform_pose_or_copy(
            base_pose,
            camera_pose.header.frame_id,
            camera_pose,
        )
        return RoiEconomicGraspTarget(
            roi=roi,
            grasp_score=float(grasp.score),
            grasp_width=float(grasp.width),
            grasp_height=float(grasp.height),
            grasp_depth=float(grasp.depth),
            camera_pose=camera_pose,
            preview_camera_pose=preview_camera_pose,
            execution_camera_pose=execution_camera_pose,
            base_pose=base_pose,
            roi_points=points.astype(np.float32, copy=False),
            roi_colors=colors.astype(np.float32, copy=False),
            orientation_source=orientation_source,
        )

    def _preview_and_confirm(self, target: RoiEconomicGraspTarget) -> bool:
        if self.popup_preview_before_execute:
            if self.popup_preview_require_confirmation:
                try:
                    confirmed = self._show_open3d_preview(target)
                except Exception as exc:
                    message = f'Failed to open EconomicGrasp preview window: {exc}'
                    self.get_logger().error(message)
                    self._publish_status(message)
                    return False
                if not confirmed:
                    self._publish_status('EconomicGrasp ROI preview canceled by operator.')
                    return False
                self._publish_status('EconomicGrasp ROI preview confirmed by operator.')
                return True
            if not self._spawn_open3d_preview_process(target):
                self._publish_status('EconomicGrasp ROI preview failed to open; executing anyway.')
            else:
                self._publish_status('EconomicGrasp ROI preview opened; executing without manual confirmation.')
            return True
        self._publish_status('EconomicGrasp ROI preview confirmed by operator.')
        return True

    def _spawn_open3d_preview_process(self, target: RoiEconomicGraspTarget) -> bool:
        try:
            payload = self._open3d_preview_payload(target)
            fd, preview_path = tempfile.mkstemp(
                prefix='economic_grasp_preview_',
                suffix='.npz',
            )
            os.close(fd)
            np.savez_compressed(
                preview_path,
                points=payload['points'],
                colors=payload['colors'],
                origin=payload['origin'],
                rotation=payload['rotation'],
                grasp_array=payload['grasp_array'],
                frame_size=np.asarray([payload['frame_size']], dtype=np.float64),
                title=np.asarray([payload['title']]),
                economic_grasp_repo_dir=np.asarray([payload['economic_grasp_repo_dir']]),
            )
            preview_script = os.path.join(os.path.dirname(__file__), 'open3d_preview.py')
            log_file = open('/tmp/economic_grasp_open3d_preview.log', 'ab')
            subprocess.Popen(
                [
                    sys.executable,
                    preview_script,
                    preview_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=log_file,
                start_new_session=True,
                close_fds=True,
            )
            log_file.close()
            return True
        except Exception as exc:
            message = f'Failed to start EconomicGrasp preview process: {exc}'
            self.get_logger().error(message)
            self._publish_status(message)
            return False

    def _open3d_preview_payload(self, target: RoiEconomicGraspTarget) -> Dict:
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

        preview_pose = target.preview_camera_pose
        rotation = self._quaternion_to_rotation_matrix(preview_pose.pose.orientation)
        mesh_axis_to_tcp_axis = np.asarray([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float64)
        gripper_rotation = rotation.dot(mesh_axis_to_tcp_axis)
        translation = np.asarray([
            preview_pose.pose.position.x,
            preview_pose.pose.position.y,
            preview_pose.pose.position.z,
        ], dtype=np.float64)
        width = target.grasp_width if math.isfinite(target.grasp_width) else 0.06
        height = target.grasp_height if math.isfinite(target.grasp_height) else 0.02
        depth = target.grasp_depth if math.isfinite(target.grasp_depth) else 0.02
        grasp_array = np.concatenate([
            np.asarray([
                target.grasp_score,
                min(0.12, max(0.0, float(width))),
                min(0.08, max(0.004, float(height))),
                min(0.12, max(0.0, float(depth))),
            ], dtype=np.float64),
            gripper_rotation.reshape(-1),
            translation,
            np.asarray([-1.0], dtype=np.float64),
        ]).reshape(1, 17)
        title = (
            f'{self.popup_preview_window_title} - '
            f'score={target.grasp_score:.3f} width={target.grasp_width:.3f}m '
            f'pose={target.orientation_source}'
        )
        return {
            'points': points.astype(np.float32, copy=True),
            'colors': np.clip(colors, 0.0, 1.0).astype(np.float32, copy=True),
            'origin': translation.astype(np.float64, copy=True),
            'rotation': rotation.astype(np.float64, copy=True),
            'grasp_array': grasp_array.astype(np.float64, copy=True),
            'frame_size': max(0.01, self.popup_preview_frame_size_m),
            'title': title,
            'economic_grasp_repo_dir': os.path.expanduser(self.economic_grasp_repo_dir),
        }

    def _show_open3d_preview_nonblocking(self, target: RoiEconomicGraspTarget) -> None:
        try:
            self._show_open3d_preview(target)
        except Exception as exc:
            message = f'Failed to open EconomicGrasp preview window: {exc}'
            self.get_logger().error(message)
            self._publish_status(message)

    def _show_open3d_preview(self, target: RoiEconomicGraspTarget) -> bool:
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
        # Keep the point-cloud preview position on the raw EconomicGrasp pose,
        # while showing the same orientation constraint used for execution.
        preview_pose = target.preview_camera_pose
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
        gripper_geometries = self._create_open3d_gripper_geometries(preview_pose, target)

        title = (
            f'{self.popup_preview_window_title} - '
            f'score={target.grasp_score:.3f} width={target.grasp_width:.3f}m '
            f'pose={target.orientation_source}'
        )
        self._publish_status(
            'Showing ROI EconomicGrasp popup preview. Focus the Open3D window: '
            'press Enter to close, or press c/q/Esc to close.'
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
            [cloud, pose_frame, *gripper_geometries],
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

    def _create_open3d_gripper_geometries(
        self,
        pose: PoseStamped,
        target: RoiEconomicGraspTarget,
    ) -> List:
        GraspGroup = self._import_graspnet_api_grasp_group()
        rotation = self._quaternion_to_rotation_matrix(pose.pose.orientation)
        # graspnetAPI draws the gripper approach along local +X and the side
        # face normal along local +Y.  In the fr3_hand_tcp preview frame those
        # should align with +Z (blue) and +Y (green), respectively.
        mesh_axis_to_tcp_axis = np.asarray([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float64)
        gripper_rotation = rotation.dot(mesh_axis_to_tcp_axis)
        translation = np.asarray([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ], dtype=np.float64)
        width = target.grasp_width if math.isfinite(target.grasp_width) else 0.06
        height = target.grasp_height if math.isfinite(target.grasp_height) else 0.02
        depth = target.grasp_depth if math.isfinite(target.grasp_depth) else 0.02
        grasp_array = np.concatenate([
            np.asarray([
                target.grasp_score,
                min(0.12, max(0.0, float(width))),
                min(0.08, max(0.004, float(height))),
                min(0.12, max(0.0, float(depth))),
            ], dtype=np.float64),
            gripper_rotation.reshape(-1),
            translation,
            np.asarray([-1.0], dtype=np.float64),
        ]).reshape(1, 17)
        return GraspGroup(grasp_array).to_open3d_geometry_list()

    def _import_graspnet_api_grasp_group(self):
        if not hasattr(np, 'float'):
            setattr(np, 'float', float)
        third_party_dir = os.path.dirname(os.path.expanduser(self.economic_grasp_repo_dir))
        candidates = [
            os.path.join(third_party_dir, 'franka-graspnet-master', 'graspnetAPI'),
            os.path.join(os.path.expanduser(self.economic_grasp_repo_dir), 'graspnetAPI'),
        ]
        for path in candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        try:
            from graspnetAPI import GraspGroup
        except Exception as exc:
            raise RuntimeError(
                'Failed to import graspnetAPI.GraspGroup for official gripper preview. '
                'Install graspnetAPI or keep third_party/franka-graspnet-master/graspnetAPI available. '
                f'Original error: {exc}'
            ) from exc
        return GraspGroup

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
        filter_foreground: bool = False,
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
        if filter_foreground:
            roi_depths = depth_m[y0:y1, x0:x1]
            roi_valid_depths = roi_depths[np.isfinite(roi_depths)]
            roi_valid_depths = roi_valid_depths[
                (roi_valid_depths >= self.min_depth_m)
                & (roi_valid_depths <= self.max_depth_m)
            ]
            if roi_valid_depths.size > 0:
                percentile = min(100.0, max(0.0, self.api_bbox_depth_percentile))
                front_depth = float(np.percentile(roi_valid_depths, percentile))
                depth_limit = front_depth + max(0.0, self.api_bbox_depth_margin_m)
                mask &= depth_m <= depth_limit
            if self.api_bbox_erode_px > 0:
                roi_mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
                roi_mask[y0:y1, x0:x1] = 255
                kernel_size = self.api_bbox_erode_px * 2 + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                eroded = cv2.erode(roi_mask, kernel, iterations=1) > 0
                mask &= eroded
        ys, xs = np.nonzero(mask)
        if len(xs) < self.economic_grasp_min_points:
            self.get_logger().warn(
                f'ROI point cloud too small for EconomicGrasp: {len(xs)} points in roi {(x0, y0, x1, y1)}'
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
        tcp_rotation = grasp_rotation.dot(self.economic_grasp_tcp_rotation_matrix)
        offset = np.asarray(self.economic_grasp_tcp_offset_xyz_grasp, dtype=np.float64).reshape(3)
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

    def _yaw_only_orientation(self, predicted_orientation: Quaternion) -> Optional[Quaternion]:
        current_orientation = self._current_end_effector_orientation()
        if current_orientation is None:
            return None

        current_roll, current_pitch, _ = self._quaternion_to_rpy(current_orientation)
        _, _, predicted_yaw = self._quaternion_to_rpy(predicted_orientation)
        yaw_only_rotation = self._rpy_to_rotation_matrix([
            current_roll,
            current_pitch,
            predicted_yaw,
        ])
        return self._rotation_matrix_to_quaternion(yaw_only_rotation)

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
                f'{target_frame}; using raw EconomicGrasp camera pose for preview/log: {exc}'
            )
            return copy.deepcopy(fallback_pose)

    def _execute_grasp_motion(self, final_pose: PoseStamped) -> bool:
        if not self.staged_grasp_enabled:
            return self._execute_pose_motion(final_pose, 'roi economic_grasp target')

        pre_grasp_pose = copy.deepcopy(final_pose)
        pre_grasp_pose.pose.position.z += self.pre_grasp_lift_m
        descend_pose = copy.deepcopy(pre_grasp_pose)
        descend_pose.pose.position.z -= self.cartesian_descend_m

        if abs(descend_pose.pose.position.z - final_pose.pose.position.z) > 0.002:
            self._publish_status(
                'staged ROI grasp warning: pre_grasp_lift_m and cartesian_descend_m are not equal; '
                f'final grasp z={final_pose.pose.position.z:.3f}m, '
                f'Cartesian descend target z={descend_pose.pose.position.z:.3f}m'
            )

        self._publish_status(
            'staged ROI grasp step 1/2: MoveGroup to pre-grasp pose '
            f'z={pre_grasp_pose.pose.position.z:.3f}m '
            f'({self.pre_grasp_lift_m:.3f}m above final pose)'
        )
        if not self._execute_pose_motion(pre_grasp_pose, 'roi economic_grasp pre-grasp'):
            return False

        self._publish_status(
            'staged ROI grasp step 2/2: Cartesian descend along base -Z '
            f'{self.cartesian_descend_m:.3f}m to z={descend_pose.pose.position.z:.3f}m'
        )
        return self._execute_cartesian_descend(descend_pose)

    def _execute_cartesian_descend(self, target_pose: PoseStamped) -> bool:
        if self.plan_only:
            self._publish_status('plan_only=true; skipping Cartesian descend execution.')
            return False
        if not self.cartesian_path_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().error(f'Cartesian path service not available: {self.cartesian_path_service}')
            return False
        with self.state_lock:
            joint_state = copy.deepcopy(self.latest_joint_state)
        if joint_state is None:
            self.get_logger().error('No /joint_states received; cannot compute Cartesian descend.')
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

        future = self.cartesian_path_client.call_async(request)
        response = self._wait_for_future(
            future,
            self.service_wait_timeout_sec + self.allowed_planning_time + 1.0,
            'ROI Cartesian descend path',
        )
        if response is None:
            return False
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'Cartesian descend path failed with code {response.error_code.val}'
            )
            return False
        if response.fraction < self.cartesian_min_fraction:
            self.get_logger().error(
                f'Cartesian descend path incomplete: fraction={response.fraction:.3f}, '
                f'required={self.cartesian_min_fraction:.3f}'
            )
            return False
        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            self.get_logger().error('Cartesian descend path returned an empty trajectory.')
            return False

        self._time_parameterize_cartesian_trajectory(
            trajectory,
            self.cartesian_descend_duration_sec,
        )
        return self._execute_joint_trajectory(
            trajectory,
            'roi economic_grasp Cartesian descend',
            self.cartesian_descend_duration_sec + self.action_wait_timeout_sec,
        )

    def _execute_pose_motion(self, target_pose: PoseStamped, label: str) -> bool:
        motion_label = label
        joint_goal = self._compute_ik(target_pose, motion_label)
        if (
            joint_goal is None
            and self.economic_grasp_fallback_to_current_orientation
            and self.economic_grasp_orientation_mode == 'economic_grasp'
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

        first_positions = np.asarray(points[0].positions, dtype=np.float64)
        last_positions = np.asarray(points[-1].positions, dtype=np.float64)
        delta = last_positions - first_positions
        min_step = 0.05
        for index, point in enumerate(points):
            u = index / float(len(points) - 1)
            smooth_u = 3.0 * u * u - 2.0 * u * u * u
            smooth_du = 6.0 * u * (1.0 - u) / duration
            smooth_ddu = 6.0 * (1.0 - 2.0 * u) / (duration * duration)
            point.positions = (first_positions + delta * smooth_u).tolist()
            point.velocities = (delta * smooth_du).tolist()
            point.accelerations = (delta * smooth_ddu).tolist()
            point.time_from_start = self._duration_msg(max(min_step, duration * u))

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
            f'{label} trajectory goal',
        )
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'Trajectory controller rejected {label}.')
            return False

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            timeout_sec,
            f'{label} trajectory result',
        )
        if action_result is None:
            return False

        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self._publish_status(f'Cartesian trajectory {label} executed successfully.')
            return True

        self.get_logger().error(
            f'Trajectory failed for {label}: {result.error_code} {result.error_string}'
        )
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

    def _temporary_motion_speed(self, speed: Optional[float]):
        return _TemporaryMotionSpeed(self, speed)

    def _joint_goal_constraints(self, joint_goal: Dict[str, float]) -> Constraints:
        constraints = Constraints()
        constraints.name = 'roi_economic_grasp_ik_joint_goal'
        for name in self.arm_joints:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = joint_goal[name]
            joint_constraint.tolerance_above = self.goal_joint_tolerance
            joint_constraint.tolerance_below = self.goal_joint_tolerance
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        return constraints

    def _publish_gripper_6d_status(self, target: RoiEconomicGraspTarget) -> None:
        self._publish_status(
            'ROI EconomicGrasp gripper 6D pose '
            f'orientation_source={target.orientation_source}; '
            f'preview_camera[{self._format_pose_6d(target.preview_camera_pose)}]; '
            f'execution_base[{self._format_pose_6d(target.base_pose)}]'
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
    node = RoiEconomicGraspController()
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
