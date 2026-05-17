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
from mcp.srv import MoveAxis, PlaceIntoContainer, PlaceRelative
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

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - fallback only for unusual ROS envs
    get_package_share_directory = None


def normalize_object_name(name: str) -> str:
    text = str(name or '').strip().strip(' \"\'“”‘’。，,.;；:：')
    return ' '.join(text.split())
