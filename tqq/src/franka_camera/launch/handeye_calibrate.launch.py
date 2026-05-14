"""All-in-one hand-eye calibration launch.

Starts:
  1. RealSense camera (with depth-color alignment)
  2. ArUco single-marker detector
  3. Franka MoveIt stack (optional, on by default)
  4. easy_handeye2 calibration server (with rqt UI)

Usage:
  ros2 launch franka_camera handeye_calibrate.launch.py robot_ip:=192.168.22.212
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    marker_id = LaunchConfiguration('marker_id')
    marker_size = LaunchConfiguration('marker_size')
    calibration_name = LaunchConfiguration('calibration_name')
    launch_moveit = LaunchConfiguration('launch_moveit')

    args = [
        DeclareLaunchArgument('robot_ip', default_value='192.168.22.212',
                              description='Franka robot IP.'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false',
                              description='Use fake hardware instead of real robot.'),
        DeclareLaunchArgument('marker_id', default_value='582',
                              description='ArUco marker ID to track.'),
        DeclareLaunchArgument('marker_size', default_value='0.15',
                              description='Real printed marker edge length (meters).'),
        DeclareLaunchArgument('calibration_name', default_value='fr3_d435i_handeye',
                              description='Name used to save calibration result.'),
        DeclareLaunchArgument('launch_moveit', default_value='true',
                              description='Whether to launch franka_fr3_moveit_config too.'),
    ]

    # 1. RealSense (no extra namespace -> topics under /camera/...)
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_camera'),
                                  'launch', 'realsense.launch.py'])
        ]),
        launch_arguments={'camera_namespace': '/'}.items(),
    )

    # 2. ArUco single-marker detector
    aruco_node = Node(
        package='aruco_ros',
        executable='single',
        name='aruco_single',
        parameters=[{
            'marker_id': marker_id,
            'marker_size': marker_size,
            'reference_frame': 'camera_color_optical_frame',
            'camera_frame': 'camera_color_optical_frame',
            'marker_frame': 'aruco_marker_frame',
            'image_is_rectified': True,
        }],
        remappings=[
            ('/image', '/camera/color/image_raw'),
            ('/camera_info', '/camera/color/camera_info'),
        ],
        output='screen',
    )

    # 3. Franka MoveIt (optional)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_fr3_moveit_config'),
                                  'launch', 'moveit.launch.py'])
        ]),
        launch_arguments={
            'robot_ip': robot_ip,
            'use_fake_hardware': use_fake_hardware,
        }.items(),
        condition=IfCondition(launch_moveit),
    )

    # 4. easy_handeye2 calibration server (eye-in-hand)
    handeye_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('easy_handeye2'),
                                  'launch', 'calibrate.launch.py'])
        ]),
        launch_arguments={
            'calibration_type': 'eye_in_hand',
            'name': calibration_name,
            'robot_base_frame': 'fr3_link0',
            'robot_effector_frame': 'fr3_hand_tcp',
            'tracking_base_frame': 'camera_color_optical_frame',
            'tracking_marker_frame': 'aruco_marker_frame',
        }.items(),
    )

    return LaunchDescription(args + [
        camera_launch,
        aruco_node,
        moveit_launch,
        handeye_launch,
    ])
