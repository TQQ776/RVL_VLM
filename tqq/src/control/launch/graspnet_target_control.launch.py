"""Launch GraspNet ROI grasp controller for FR3."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('control')
    default_params = f'{pkg_share}/config/graspnet_target_control.yaml'

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the GraspNet target controller parameter file.',
    )

    node = Node(
        package='control',
        executable='graspnet_target_controller',
        name='graspnet_target_controller',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        emulate_tty=True,
    )

    return LaunchDescription([
        params_file_arg,
        node,
    ])
