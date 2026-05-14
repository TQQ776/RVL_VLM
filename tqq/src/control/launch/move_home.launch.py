"""Launch a node that returns FR3 to the configured home joint posture."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('control')
    default_params = f'{pkg_share}/config/move_home.yaml'

    params_file = LaunchConfiguration('params_file')

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params),
    ]

    node = Node(
        package='control',
        executable='move_home',
        name='move_home',
        output='screen',
        parameters=[params_file],
        emulate_tty=True,
    )

    return LaunchDescription(args + [node])
