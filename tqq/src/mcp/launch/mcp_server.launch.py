"""Launch the MCP robot tool service node."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('mcp')
    default_params = f'{pkg_share}/config/mcp_server.yaml'
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        Node(
            package='mcp',
            executable='mcp_server',
            name='mcp_server',
            output='screen',
            parameters=[params_file],
            emulate_tty=True,
        ),
    ])
