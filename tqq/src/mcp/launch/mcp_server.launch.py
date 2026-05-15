"""Launch the MCP robot tool service node."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context):
    params_file = LaunchConfiguration('params_file')
    optional_overrides = {}
    for name in (
        'omni_api_key_env',
        'omni_base_url',
        'omni_text_model',
        'vision_image_topic',
        'vision_show_window',
        'vision_window_name',
        'vision_save_images',
        'vision_output_dir',
        'vision_result_hold_sec',
        'api_detections_topic',
        'target_command_topic',
    ):
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            optional_overrides[name] = value

    parameters = [params_file]
    if optional_overrides:
        parameters.append(optional_overrides)

    return [
        Node(
            package='mcp',
            executable='mcp_server',
            name='mcp_server',
            output='screen',
            parameters=parameters,
            emulate_tty=True,
        ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('mcp')
    default_params = f'{pkg_share}/config/mcp_server.yaml'

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('omni_api_key_env', default_value=''),
        DeclareLaunchArgument('omni_base_url', default_value=''),
        DeclareLaunchArgument('omni_text_model', default_value=''),
        DeclareLaunchArgument('vision_image_topic', default_value=''),
        DeclareLaunchArgument('vision_show_window', default_value=''),
        DeclareLaunchArgument('vision_window_name', default_value=''),
        DeclareLaunchArgument('vision_save_images', default_value=''),
        DeclareLaunchArgument('vision_output_dir', default_value=''),
        DeclareLaunchArgument('vision_result_hold_sec', default_value=''),
        DeclareLaunchArgument('api_detections_topic', default_value=''),
        DeclareLaunchArgument('target_command_topic', default_value=''),
        OpaqueFunction(function=_launch_setup),
    ])
