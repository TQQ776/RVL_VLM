"""Launch a RealSense camera with depth-to-color alignment enabled."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('franka_camera')
    default_params = f"{pkg_share}/config/realsense.yaml"

    camera_name_arg = DeclareLaunchArgument(
        'camera_name', default_value='camera',
        description='Node name and TF prefix for the RealSense camera.')
    camera_namespace_arg = DeclareLaunchArgument(
        'camera_namespace', default_value='camera',
        description='Topic namespace for the RealSense camera.')
    serial_no_arg = DeclareLaunchArgument(
        'serial_no', default_value='',
        description='Serial number of the device. Empty = first found.')
    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to the YAML parameter file.')

    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=LaunchConfiguration('camera_name'),
        namespace=LaunchConfiguration('camera_namespace'),
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'serial_no': ParameterValue(LaunchConfiguration('serial_no'), value_type=str)},
        ],
        emulate_tty=True,
    )

    return LaunchDescription([
        camera_name_arg,
        camera_namespace_arg,
        serial_no_arg,
        params_file_arg,
        realsense_node,
    ])
