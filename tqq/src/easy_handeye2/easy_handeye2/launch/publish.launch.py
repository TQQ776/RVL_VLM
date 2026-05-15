from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arg_name = DeclareLaunchArgument('name')
    arg_publish_parent_frame = DeclareLaunchArgument('publish_parent_frame', default_value='')
    arg_auto_publish_optical_parent = DeclareLaunchArgument(
        'auto_publish_optical_parent',
        default_value='true',
    )
    arg_parent_lookup_timeout_sec = DeclareLaunchArgument(
        'parent_lookup_timeout_sec',
        default_value='3.0',
    )

    handeye_publisher = Node(package='easy_handeye2', executable='handeye_publisher', name='handeye_publisher', parameters=[{
        'name': LaunchConfiguration('name'),
        'publish_parent_frame': LaunchConfiguration('publish_parent_frame'),
        'auto_publish_optical_parent': LaunchConfiguration('auto_publish_optical_parent'),
        'parent_lookup_timeout_sec': LaunchConfiguration('parent_lookup_timeout_sec'),
    }])

    return LaunchDescription([
        arg_name,
        arg_publish_parent_frame,
        arg_auto_publish_optical_parent,
        arg_parent_lookup_timeout_sec,
        handeye_publisher,
    ])
