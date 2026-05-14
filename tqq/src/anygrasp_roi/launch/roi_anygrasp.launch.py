from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value='/home/tqq/TQQ_ws/tqq/src/anygrasp_roi/config/roi_anygrasp.yaml',
            description='Path to roi_anygrasp_controller parameters.',
        ),
        Node(
            package='anygrasp_roi',
            executable='roi_anygrasp_controller',
            name='roi_anygrasp_controller',
            output='screen',
            emulate_tty=True,
            parameters=[params_file],
        ),
    ])
