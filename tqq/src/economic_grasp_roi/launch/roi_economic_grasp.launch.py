from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value='/home/tqq/TQQ_ws/tqq/src/economic_grasp_roi/config/roi_economic_grasp.yaml',
            description='Path to roi_economic_grasp_controller parameters.',
        ),
        Node(
            package='economic_grasp_roi',
            executable='roi_economic_grasp_controller',
            name='roi_economic_grasp_controller',
            output='screen',
            emulate_tty=True,
            parameters=[params_file],
        ),
    ])
