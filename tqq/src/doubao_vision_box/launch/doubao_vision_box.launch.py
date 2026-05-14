"""Launch the independent Doubao vision box node."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('doubao_vision_box')
    default_params = f'{pkg_share}/config/doubao_vision_box.yaml'

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the Doubao vision box parameter file.',
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/camera/color/image_raw',
        description='Camera color image topic.',
    )
    show_window_arg = DeclareLaunchArgument(
        'show_window',
        default_value='true',
        description='Show an OpenCV window with the annotated image.',
    )
    enable_stdin_arg = DeclareLaunchArgument(
        'enable_stdin',
        default_value='false',
        description='Read text commands from this terminal.',
    )
    text_popup_enabled_arg = DeclareLaunchArgument(
        'text_popup_enabled',
        default_value='true',
        description='Press text_popup_key to open a text input popup.',
    )
    text_popup_key_arg = DeclareLaunchArgument(
        'text_popup_key',
        default_value='t',
        description='Keyboard key used to open the text input popup.',
    )

    node = Node(
        package='doubao_vision_box',
        executable='doubao_vision_box',
        name='doubao_vision_box',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'image_topic': LaunchConfiguration('image_topic'),
                'show_window': ParameterValue(LaunchConfiguration('show_window'), value_type=bool),
                'enable_stdin': ParameterValue(LaunchConfiguration('enable_stdin'), value_type=bool),
                'text_popup_enabled': ParameterValue(
                    LaunchConfiguration('text_popup_enabled'),
                    value_type=bool,
                ),
                'text_popup_key': LaunchConfiguration('text_popup_key'),
            },
        ],
        emulate_tty=True,
    )

    return LaunchDescription([
        params_file_arg,
        image_topic_arg,
        show_window_arg,
        enable_stdin_arg,
        text_popup_enabled_arg,
        text_popup_key_arg,
        node,
    ])
