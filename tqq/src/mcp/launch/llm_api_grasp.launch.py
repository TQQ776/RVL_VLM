"""One-shot launch for LLM/API-vision controlled grasping on the FR3.

This launch file starts the same stack that was previously launched from
separate terminals:

RealSense -> MoveIt/FR3 -> hand-eye TF -> EconomicGrasp ROI controller
-> MCP tool server -> Qwen-Omni client.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _pkg_file(package_name: str, *parts: str) -> str:
    return '/'.join((get_package_share_directory(package_name), *parts))


def _include(package_name: str, launch_name: str, launch_arguments=None, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare(package_name), 'launch', launch_name])
        ]),
        launch_arguments=(launch_arguments or {}).items(),
        condition=condition,
    )


def _delayed(seconds: float, action):
    return TimerAction(period=seconds, actions=[action])


def generate_launch_description():
    launch_camera = LaunchConfiguration('launch_camera')
    launch_moveit = LaunchConfiguration('launch_moveit')
    launch_handeye = LaunchConfiguration('launch_handeye')
    launch_control = LaunchConfiguration('launch_control')
    launch_mcp_server = LaunchConfiguration('launch_mcp_server')
    launch_omni_client = LaunchConfiguration('launch_omni_client')
    launch_aux_camera = LaunchConfiguration('launch_aux_camera')
    launch_aux_camera_view = LaunchConfiguration('launch_aux_camera_view')

    args = [
        DeclareLaunchArgument('robot_ip', default_value='192.168.22.212'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        DeclareLaunchArgument('camera_namespace', default_value='camera'),
        DeclareLaunchArgument('camera_name', default_value='camera'),
        DeclareLaunchArgument(
            'camera_serial_no',
            default_value='327122079035',
            description='RealSense serial number for the hand-mounted D435i camera.',
        ),
        DeclareLaunchArgument('aux_camera_namespace', default_value='d435'),
        DeclareLaunchArgument('aux_camera_name', default_value='d435'),
        DeclareLaunchArgument(
            'aux_camera_serial_no',
            default_value='327122070172',
            description='RealSense serial number for the auxiliary D435 camera.',
        ),
        DeclareLaunchArgument('aux_camera_view_topic', default_value='/d435/d435/color/image_raw'),
        DeclareLaunchArgument('aux_camera_view_window_name', default_value='全局相机视野'),
        DeclareLaunchArgument('image_topic', default_value='/camera/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/camera/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera/color/camera_info'),
        DeclareLaunchArgument('api_detections_topic', default_value='/mcp_omni_client/api_detections_json'),
        DeclareLaunchArgument('realsense_params_file', default_value=_pkg_file('control', 'config', 'realsense_aligned_depth.yaml')),
        DeclareLaunchArgument('handeye_name', default_value='fr3_d435i_handeye'),
        DeclareLaunchArgument('handeye_publish_parent_frame', default_value=''),
        DeclareLaunchArgument('handeye_parent_lookup_timeout_sec', default_value='3.0'),
        DeclareLaunchArgument(
            'economic_grasp_params_file',
            default_value=_pkg_file('economic_grasp_roi', 'config', 'roi_economic_grasp.yaml'),
        ),
        DeclareLaunchArgument('mcp_server_params_file', default_value=_pkg_file('mcp', 'config', 'mcp_server.yaml')),
        DeclareLaunchArgument('mcp_omni_client_params_file', default_value=_pkg_file('mcp', 'config', 'mcp_omni_client.yaml')),
        DeclareLaunchArgument('push_to_talk_enabled', default_value='false'),
        DeclareLaunchArgument('push_to_talk_key', default_value='r'),
        DeclareLaunchArgument('stop_record_key', default_value='q'),
        DeclareLaunchArgument('text_popup_enabled', default_value='true'),
        DeclareLaunchArgument('text_popup_key', default_value='t'),
        DeclareLaunchArgument('text_popup_auto_open', default_value='true'),
        DeclareLaunchArgument('tts_engine', default_value='auto'),
        DeclareLaunchArgument('audio_device', default_value='default'),
        DeclareLaunchArgument('omni_text_model', default_value=''),
        DeclareLaunchArgument('omni_realtime_voice', default_value=''),
        DeclareLaunchArgument('omni_speech_rate', default_value=''),
        DeclareLaunchArgument('omni_speech_emotion', default_value=''),
        DeclareLaunchArgument('vision_show_window', default_value=''),
        DeclareLaunchArgument('vision_save_images', default_value=''),
        DeclareLaunchArgument('vision_output_dir', default_value=''),
        DeclareLaunchArgument('launch_camera', default_value='true'),
        DeclareLaunchArgument('launch_aux_camera', default_value='true'),
        DeclareLaunchArgument('launch_aux_camera_view', default_value='true'),
        DeclareLaunchArgument('launch_moveit', default_value='true'),
        DeclareLaunchArgument('launch_handeye', default_value='true'),
        DeclareLaunchArgument('launch_control', default_value='true'),
        DeclareLaunchArgument('launch_mcp_server', default_value='true'),
        DeclareLaunchArgument('launch_omni_client', default_value='true'),
    ]

    camera = _include(
        'franka_camera',
        'realsense.launch.py',
        {
            'camera_namespace': LaunchConfiguration('camera_namespace'),
            'camera_name': LaunchConfiguration('camera_name'),
            'serial_no': LaunchConfiguration('camera_serial_no'),
            'params_file': LaunchConfiguration('realsense_params_file'),
        },
        IfCondition(launch_camera),
    )

    aux_camera = _include(
        'franka_camera',
        'realsense.launch.py',
        {
            'camera_namespace': LaunchConfiguration('aux_camera_namespace'),
            'camera_name': LaunchConfiguration('aux_camera_name'),
            'serial_no': LaunchConfiguration('aux_camera_serial_no'),
            'params_file': LaunchConfiguration('realsense_params_file'),
        },
        IfCondition(launch_aux_camera),
    )

    aux_camera_view = Node(
        package='mcp',
        executable='camera_viewer',
        name='global_camera_view',
        parameters=[{
            'image_topic': LaunchConfiguration('aux_camera_view_topic'),
            'window_name': LaunchConfiguration('aux_camera_view_window_name'),
        }],
        output='screen',
        condition=IfCondition(launch_aux_camera_view),
    )

    moveit = _include(
        'franka_fr3_moveit_config',
        'moveit.launch.py',
        {
            'robot_ip': LaunchConfiguration('robot_ip'),
            'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
        },
        IfCondition(launch_moveit),
    )

    handeye = _include(
        'easy_handeye2',
        'publish.launch.py',
        {
            'name': LaunchConfiguration('handeye_name'),
            'publish_parent_frame': LaunchConfiguration('handeye_publish_parent_frame'),
            'parent_lookup_timeout_sec': LaunchConfiguration('handeye_parent_lookup_timeout_sec'),
        },
        IfCondition(launch_handeye),
    )

    economic_grasp = _include(
        'economic_grasp_roi',
        'roi_economic_grasp.launch.py',
        {
            'params_file': LaunchConfiguration('economic_grasp_params_file'),
        },
        IfCondition(launch_control),
    )

    mcp_server = _include(
        'mcp',
        'mcp_server.launch.py',
        {
            'params_file': LaunchConfiguration('mcp_server_params_file'),
            'vision_image_topic': LaunchConfiguration('image_topic'),
            'api_detections_topic': LaunchConfiguration('api_detections_topic'),
            'target_command_topic': '/economic_grasp_roi/target_class_name',
            'vision_show_window': LaunchConfiguration('vision_show_window'),
            'vision_save_images': LaunchConfiguration('vision_save_images'),
            'vision_output_dir': LaunchConfiguration('vision_output_dir'),
            'omni_text_model': LaunchConfiguration('omni_text_model'),
        },
        IfCondition(launch_mcp_server),
    )

    omni_client = _include(
        'mcp',
        'mcp_omni_client.launch.py',
        {
            'params_file': LaunchConfiguration('mcp_omni_client_params_file'),
            'push_to_talk_enabled': LaunchConfiguration('push_to_talk_enabled'),
            'push_to_talk_key': LaunchConfiguration('push_to_talk_key'),
            'stop_record_key': LaunchConfiguration('stop_record_key'),
            'text_popup_enabled': LaunchConfiguration('text_popup_enabled'),
            'text_popup_key': LaunchConfiguration('text_popup_key'),
            'text_popup_auto_open': LaunchConfiguration('text_popup_auto_open'),
            'tts_engine': LaunchConfiguration('tts_engine'),
            'audio_device': LaunchConfiguration('audio_device'),
            'omni_realtime_voice': LaunchConfiguration('omni_realtime_voice'),
            'omni_speech_rate': LaunchConfiguration('omni_speech_rate'),
            'omni_speech_emotion': LaunchConfiguration('omni_speech_emotion'),
        },
        IfCondition(launch_omni_client),
    )

    return LaunchDescription(
        args
        + [
            LogInfo(msg='Starting LLM + API-vision grasp stack. Make sure DASHSCOPE_API_KEY is exported before using the Omni client.'),
            camera,
            aux_camera,
            _delayed(4.0, aux_camera_view),
            _delayed(1.0, moveit),
            _delayed(5.0, handeye),
            _delayed(7.0, economic_grasp),
            _delayed(8.0, mcp_server),
            _delayed(9.0, omni_client),
        ]
    )
