"""Launch the MCP direct-audio Qwen-Omni client node."""

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
        'omni_model',
        'omni_text_model',
        'omni_realtime_url',
        'omni_realtime_voice',
        'omni_speech_rate',
        'omni_speech_volume',
        'omni_speech_emotion',
        'omni_speech_style',
        'omni_native_audio_output_enabled',
        'omni_native_audio_fallback_to_local_tts',
        'list_tools_service',
        'call_tool_service',
    ):
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            optional_overrides[name] = value

    parameters = [
        params_file,
        {
            'push_to_talk_enabled': LaunchConfiguration('push_to_talk_enabled'),
            'push_to_talk_key': LaunchConfiguration('push_to_talk_key'),
            'stop_record_key': LaunchConfiguration('stop_record_key'),
            'text_popup_enabled': LaunchConfiguration('text_popup_enabled'),
            'text_popup_key': LaunchConfiguration('text_popup_key'),
            'text_popup_auto_open': LaunchConfiguration('text_popup_auto_open'),
            'tts_engine': LaunchConfiguration('tts_engine'),
            'audio_device': LaunchConfiguration('audio_device'),
        },
    ]
    if optional_overrides:
        parameters.append(optional_overrides)

    return [
        Node(
            package='mcp',
            executable='mcp_omni_client',
            name='mcp_omni_client',
            output='screen',
            parameters=parameters,
            emulate_tty=True,
        )
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('mcp')
    default_params = f'{pkg_share}/config/mcp_omni_client.yaml'

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('push_to_talk_enabled', default_value='false'),
        DeclareLaunchArgument('push_to_talk_key', default_value='r'),
        DeclareLaunchArgument('stop_record_key', default_value='q'),
        DeclareLaunchArgument('text_popup_enabled', default_value='true'),
        DeclareLaunchArgument('text_popup_key', default_value='t'),
        DeclareLaunchArgument('text_popup_auto_open', default_value='true'),
        DeclareLaunchArgument('tts_engine', default_value='auto'),
        DeclareLaunchArgument('audio_device', default_value='default'),
        DeclareLaunchArgument('omni_api_key_env', default_value=''),
        DeclareLaunchArgument('omni_base_url', default_value=''),
        DeclareLaunchArgument('omni_model', default_value=''),
        DeclareLaunchArgument('omni_text_model', default_value=''),
        DeclareLaunchArgument('omni_realtime_url', default_value=''),
        DeclareLaunchArgument('omni_realtime_voice', default_value=''),
        DeclareLaunchArgument('omni_speech_rate', default_value=''),
        DeclareLaunchArgument('omni_speech_volume', default_value=''),
        DeclareLaunchArgument('omni_speech_emotion', default_value=''),
        DeclareLaunchArgument('omni_speech_style', default_value=''),
        DeclareLaunchArgument('omni_native_audio_output_enabled', default_value=''),
        DeclareLaunchArgument('omni_native_audio_fallback_to_local_tts', default_value=''),
        DeclareLaunchArgument('list_tools_service', default_value=''),
        DeclareLaunchArgument('call_tool_service', default_value=''),
    ]

    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
