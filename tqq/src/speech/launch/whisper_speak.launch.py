"""Launch a node that records, transcribes, and speaks back the text."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('speech')
    default_params = f'{pkg_share}/config/whisper_speak.yaml'

    params_file = LaunchConfiguration('params_file')

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('record_seconds', default_value='5.0'),
        DeclareLaunchArgument('asr_engine', default_value='whisper'),
        DeclareLaunchArgument('whisper_model', default_value='small'),
        DeclareLaunchArgument('whisper_model_dir', default_value='/home/tqq/TQQ_ws/tqq/src/speech/model'),
        DeclareLaunchArgument(
            'funasr_model',
            default_value='/home/tqq/TQQ_ws/tqq/src/speech/model/funasr/paraformer-zh',
        ),
        DeclareLaunchArgument('funasr_vad_model', default_value=''),
        DeclareLaunchArgument('funasr_punc_model', default_value=''),
        DeclareLaunchArgument('funasr_device', default_value=''),
        DeclareLaunchArgument('funasr_hub', default_value='ms'),
        DeclareLaunchArgument('funasr_hotword', default_value=''),
        DeclareLaunchArgument('tts_engine', default_value='auto'),
        DeclareLaunchArgument('tts_language', default_value='zh-cn'),
        DeclareLaunchArgument('tts_edge_voice', default_value='zh-CN-XiaoxiaoNeural'),
        DeclareLaunchArgument('tts_edge_rate', default_value='+0%'),
        DeclareLaunchArgument('tts_edge_pitch', default_value='+8Hz'),
        DeclareLaunchArgument('tts_edge_volume', default_value='+0%'),
        DeclareLaunchArgument('push_to_talk_enabled', default_value='true'),
        DeclareLaunchArgument('push_to_talk_key', default_value='r'),
        DeclareLaunchArgument('stop_record_key', default_value='q'),
        DeclareLaunchArgument('min_record_seconds', default_value='0.3'),
        DeclareLaunchArgument('audio_device', default_value='default'),
        DeclareLaunchArgument('auto_run', default_value='false'),
    ]

    node = Node(
        package='speech',
        executable='whisper_speak',
        name='whisper_speak',
        output='screen',
        parameters=[
            params_file,
            {
                'record_seconds': LaunchConfiguration('record_seconds'),
                'asr_engine': LaunchConfiguration('asr_engine'),
                'whisper_model': LaunchConfiguration('whisper_model'),
                'whisper_model_dir': LaunchConfiguration('whisper_model_dir'),
                'funasr_model': LaunchConfiguration('funasr_model'),
                'funasr_vad_model': LaunchConfiguration('funasr_vad_model'),
                'funasr_punc_model': LaunchConfiguration('funasr_punc_model'),
                'funasr_device': LaunchConfiguration('funasr_device'),
                'funasr_hub': LaunchConfiguration('funasr_hub'),
                'funasr_hotword': LaunchConfiguration('funasr_hotword'),
                'tts_engine': LaunchConfiguration('tts_engine'),
                'tts_language': LaunchConfiguration('tts_language'),
                'tts_edge_voice': LaunchConfiguration('tts_edge_voice'),
                'tts_edge_rate': LaunchConfiguration('tts_edge_rate'),
                'tts_edge_pitch': LaunchConfiguration('tts_edge_pitch'),
                'tts_edge_volume': LaunchConfiguration('tts_edge_volume'),
                'push_to_talk_enabled': LaunchConfiguration('push_to_talk_enabled'),
                'push_to_talk_key': LaunchConfiguration('push_to_talk_key'),
                'stop_record_key': LaunchConfiguration('stop_record_key'),
                'min_record_seconds': LaunchConfiguration('min_record_seconds'),
                'audio_device': LaunchConfiguration('audio_device'),
                'auto_run': LaunchConfiguration('auto_run'),
            },
        ],
        emulate_tty=True,
    )

    return LaunchDescription(args + [node])
