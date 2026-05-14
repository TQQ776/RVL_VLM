"""Launch image-detection/depth-to-MoveIt target control for FR3."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import yaml


def read_yaml_param(params_file_path, name):
    try:
        with open(params_file_path, 'r') as param_file:
            data = yaml.safe_load(param_file) or {}
    except OSError:
        return None

    for key in ('/**', '/object_target_controller', 'object_target_controller'):
        params = data.get(key, {}).get('ros__parameters', {})
        if name in params:
            return params[name]
    return None


def launch_setup(context, *args, **kwargs):
    params_file_path = LaunchConfiguration('params_file').perform(context)
    overrides = {
        'depth_topic': LaunchConfiguration('depth_topic'),
        'camera_info_topic': LaunchConfiguration('camera_info_topic'),
        'detections_topic': LaunchConfiguration('detections_topic'),
        'target_class_name': LaunchConfiguration('target_class_name'),
        'execution_mode': LaunchConfiguration('execution_mode'),
        'plan_only': ParameterValue(LaunchConfiguration('plan_only'), value_type=bool),
        'auto_execute': ParameterValue(LaunchConfiguration('auto_execute'), value_type=bool),
    }

    for name in ('max_velocity_scaling', 'max_acceleration_scaling'):
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            overrides[name] = float(value)
            continue

        yaml_value = read_yaml_param(params_file_path, name)
        if yaml_value is not None:
            overrides[name] = float(yaml_value)

    return [
        LogInfo(msg=f'control params_file={params_file_path}'),
        LogInfo(
            msg=(
                'control velocity_scaling='
                f'{overrides.get("max_velocity_scaling", "node_default")}, '
                'acceleration_scaling='
                f'{overrides.get("max_acceleration_scaling", "node_default")}'
            )
        ),
        Node(
            package='control',
            executable='object_target_controller',
            name='object_target_controller',
            output='screen',
            parameters=[
                params_file_path,
                overrides,
            ],
            emulate_tty=True,
        )
    ]

def generate_launch_description():
    pkg_share = get_package_share_directory('control')
    default_params = f'{pkg_share}/config/object_target_control.yaml'
    default_realsense_params = f'{pkg_share}/config/realsense_aligned_depth.yaml'

    params_file = LaunchConfiguration('params_file')
    realsense_params_file = LaunchConfiguration('realsense_params_file')
    launch_camera = LaunchConfiguration('launch_camera')
    launch_moveit = LaunchConfiguration('launch_moveit')
    camera_namespace = LaunchConfiguration('camera_namespace')
    camera_name = LaunchConfiguration('camera_name')
    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('realsense_params_file', default_value=default_realsense_params),
        DeclareLaunchArgument('launch_camera', default_value='false'),
        DeclareLaunchArgument('launch_moveit', default_value='false'),
        DeclareLaunchArgument('camera_namespace', default_value='camera'),
        DeclareLaunchArgument('camera_name', default_value='camera'),
        DeclareLaunchArgument('robot_ip', default_value='192.168.22.212'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        DeclareLaunchArgument('image_topic', default_value='/camera/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/camera/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera/color/camera_info'),
        DeclareLaunchArgument('detections_topic', default_value='/mcp_omni_client/api_detections_json'),
        DeclareLaunchArgument('target_class_name', default_value=''),
        DeclareLaunchArgument('execution_mode', default_value='move_group'),
        DeclareLaunchArgument('plan_only', default_value='false'),
        DeclareLaunchArgument(
            'max_velocity_scaling',
            default_value='',
            description='Optional override. Empty means use params_file value.',
        ),
        DeclareLaunchArgument(
            'max_acceleration_scaling',
            default_value='',
            description='Optional override. Empty means use params_file value.',
        ),
        DeclareLaunchArgument('auto_execute', default_value='false'),
    ]

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_camera'), 'launch', 'realsense.launch.py'])
        ]),
        launch_arguments={
            'camera_namespace': camera_namespace,
            'camera_name': camera_name,
            'params_file': realsense_params_file,
        }.items(),
        condition=IfCondition(launch_camera),
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_fr3_moveit_config'), 'launch', 'moveit.launch.py'])
        ]),
        launch_arguments={
            'robot_ip': robot_ip,
            'use_fake_hardware': use_fake_hardware,
        }.items(),
        condition=IfCondition(launch_moveit),
    )

    return LaunchDescription(args + [
        camera_launch,
        moveit_launch,
        OpaqueFunction(function=launch_setup),
    ])
