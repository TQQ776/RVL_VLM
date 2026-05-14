from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _pkg_file(package_name: str, *parts: str) -> str:
    return '/'.join((get_package_share_directory(package_name), *parts))


def generate_launch_description():
    launch_franka = LaunchConfiguration('launch_franka')
    launch_udp_bridge = LaunchConfiguration('launch_udp_bridge')
    launch_gripper_bridge = LaunchConfiguration('launch_gripper_bridge')
    launch_rviz = LaunchConfiguration('launch_rviz')

    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    controllers_yaml = LaunchConfiguration('controllers_yaml')
    teleop_params = LaunchConfiguration('teleop_params')
    axis_mapping_params = LaunchConfiguration('axis_mapping_params')

    franka = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])
        ]),
        launch_arguments={
            'robot_type': 'fr3',
            'arm_prefix': 'fr3_',
            'namespace': '',
            'robot_ip': robot_ip,
            'load_gripper': 'true',
            'use_fake_hardware': use_fake_hardware,
            'fake_sensor_commands': 'false',
            'joint_state_rate': '60',
            'controllers_yaml': controllers_yaml,
        }.items(),
        condition=IfCondition(launch_franka),
    )

    spawn_vr_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'vr_cartesian_velocity_controller',
            '--controller-manager-timeout',
            '30',
        ],
        output='screen',
    )

    quest_pose_bridge = Node(
        package='franka_vr_teleop',
        executable='quest3_pose_to_twist.py',
        name='quest3_pose_to_twist',
        output='screen',
        parameters=[teleop_params, axis_mapping_params],
    )

    udp_bridge = Node(
        package='franka_vr_teleop',
        executable='quest3_udp_bridge.py',
        name='quest3_udp_bridge',
        output='screen',
        parameters=[teleop_params],
        condition=IfCondition(launch_udp_bridge),
    )

    gripper_bridge = Node(
        package='franka_vr_teleop',
        executable='quest3_gripper_bridge.py',
        name='quest3_gripper_bridge',
        output='screen',
        parameters=[teleop_params],
        condition=IfCondition(launch_gripper_bridge),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '--display-config',
            PathJoinSubstitution([
                FindPackageShare('franka_description'),
                'rviz',
                'visualize_franka.rviz',
            ]),
        ],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.22.212'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        DeclareLaunchArgument('launch_franka', default_value='true'),
        DeclareLaunchArgument('launch_udp_bridge', default_value='true'),
        DeclareLaunchArgument('launch_gripper_bridge', default_value='true'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument(
            'controllers_yaml',
            default_value=_pkg_file('franka_vr_teleop', 'config', 'vr_teleop_controllers.yaml'),
        ),
        DeclareLaunchArgument(
            'teleop_params',
            default_value=_pkg_file('franka_vr_teleop', 'config', 'quest3_teleop.yaml'),
        ),
        DeclareLaunchArgument(
            'axis_mapping_params',
            default_value=_pkg_file('franka_vr_teleop', 'config', 'quest3_axis_mapping.yaml'),
        ),
        franka,
        TimerAction(period=4.0, actions=[spawn_vr_controller]),
        TimerAction(period=5.0, actions=[udp_bridge, quest_pose_bridge, gripper_bridge, rviz]),
    ])
