import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, Shutdown, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import yaml


def _pkg_file(package_name: str, *parts: str) -> str:
    return '/'.join((get_package_share_directory(package_name), *parts))


def _load_yaml(package_name: str, *parts: str):
    path = os.path.join(get_package_share_directory(package_name), *parts)
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def generate_launch_description():
    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    launch_udp_bridge = LaunchConfiguration('launch_udp_bridge')
    launch_gripper_bridge = LaunchConfiguration('launch_gripper_bridge')
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_move_group = LaunchConfiguration('launch_move_group')

    controllers_yaml = LaunchConfiguration('controllers_yaml')
    teleop_params = LaunchConfiguration('teleop_params')
    servo_teleop_params = LaunchConfiguration('servo_teleop_params')
    axis_mapping_params = LaunchConfiguration('axis_mapping_params')
    servo_params_file = LaunchConfiguration('servo_params')

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        'fr3',
        'fr3.urdf.xacro',
    )
    robot_description_config = Command([
        FindExecutable(name='xacro'),
        ' ',
        franka_xacro_file,
        ' hand:=true',
        ' robot_ip:=',
        robot_ip,
        ' ee_id:=franka_hand',
        ' use_fake_hardware:=',
        use_fake_hardware,
        ' fake_sensor_commands:=false',
        ' ros2_control:=true',
    ])
    robot_description = {
        'robot_description': ParameterValue(robot_description_config, value_type=str)
    }

    srdf_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        'fr3',
        'fr3.srdf.xacro',
    )
    robot_description_semantic_config = Command([
        FindExecutable(name='xacro'),
        ' ',
        srdf_xacro_file,
        ' hand:=true',
        ' ee_id:=franka_hand',
    ])
    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(robot_description_semantic_config, value_type=str)
    }
    robot_description_kinematics = _load_yaml('franka_fr3_moveit_config', 'config', 'kinematics.yaml')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='screen',
    )

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[controllers_yaml, robot_description, {'load_gripper': True}],
        remappings=[('joint_states', 'franka/joint_states')],
        output='screen',
        on_exit=Shutdown(),
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'source_list': ['franka/joint_states', 'franka_gripper/joint_states'],
            'rate': 60,
            'use_robot_description': False,
        }],
        output='screen',
    )

    spawn_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60'],
        output='screen',
    )

    spawn_franka_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['franka_robot_state_broadcaster', '--controller-manager-timeout', '60'],
        parameters=[{'robot_type': 'fr3'}],
        condition=UnlessCondition(use_fake_hardware),
        output='screen',
    )

    spawn_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['fr3_arm_controller', '--controller-manager-timeout', '60'],
        output='screen',
    )

    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])
        ]),
        launch_arguments={
            'robot_ip': robot_ip,
            'use_fake_hardware': use_fake_hardware,
            'namespace': '',
        }.items(),
    )

    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        parameters=[
            servo_params_file,
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
        ],
        output='screen',
    )

    start_servo = Node(
        package='franka_vr_teleop',
        executable='start_moveit_servo.py',
        name='start_moveit_servo',
        output='screen',
    )

    quest_pose_bridge = Node(
        package='franka_vr_teleop',
        executable='quest3_pose_to_twist.py',
        name='quest3_pose_to_twist',
        output='screen',
        parameters=[teleop_params, servo_teleop_params, axis_mapping_params],
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

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_fr3_moveit_config'), 'launch', 'move_group.launch.py'])
        ]),
        launch_arguments={
            'robot_ip': robot_ip,
            'load_gripper': 'true',
            'use_fake_hardware': use_fake_hardware,
            'fake_sensor_commands': 'false',
            'namespace': '',
        }.items(),
        condition=IfCondition(launch_move_group),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '--display-config',
            PathJoinSubstitution([FindPackageShare('franka_fr3_moveit_config'), 'rviz', 'moveit.rviz']),
        ],
        parameters=[robot_description, robot_description_semantic, robot_description_kinematics],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.22.212'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        DeclareLaunchArgument('launch_udp_bridge', default_value='true'),
        DeclareLaunchArgument('launch_gripper_bridge', default_value='true'),
        DeclareLaunchArgument('launch_move_group', default_value='false'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument(
            'controllers_yaml',
            default_value=_pkg_file('franka_vr_teleop', 'config', 'servo_fr3_ros_controllers.yaml'),
        ),
        DeclareLaunchArgument(
            'teleop_params',
            default_value=_pkg_file('franka_vr_teleop', 'config', 'quest3_teleop.yaml'),
        ),
        DeclareLaunchArgument(
            'servo_teleop_params',
            default_value=_pkg_file('franka_vr_teleop', 'config', 'quest3_servo_teleop.yaml'),
        ),
        DeclareLaunchArgument(
            'axis_mapping_params',
            default_value=_pkg_file('franka_vr_teleop', 'config', 'quest3_axis_mapping.yaml'),
        ),
        DeclareLaunchArgument(
            'servo_params',
            default_value=_pkg_file('franka_vr_teleop', 'config', 'fr3_moveit_servo.yaml'),
        ),
        robot_state_publisher,
        ros2_control_node,
        joint_state_publisher,
        spawn_joint_state_broadcaster,
        spawn_franka_state_broadcaster,
        spawn_arm_controller,
        gripper_launch,
        TimerAction(period=2.0, actions=[servo_node]),
        TimerAction(period=4.0, actions=[start_servo]),
        TimerAction(period=5.0, actions=[udp_bridge, quest_pose_bridge, gripper_bridge, move_group, rviz]),
    ])
