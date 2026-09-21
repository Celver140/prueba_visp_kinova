"""Kinova Gen3 (Gazebo Classic) + mock spacecraft + model-based IBVS node.

ros2 launch spacecraft_servoing spacecraft_servoing.launch.py
ros2 launch spacecraft_servoing spacecraft_servoing.launch.py initial_target:=screw_1
ros2 launch spacecraft_servoing spacecraft_servoing.launch.py start_servoing:=false   # sim only
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('spacecraft_servoing')
    kortex = get_package_share_directory('kortex_bringup')
    models_dir = os.path.join(pkg, 'models')
    default_model = os.path.join(models_dir, 'mock_spacecraft', 'model.sdf')

    LC = LaunchConfiguration
    pose_args = {'target_x': '0.55', 'target_y': '0.0', 'target_z': '0.0',
                 'target_roll': '0.0', 'target_pitch': '0.0', 'target_yaw': '0.0'}
    args = [DeclareLaunchArgument(k, default_value=v, description='spacecraft pose in base_link')
            for k, v in pose_args.items()]
    args += [
        DeclareLaunchArgument('spacecraft_sdf', default_value=default_model),
        DeclareLaunchArgument('targets_file', default_value=os.path.join(pkg, 'config', 'targets.yaml')),
        DeclareLaunchArgument('params_file', default_value=os.path.join(pkg, 'config', 'servoing.yaml')),
        DeclareLaunchArgument('initial_target', default_value=''),
        DeclareLaunchArgument('start_servoing', default_value='true'),
        DeclareLaunchArgument('launch_rviz', default_value='false'),
        DeclareLaunchArgument('zero_gravity', default_value='false',
                              description='gz physics -g 0,0,0 after start-up (Gazebo Classic CLI)'),
    ]

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(kortex, 'launch', 'kortex_sim_control.launch.py')),
        launch_arguments={
            'sim_gazebo': 'true', 'sim_ignition': 'false', 'robot_type': 'gen3', 'dof': '6',
            'vision': 'true', 'use_sim_time': 'true', 'launch_rviz': LC('launch_rviz'),
            'gripper': '', 'robot_controller': 'joint_trajectory_controller',
        }.items())

    spawn = ExecuteProcess(
        cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-file', LC('spacecraft_sdf'),
             '-entity', 'spacecraft',
             '-x', LC('target_x'), '-y', LC('target_y'), '-z', LC('target_z'),
             '-R', LC('target_roll'), '-P', LC('target_pitch'), '-Y', LC('target_yaw')],
        output='screen')

    zero_g = TimerAction(period=8.0, actions=[ExecuteProcess(
        cmd=['gz', 'physics', '-g', '0,0,0'], output='screen', condition=IfCondition(LC('zero_gravity')))])

    node = Node(
        package='spacecraft_servoing', executable='servoing_node', name='spacecraft_servoing',
        output='screen', condition=IfCondition(LC('start_servoing')),
        parameters=[LC('params_file'), {
            'use_sim_time': True,
            'targets_file': LC('targets_file'),
            'initial_target': LC('initial_target'),
            **{k: ParameterValue(LC(k), value_type=float) for k in pose_args},
        }])

    return LaunchDescription(args + [
        AppendEnvironmentVariable('GAZEBO_MODEL_PATH', models_dir),
        sim, spawn, zero_g, node,
    ])
