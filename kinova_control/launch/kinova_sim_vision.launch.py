import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # Get kortex_bringup package
    kortex_bringup_pkg = FindPackageShare('kortex_bringup').find('kortex_bringup')
    kinova_control_pkg = FindPackageShare('kinova_control').find('kinova_control')
    
    base_launch_file = os.path.join(kortex_bringup_pkg, 'launch', 'kortex_sim_control.launch.py')
    cube_urdf_file = os.path.join(kinova_control_pkg, 'urdf', 'scene_cube.urdf')

    if not os.path.exists(cube_urdf_file):
        print(f"\n[ERROR LAUNCH] No se encontro el archivo URDF en: {cube_urdf_file}\n")
    else:
        print(f"\n[OK LAUNCH] URDF encontrado correctamente en: {cube_urdf_file}\n")

    # Set simulation parameters
    sim_arguments = {
        'sim_gazebo': 'true',
        'sim_ignition': 'false',
        'robot_type': 'gen3',
        'dof': '6',
        'vision': 'true',
        'use_sim_time': 'true',
        'launch_rviz': 'true',
        'gripper': '', # 'robotiq_2f_85' 
        'robot_controller': 'joint_trajectory_controller'
    }

    # Launch from kinova bringup
    kinova_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch_file),
        launch_arguments=sim_arguments.items()
    )
    
    # Launch the scene
    spawn_cube = ExecuteProcess(
        cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', 
             '-file', cube_urdf_file, 
             '-entity', 'red_cube', 
             '-x', '0.5', '-y', '0.0', '-z', '0.0'],
        output='screen'
    )

    # IBVS Visual Servoing Node
    IBVS_node = Node(
        package='kinova_control',
        executable='IBVS_node',
        name='IBVS_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # PBVS Visual Servoing Node
    PBVS_node = Node(
        package='kinova_control',
        executable='PBVS_node',
        name='PBVS_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        LogInfo(msg="Initializing Kinova Gen3 6DOF + Vision Simulation in Gazebo Classic..."),
        kinova_simulation,
        spawn_cube,
        IBVS_node,
        # PBVS_node
    ])