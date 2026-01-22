import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    delay_arg = DeclareLaunchArgument(
        "wbc_start_delay",
        default_value="0.0",
        description="Seconds to wait before starting WBC node.",
    )
    target_height_arg = DeclareLaunchArgument(
        "target_height",
        default_value="0.693",
        description="Target base height for WBC (meters).",
    )
    wait_controller_arg = DeclareLaunchArgument(
        "wait_for_controller",
        default_value="false",
        description="Wait for forward_command_controller to become active.",
    )
    controller_name_arg = DeclareLaunchArgument(
        "controller_name",
        default_value="forward_command_controller",
        description="Controller name to wait for before enabling WBC.",
    )
    debug_fall_log_arg = DeclareLaunchArgument(
        "debug_fall_log",
        default_value="false",
        description="Enable detailed instability logging in WBC.",
    )
    fall_log_rp_arg = DeclareLaunchArgument(
        "fall_log_roll_pitch_threshold",
        default_value="0.5",
        description="Roll/pitch threshold (rad) to trigger instability logs.",
    )
    fall_log_z_arg = DeclareLaunchArgument(
        "fall_log_z_threshold",
        default_value="0.4",
        description="Z threshold (m) to trigger instability logs.",
    )
    weight_contact_force_arg = DeclareLaunchArgument(
        "weight_contact_force",
        default_value="-1.0",
        description="Contact force task weight (-1 uses task_file if available).",
    )
    weight_contact_arg = DeclareLaunchArgument(
        "weight_contact",
        default_value="0.0",
        description="Contact kinematics task weight.",
    )
    base_angular_kp_arg = DeclareLaunchArgument(
        "base_angular_kp",
        default_value="-1.0",
        description="Base angular kp (-1 uses task_file if available).",
    )
    base_angular_kd_arg = DeclareLaunchArgument(
        "base_angular_kd",
        default_value="-1.0",
        description="Base angular kd (-1 uses task_file if available).",
    )
    com_target_x_arg = DeclareLaunchArgument(
        "com_target_x",
        default_value="0.10",
        description="COM target x offset (meters, positive is forward).",
    )
    weight_com_arg = DeclareLaunchArgument(
        "weight_com",
        default_value="80.0",
        description="COM task weight.",
    )
    reference_file_arg = DeclareLaunchArgument(
        "reference_file",
        default_value=os.path.join(
            get_package_share_directory("legged_hunter_controllers"),
            "config",
            "ocs2",
            "reference.info",
        ),
        description="Reference info file for defaultJointState.",
    )

    # 1. Start the simulation with forward_command_controller
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("legged_hunter_controllers"), "launch", "hunter_simulation.launch.py")
        ]),
        launch_arguments={"controller": "forward_command_controller"}.items()
    )

    # 2. Generate URDF
    hunter_pkg_path = get_package_share_directory("legged_hunter_description")
    controllers_pkg = get_package_share_directory("legged_hunter_controllers")
    xacro_file = os.path.join(hunter_pkg_path, "urdf", "hunter_ros2_control.xacro")
    doc = xacro.process_file(xacro_file)
    urdf_path = "/tmp/hunter_generated.urdf"
    with open(urdf_path, "w") as f:
        f.write(doc.toxml())

    # 3. WBC Node
    wbc_node = Node(
        package="legged_wbc",
        executable="wbc_node",
        parameters=[{
            "urdf_path": urdf_path,
            "control_frequency": 500.0,
            "target_height": LaunchConfiguration("target_height"),
            "wait_for_controller": LaunchConfiguration("wait_for_controller"),
            "controller_name": LaunchConfiguration("controller_name"),
            "debug_fall_log": LaunchConfiguration("debug_fall_log"),
            "fall_log_roll_pitch_threshold": LaunchConfiguration("fall_log_roll_pitch_threshold"),
            "fall_log_z_threshold": LaunchConfiguration("fall_log_z_threshold"),
            "task_file": os.path.join(controllers_pkg, "config", "ocs2", "task.info"),
            "reference_file": LaunchConfiguration("reference_file"),
            "weight_contact_force": LaunchConfiguration("weight_contact_force"),
            "weight_contact": LaunchConfiguration("weight_contact"),
            "base_angular_kp": LaunchConfiguration("base_angular_kp"),
            "base_angular_kd": LaunchConfiguration("base_angular_kd"),
            "com_target_x": LaunchConfiguration("com_target_x"),
            "weight_com": LaunchConfiguration("weight_com"),
        }],
        output="screen"
    )

    return LaunchDescription([
        delay_arg,
        target_height_arg,
        wait_controller_arg,
        controller_name_arg,
        debug_fall_log_arg,
        fall_log_rp_arg,
        fall_log_z_arg,
        weight_contact_force_arg,
        weight_contact_arg,
        base_angular_kp_arg,
        base_angular_kd_arg,
        com_target_x_arg,
        weight_com_arg,
        reference_file_arg,
        sim_launch,
        TimerAction(
            period=LaunchConfiguration("wbc_start_delay"),
            actions=[wbc_node],
        ),
    ])
