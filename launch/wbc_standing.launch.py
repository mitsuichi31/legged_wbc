import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # 1. Start the simulation with forward_command_controller
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("legged_hunter_controllers"), "launch", "hunter_simulation.launch.py")
        ]),
        launch_arguments={"controller": "forward_command_controller"}.items()
    )

    # 2. Generate URDF
    hunter_pkg_path = get_package_share_directory("legged_hunter_description")
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
            "control_frequency": 500.0
        }],
        output="screen"
    )

    return LaunchDescription([
        sim_launch,
        wbc_node
    ])
