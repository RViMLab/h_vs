from launch.launch_description import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    h_vs_params = PathJoinSubstitution([
        FindPackageShare(LaunchConfiguration("h_vs_pkg")),
        LaunchConfiguration("h_vs_params")
    ])

    # homography to end effector twist node
    h_vs = Node(
        package="h_vs",
        executable="h_vs_node",
        parameters=[
            h_vs_params, {
                "url": LaunchConfiguration("url"), 
                "cname": LaunchConfiguration("cname")
            }],
        remappings=[
            ("h_vs_node/G", "h_gen_node/G"),
            ("h_vs_node/K", "h_gen_node/K"),
            ("h_vs_node/twist", PathJoinSubstitution([
                LaunchConfiguration("controller"),
                "twist"
            ]))
        ]
    )

    # homography generation node
    h_gen_data_collect_node = Node(
        package="h_vs",
        executable="h_gen_data_collect_node.py",
        remappings=[
            ("h_gen_data_collect_node/image_raw", PathJoinSubstitution([
                LaunchConfiguration("cname"), "image_raw/crop"
            ])),
            ("h_gen_data_collect_node/camera_info", PathJoinSubstitution([
                LaunchConfiguration("cname"), "camera_info/crop"
            ])),
            ("h_gen_data_collect_node/wrench", PathJoinSubstitution([
                LaunchConfiguration("controller"),
                "wrench"
            ])),
            ("h_gen_data_collect_node/class_probabilities", PathJoinSubstitution([
                LaunchConfiguration("controller"),
                "class_probabilities"
            ]))
        ]
    )

    return [
        h_vs,
        h_gen_data_collect_node
    ]

def generate_launch_description():
    launch_args = []
    launch_args.append(DeclareLaunchArgument(
        name="h_vs_pkg",
        default_value="h_vs",
        description="Visual servo package with h_vs paramters YAML file."
    ))

    launch_args.append(DeclareLaunchArgument(
        name="h_vs_params",
        default_value="config/h_vs_params.yml",
        description="h_vs parameters YAML file, relative to h_vs_pkg."
    ))

    launch_args.append(DeclareLaunchArgument(
        name="cname",
        default_value="narrow_stereo",
        description="Camera name."
    ))

    launch_args.append(DeclareLaunchArgument(
        name="url",
        default_value="package://h_vs/config/cam_params.yml",
        description="Path to camera calibration file."
    ))

    launch_args.append(
        DeclareLaunchArgument(
            name="controller",
            default_value="collaborative_twist_position_controller",
            description="Robot controller, used for remapping h_vs_node/twist to controller/twist."
        )
    )

    return LaunchDescription(
        launch_args + [
            OpaqueFunction(function=launch_setup)
    ])
