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

    h_vs = Node(
        package="h_vs",
        executable="h_vs_node",
        parameters=[
            h_vs_params, {
                "url": "file://" + LaunchConfiguration("url").perform(context), 
                "cname": LaunchConfiguration("cname")
            }]
    )

    return [h_vs]

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
        default_value=PathJoinSubstitution([
            FindPackageShare("h_vs"), "config/cam_params.yml"
        ]),
        description="Absolut path to camera calibration file."
    ))
    
    return LaunchDescription(
        launch_args + [
            OpaqueFunction(function=launch_setup)
    ])
