from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='yolov8n.pt',
        description='Path or name of the YOLOv8 weights file (ultralytics).'
    )
    confidence_arg = DeclareLaunchArgument(
        'confidence_threshold', default_value='0.45',
        description='Minimum detection confidence for the bottle class.'
    )
    device_arg = DeclareLaunchArgument(
        'device', default_value='0',
        description="Inference device: '0' for first CUDA GPU, 'cpu' for CPU."
    )
    tracker_cfg_arg = DeclareLaunchArgument(
        'tracker_config', default_value='bytetrack.yaml',
        description='Ultralytics tracker config (bytetrack.yaml or botsort.yaml).'
    )
    publish_debug_arg = DeclareLaunchArgument(
        'publish_debug_image', default_value='true',
        description='Whether to publish the annotated debug image.'
    )

    bottle_pose_node = Node(
        package='bottle_pose_tracker',
        executable='bottle_pose_node',
        name='bottle_pose_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'device': LaunchConfiguration('device'),
            'tracker_config': LaunchConfiguration('tracker_config'),
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
            'image_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
        }],
    )

    return LaunchDescription([
        model_path_arg,
        confidence_arg,
        device_arg,
        tracker_cfg_arg,
        publish_debug_arg,
        bottle_pose_node,
    ])
