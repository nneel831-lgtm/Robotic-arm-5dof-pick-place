#!/usr/bin/env python3

"""
ROS2 Standard Transformer Tool
--------------------------------------------------
Subscribes: /bottle_pose_camera (PoseStamped)
Publishes:  /bottle_target_base (PoseStamped)
--------------------------------------------------
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # Standard ROS2 transformation tool


class EMAFilter:
    """Filters noisy Z-depth cleanly in the absolute static frame."""
    def __init__(self, alpha=0.15, deadband=0.02):
        self.alpha = alpha
        self.deadband = deadband
        self.current = None

    def filter(self, new_val):
        if self.current is None:
            self.current = new_val
            return self.current
            
        if abs(new_val - self.current) > 0.2:  # Huge jump (e.g. human moved the bottle)
            self.current = new_val
        elif abs(new_val - self.current) > self.deadband: # Smooth out minor noise
            self.current = (self.alpha * new_val) + ((1.0 - self.alpha) * self.current)
            
        return self.current

    def reset(self):
        self.current = None


class PoseTransformer(Node):
    def __init__(self):
        super().__init__('ros2_tool_transformer')
        
        self.base_frame = 'base_link'
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.filter_x = EMAFilter()
        self.filter_y = EMAFilter()
        self.filter_z = EMAFilter()
        
        self.publisher = self.create_publisher(PoseStamped, '/bottle_target_base', 10)
        self.subscription = self.create_subscription(PoseStamped, '/bottle_pose_camera', self.pose_callback, 10)
        self.track_sub = self.create_subscription(Bool, '/bottle_tracked', self.track_cb, 10)
        
        self.get_logger().info("ROS2 Standard Transformer Online.")

    def track_cb(self, msg):
        if not msg.data:
            self.filter_x.reset()
            self.filter_y.reset()
            self.filter_z.reset()

    def pose_callback(self, msg):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, 
                msg.header.frame_id, 
                msg.header.stamp, 
                timeout=Duration(seconds=0.05)
            )
        except Exception as e:
            # Ignore the first few dropped frames while TF cache fills up
            return

        # Note: We pass msg.pose (which is a Pose), NOT msg (which is a PoseStamped). 
        # This completely fixes the "has no attribute 'position'" error.
        try:
            transformed_pose = tf2_geometry_msgs.do_transform_pose(msg.pose, trans)
        except Exception as e:
            self.get_logger().error(f"ROS2 Transform Tool Error: {e}")
            return

        target_msg = PoseStamped()
        target_msg.header.stamp = msg.header.stamp
        target_msg.header.frame_id = self.base_frame
        
        target_msg.pose.position.x = self.filter_x.filter(transformed_pose.position.x)
        target_msg.pose.position.y = self.filter_y.filter(transformed_pose.position.y)
        target_msg.pose.position.z = self.filter_z.filter(transformed_pose.position.z)
        
        # Keep flat grasping orientation
        target_msg.pose.orientation.w = 1.0
        
        self.publisher.publish(target_msg)
        
        self.get_logger().info(
            f"Base Target -> X: {target_msg.pose.position.x:.3f}, "
            f"Y: {target_msg.pose.position.y:.3f}, Z: {target_msg.pose.position.z:.3f}",
            throttle_duration_sec=0.5
        )

def main(args=None):
    rclpy.init(args=args)
    node = PoseTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
