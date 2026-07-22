#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf2_geometry_msgs  # Required for do_transform_pose

class PoseTransformerNode(Node):
    def __init__(self):
        super().__init__('bottle_pose_transformer')

        # --- Parameters ---
        self.declare_parameter('input_topic', '/bottle/pose')
        self.declare_parameter('output_topic', '/bottle/world_pose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('bottle_frame', 'bottle')
        
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.bottle_frame = self.get_parameter('bottle_frame').value

        # --- TF2 Components ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Publishers & Subscribers ---
        self.pose_pub = self.create_publisher(PoseStamped, self.output_topic, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            self.input_topic, 
            self.pose_callback, 
            10
        )

        self.get_logger().info(f"Pose Transformer Node Started.")
        self.get_logger().info(f"Listening on: {self.input_topic}")
        self.get_logger().info(f"Target Frame: {self.map_frame} | Child TF: {self.bottle_frame}")

    def pose_callback(self, msg: PoseStamped):
        # Extract the incoming frame dynamically
        source_frame = msg.header.frame_id
        
        if not source_frame:
            self.get_logger().warn("Incoming PoseStamped has no frame_id. Ignoring.")
            return

        try:
            # Robust TF lookup: Try to get the transform at the EXACT timestamp of the image
            msg_time = Time.from_msg(msg.header.stamp)
            
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                source_frame, 
                msg_time, 
                timeout=Duration(seconds=0.1) # Timeout to wait for TF to propagate
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed from {source_frame} to {self.map_frame}: {e}")
            return

        try:
            # Transform the pose into the map frame
            world_pose = tf2_geometry_msgs.do_transform_pose(msg.pose, trans)
            
            # Reconstruct the PoseStamped for the map frame
            world_pose_stamped = PoseStamped()
            world_pose_stamped.header.stamp = msg.header.stamp
            world_pose_stamped.header.frame_id = self.map_frame
            world_pose_stamped.pose = world_pose
            
            # Publish the transformed PoseStamped
            self.pose_pub.publish(world_pose_stamped)

            # --- Broadcast the Dynamic TF ---
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = self.map_frame
            t.child_frame_id = self.bottle_frame

            t.transform.translation.x = world_pose.position.x
            t.transform.translation.y = world_pose.position.y
            t.transform.translation.z = world_pose.position.z

            t.transform.rotation = world_pose.orientation

            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().error(f"Failed to transform or broadcast pose: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseTransformerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
