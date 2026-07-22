#!/usr/bin/env python3

"""
Production-Grade Pure Transform Node (Kinematically Invariant)
--------------------------------------------------
Subscribes: /bottle_pose_camera (PoseStamped)
Publishes:  /bottle_target_base (PoseStamped)
--------------------------------------------------c 
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

def quaternion_to_rotation_matrix(x, y, z, w):
    """Converts a quaternion into a 3x3 rotation matrix for pure dot-product operations."""
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

class PoseTransformer(Node):
    def __init__(self):
        super().__init__('pure_pose_transformer')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.publisher = self.create_publisher(PoseStamped, '/bottle_target_base', 10)
        self.subscription = self.create_subscription(
            PoseStamped, '/bottle_pose_camera', self.pose_callback, 10
        )
        
        # Absolute frame filtering state
        self.filtered_base_pose = None
        self.ema_alpha = 0.15  # 15% new data, 85% old data for aggressive Z-noise smoothing
        
        self.get_logger().info("Kinematically Invariant Transform Node Online.")

    def pose_callback(self, msg):
        try:
            # FIX 1: EXACT TIME SYNCHRONIZATION
            # By passing msg.header.stamp, we access the TF buffer's history.
            # This fetches the exact joint angles the robot had at the millisecond the camera took the picture.
            trans = self.tf_buffer.lookup_transform(
                'base_link', 
                msg.header.frame_id, 
                msg.header.stamp, 
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Normal during startup or if a frame drops
            return

        # 1. Axis Mapping: YOLO (OpenCV) -> MoveIt Camera Frame
        # YOLO  : +X Right, -Y Up (+Y Down), +Z Forward
        # MoveIt: +X Left,  +Y Up,           +Z Forward
        P_camera = np.array([
            -msg.pose.position.x,  # Invert X (Right to Left)
            -msg.pose.position.y,  # Invert Y (Down to Up)
            msg.pose.position.z    # Z remains Forward
        ])

        # 2. Extract Translation from Historical TF
        t = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z
        ])

        # 3. Extract Rotation Matrix from Historical TF
        R = quaternion_to_rotation_matrix(
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        )

        # 4. Pure Matrix Math: P_base = (R * P_camera) + T
        P_base_raw = R.dot(P_camera) + t

        # FIX 2: STATIC FRAME FILTERING
        # Apply the EMA filter strictly in the absolute base_link frame where the bottle is physically static.
        if self.filtered_base_pose is None:
            self.filtered_base_pose = P_base_raw
        else:
            # Prevent the filter from dragging if the bottle is genuinely moved by a human (> 20cm change)
            if np.linalg.norm(P_base_raw - self.filtered_base_pose) > 0.2:
                self.filtered_base_pose = P_base_raw
            else:
                self.filtered_base_pose = (self.ema_alpha * P_base_raw) + ((1.0 - self.ema_alpha) * self.filtered_base_pose)

        # 5. Publish clean PoseStamped
        target_msg = PoseStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = 'base_link'
        
        target_msg.pose.position.x = float(self.filtered_base_pose[0])
        target_msg.pose.position.y = float(self.filtered_base_pose[1])
        target_msg.pose.position.z = float(self.filtered_base_pose[2])
        
        # Identity orientation for pick-and-place
        target_msg.pose.orientation.w = 1.0
        
        self.publisher.publish(target_msg)
        
        self.get_logger().info(
            f"Static Base XYZ -> X: {target_msg.pose.position.x:.3f}, "
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
