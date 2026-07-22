#!/usr/bin/env python3

"""
Closed-Loop Visual Servoing PID Controller Bridge to IKPy
---------------------------------------------------------
Listens to:   /bottle_pose_camera (geometry_msgs/PoseStamped)
Transforms:  camera_color_optical_frame -> base_link
Outputs:     /ikpy_eef_target (geometry_msgs/PoseStamped)
---------------------------------------------------------
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import tf2_ros
import tf2_geometry_msgs  # Crucial for transforming PoseStamped objects natively


class PIDController:
    def __init__(self, kp, ki, kd, max_output, min_output, deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        self.deadband = deadband

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, current_time, error):
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0

        dt = (current_time - self.prev_time).nanoseconds / 1e9
        if dt <= 0.0:
            return 0.0

        # Apply deadband to suppress high-frequency signal noise/jitter
        if abs(error) < self.deadband:
            error = 0.0

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        output = max(self.min_output, min(self.max_output, output))

        self.prev_error = error
        self.prev_time = current_time
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class VisualServoPidNode(Node):
    def __init__(self):
        super().__init__('visual_servo_ik_bridge')

        # --- Frames Configuration ---
        self.base_frame = 'base_link'
        self.eef_frame = 'Wrist_R'
        self.camera_frame = 'camera_color_optical_frame'

        # --- PID Configuration ---
        # Camera X axis corresponds to Arm Y axis (Left/Right)
        self.pid_x = PIDController(kp=0.4, ki=0.01, kd=0.05, max_output=0.1, min_output=-0.1, deadband=0.01)
        # Camera Y axis corresponds to Arm Z axis (Up/Down)
        self.pid_y = PIDController(kp=0.4, ki=0.01, kd=0.05, max_output=0.1, min_output=-0.1, deadband=0.01)
        # Camera Z axis corresponds to Arm X axis (Depth/Approach)
        # Set a wider deadband (3.5cm) to handle the 3-4cm Z-depth noise explicitly
        self.pid_z = PIDController(kp=0.3, ki=0.00, kd=0.02, max_output=0.08, min_output=-0.08, deadband=0.035)

        # Target setpoints inside the Camera Frame (Centering the bottle)
        self.target_cam_x = 0.0  # Perfectly center X
        self.target_cam_y = 0.0  # Perfectly center Y
        self.target_cam_z = 0.40 # Target stop distance from camera lens (40 cm)

        # --- TF Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Pubs & Subs ---
        self.ik_target_pub = self.create_publisher(PoseStamped, '/ikpy_eef_target', 10)
        self.cam_pose_sub = self.create_subscription(PoseStamped, '/bottle_pose_camera', self.vision_callback, 10)
        self.tracking_status_sub = self.create_subscription(Bool, '/bottle_tracked', self.tracking_status_callback, 10)

        self.is_tracked = False
        self.get_logger().info("Visual Servo PID Controller Active. Bridging to IKPy...")

    def tracking_status_callback(self, msg):
        self.is_tracked = msg.data
        if not self.is_tracked:
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()

    def vision_callback(self, msg):
        if not self.is_tracked:
            return

        current_ros_time = self.get_clock().now()

        # --- 1. SAFE TF TREE LOOKUP WITH RETRY FAILSAFE ---
        try:
            # We add a small timeout (50ms) to allow the incoming TF buffer cache to catch up
            trans_base_to_cam = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05)
            )
            
            trans_base_to_eef = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.eef_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # Catch the startup lag cleanly instead of breaking execution loops
            self.get_logger().warn(f"TF Warm-up in progress: {str(e)}")
            return

        # --- 2. EXTRACT TRACKING ERRORS IN CAMERA FRAME ---
        err_x = msg.pose.position.x - self.target_cam_x
        err_y = msg.pose.position.y - self.target_cam_y
        err_z = msg.pose.position.z - self.target_cam_z

        # Compute velocity adjustments relative to the local camera coordinate frame
        vx_cam = self.pid_x.compute(current_ros_time, err_x)
        vy_cam = self.pid_y.compute(current_ros_time, err_y)
        vz_cam = self.pid_z.compute(current_ros_time, err_z)

        # --- 3. TRANSFORM LOCAL CAMERA ERRORS TO GLOBAL BASE_LINK FRAME ---
        # Package the local error velocities into a temporary PoseStamped message
        local_delta = PoseStamped()
        local_delta.header.frame_id = self.camera_frame
        local_delta.header.stamp = msg.header.stamp
        local_delta.pose.position.x = vx_cam
        local_delta.pose.position.y = vy_cam
        local_delta.pose.position.z = vz_cam
        local_delta.pose.orientation.w = 1.0

        try:
            # Shift the delta values cleanly into base_link orientation space
            base_delta = tf2_geometry_msgs.do_transform_pose(local_delta, trans_base_to_cam)
        except Exception as e:
            self.get_logger().error(f"Delta translation calculation failed: {str(e)}")
            return

        # --- 4. APPLY COMPUTED SHIFTS TO THE CURRENT WRIST COORDINATES ---
        ik_target = PoseStamped()
        ik_target.header.stamp = current_ros_time.to_msg()
        ik_target.header.frame_id = self.base_frame

        # Target global XYZ = Current EEF positional parameters + calculated servo delta step
        ik_target.pose.position.x = trans_base_to_eef.transform.translation.x + base_delta.pose.position.x
        ik_target.pose.position.y = trans_base_to_eef.transform.translation.y + base_delta.pose.position.y
        ik_target.pose.position.z = trans_base_to_eef.transform.translation.z + base_delta.pose.position.z

        # Maintain flat orientation parallel to base framework
        ik_target.pose.orientation.x = 0.0
        ik_target.pose.orientation.y = 0.0
        ik_target.pose.orientation.z = 0.0
        ik_target.pose.orientation.w = 1.0

        # --- 5. PUBLISH OUTPUT POSE STREAM FOR IKPY ---
        self.ik_target_pub.publish(ik_target)
        
        self.get_logger().info(
            f"Servo Active -> Target Base XYZ: [{ik_target.pose.position.x:.3f}, "
            f"{ik_target.pose.position.y:.3f}, {ik_target.pose.position.z:.3f}]",
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoPidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
