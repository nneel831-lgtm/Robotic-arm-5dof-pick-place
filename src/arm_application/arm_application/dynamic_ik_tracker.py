#!/usr/bin/env python3
"""
dynamic_ik_tracker.py
======================
ROS2 Humble visual-servoing node with a decoupled Pan-then-Reach state machine.
"""

import numpy as np
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Int32  # Added for gripper control

from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs  # noqa: F401 

from ikpy.chain import Chain


class DynamicIKTracker(Node):

    # ------------------------------------------------------------------
    # Static config
    # ------------------------------------------------------------------
    URDF_PATH = "/home/neel/your_ros2_ws/src/arm_description/urdf/Reference_Assembly44.urdf"
    BASE_FRAME = "base_link"
    CAMERA_OPTICAL_FRAME = "camera_color_frame"
    WRIST_FRAME = "Wrist_P"     
    BOTTLE_TF_CHILD = "bottle"  
    JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5"]
    TRAJ_TIME_FROM_START_S = 2.0  
    TRACKING_TIME_FROM_START_S = 1.0

    # --------------------------------------------------------------------
    # Visual-servo stabilization (J2-J5 Reach)
    # --------------------------------------------------------------------
    EMA_ALPHA = 0.25                    
    LATERAL_BIAS_CORRECTION_M = 0.01     
    POSITION_DEADBAND_XZ_M = 0.015      
    PID_KP = 0.13
    PID_KI = 0.005
    PID_KD = 0.1
    PID_INTEGRAL_CLAMP = 0.01
    PID_MIN_DT_S = 0.01
    PID_MAX_STEP_M = 0.03        

    IK_RESIDUAL_WARN_M = 0.03    
    IK_RESIDUAL_SKIP_M = 0.15    
    FREEZE_MAX_RESIDUAL_M = 0.02

    # --------------------------------------------------------------------
    # Anti-collision / Framing target
    # --------------------------------------------------------------------
    PRE_GRASP_STANDOFF_M = 0.1250           
    APPROACH_DISTANCE_THRESHOLD_M = 0.1250  
    
    # --------------------------------------------------------------------
    # ALIGNING Phase (Direct Visual Servoing for J1)
    # --------------------------------------------------------------------
    CAMERA_LATERAL_TARGET_M = 0.01       
    CAMERA_LATERAL_TOLERANCE_M = 0.01    
    ALIGN_STABILITY_TIME_S = 1.5         
    
    PAN_INVERT_DIR = True                 
    PAN_KP = 0.015
    PAN_KI = 0.001
    PAN_KD = 0.1                          

    # --------------------------------------------------------------------
    # Searching parameters
    # --------------------------------------------------------------------
    SEARCH_J1_LIMIT_RAD = 2.356           
    SEARCH_STEP_RAD = 0.35                
    BOTTLE_TIMEOUT_S = 1.0                

    # --------------------------------------------------------------------
    # Gripper close-on-approach
    # --------------------------------------------------------------------
    STATE_TICK_PERIOD_S = 0.1            
    GRASP_TOLERANCE_M = 0.01             
    APPROACH_TIMEOUT_S = 8.0             

    GRIPPER_CLOSE_POS = 490     
    GRIPPER_CLAMP_WAIT_S = 1.0  

    # --------------------------------------------------------------------
    # State machine states
    # --------------------------------------------------------------------
    STATE_SEARCHING = "SEARCHING"
    STATE_ALIGNING = "ALIGNING"
    STATE_TRACKING = "TRACKING"
    STATE_APPROACHING = "APPROACHING"
    STATE_GRASPING = "GRASPING"
    STATE_HOME = "HOME"
    
    HOME_JOINT_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0]  
    HOME_TIME_FROM_START_S = 4.0                     


    def __init__(self):
        super().__init__("dynamic_ik_tracker")

        self.chain = Chain.from_urdf_file(
            self.URDF_PATH,
            active_links_mask=[False, True, True, True, True, True, False],
        )
        
        self._last_ik_angles = [0.0] * len(self.chain.links)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.get_logger().info(f"Waiting for TF chain {self.CAMERA_OPTICAL_FRAME} -> base_link...")
        tf_wait_start = self.get_clock().now()
        while rclpy.ok() and not self.tf_buffer.can_transform(
            self.BASE_FRAME, self.CAMERA_OPTICAL_FRAME, rclpy.time.Time()
        ):
            rclpy.spin_once(self, timeout_sec=0.2)
            waited_s = (self.get_clock().now() - tf_wait_start).nanoseconds / 1e9
            if waited_s > 10.0:
                self.get_logger().warn(
                    f"Still waiting on {self.CAMERA_OPTICAL_FRAME} -> base_link after 10s. "
                )
                tf_wait_start = self.get_clock().now()
        self.get_logger().info("TF chain available.")

        self.pose_sub = self.create_subscription(
            PoseStamped, "/bottle_pose_camera", self.pose_callback, cam_qos
        )

        self._action_client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        
        # --- NEW: Publisher to trigger the Gripper in hiwonder.py ---
        self.gripper_pub = self.create_publisher(Int32, '/gripper_command', 10)
        
        self._goal_in_flight = False

        self._filtered_target = None
        self._commanded_target = None
        self._pid_integral = np.zeros(3)
        self._pid_prev_error = np.zeros(3)
        self._pid_last_time = None

        self._pan_integral = 0.0
        self._pan_prev_error = 0.0
        self._pan_last_time = None
        
        self._align_stable_start_time = None

        self.state = self.STATE_SEARCHING
        self._search_j1_angle = 0.0
        self._search_dir = 1.0
        self._last_bottle_time = None

        self._frozen_target = None  
        self._approach_start_time = None

        self._state_tick_timer = self.create_timer(
            self.STATE_TICK_PERIOD_S, self.state_machine_tick
        )

        self.get_logger().info("dynamic_ik_tracker up (state=SEARCHING). Looking for bottle...")

    def pose_callback(self, msg: PoseStamped):
        now = self.get_clock().now()
        self._last_bottle_time = now

        if self.state == self.STATE_SEARCHING:
            self.get_logger().info("Bottle detected! Breaking search and switching to ALIGNING.")
            self.state = self.STATE_ALIGNING

        if self.state not in [self.STATE_ALIGNING, self.STATE_TRACKING]:
            return

        corrected = PoseStamped()
        corrected.header = msg.header
        corrected.header.frame_id = self.CAMERA_OPTICAL_FRAME
        corrected.pose = msg.pose

        try:
            pose_base = self.tf_buffer.transform(
                corrected, self.BASE_FRAME, timeout=Duration(seconds=0.1)
            )
        except ExtrapolationException:
            try:
                corrected.header.stamp = rclpy.time.Time().to_msg()
                pose_base = self.tf_buffer.transform(
                    corrected, self.BASE_FRAME, timeout=Duration(seconds=0.1)
                )
            except (LookupException, ConnectivityException, ExtrapolationException):
                return
        except (LookupException, ConnectivityException):
            return

        bottle_tf = TransformStamped()
        bottle_tf.header.stamp = now.to_msg()
        bottle_tf.header.frame_id = self.BASE_FRAME
        bottle_tf.child_frame_id = self.BOTTLE_TF_CHILD
        bottle_tf.transform.translation.x = pose_base.pose.position.x
        bottle_tf.transform.translation.y = pose_base.pose.position.y
        bottle_tf.transform.translation.z = pose_base.pose.position.z
        bottle_tf.transform.rotation = pose_base.pose.orientation
        self.tf_broadcaster.sendTransform(bottle_tf)

        cam_x = msg.pose.position.x
        lateral_error = cam_x - self.CAMERA_LATERAL_TARGET_M

        dt_pan = self.PID_MIN_DT_S if self._pan_last_time is None else (now - self._pan_last_time).nanoseconds / 1e9
        dt_pan = max(dt_pan, self.PID_MIN_DT_S)
        self._pan_last_time = now

        self._pan_integral += lateral_error * dt_pan
        self._pan_integral = np.clip(self._pan_integral, -self.PID_INTEGRAL_CLAMP, self.PID_INTEGRAL_CLAMP)
        
        pan_derivative = (lateral_error - self._pan_prev_error) / dt_pan
        self._pan_prev_error = lateral_error

        pan_step = (self.PAN_KP * lateral_error) + (self.PAN_KI * self._pan_integral) + (self.PAN_KD * pan_derivative)
        
        if self.PAN_INVERT_DIR:
            pan_step = -pan_step
            
        target_j1 = self._last_ik_angles[1] + pan_step
        target_j1 = max(min(target_j1, self.SEARCH_J1_LIMIT_RAD), -self.SEARCH_J1_LIMIT_RAD)

        if self.state == self.STATE_ALIGNING:
            self._last_ik_angles[1] = target_j1
            self.send_trajectory([target_j1, 0.0, 0.0, 0.0, 0.0], self.TRACKING_TIME_FROM_START_S)
            
            if abs(lateral_error) > self.CAMERA_LATERAL_TOLERANCE_M:
                if self._align_stable_start_time is not None:
                    self._align_stable_start_time = None
                return  
            else:
                if self._align_stable_start_time is None:
                    self._align_stable_start_time = now
                    return
                
                elapsed_stable = (now - self._align_stable_start_time).nanoseconds / 1e9
                if elapsed_stable >= self.ALIGN_STABILITY_TIME_S:
                    self.get_logger().info(f"[ALIGNING] Stabilized for {elapsed_stable:.1f}s! Switching to TRACKING.")
                    self.state = self.STATE_TRACKING
                    self._align_stable_start_time = None
                else:
                    return 

        self.run_ik_and_actuate(pose_base, target_j1)

    def state_machine_tick(self):
        now = self.get_clock().now()

        if self.state == self.STATE_SEARCHING:
            if not self._goal_in_flight:
                self._search_j1_angle += self._search_dir * self.SEARCH_STEP_RAD
                
                if self._search_j1_angle >= self.SEARCH_J1_LIMIT_RAD:
                    self._search_j1_angle = self.SEARCH_J1_LIMIT_RAD
                    self._search_dir = -1.0
                elif self._search_j1_angle <= -self.SEARCH_J1_LIMIT_RAD:
                    self._search_j1_angle = -self.SEARCH_J1_LIMIT_RAD
                    self._search_dir = 1.0
                
                self.send_trajectory([self._search_j1_angle, 0.0, 0.0, 0.0, 0.0], self.TRACKING_TIME_FROM_START_S)
        
        elif self.state in [self.STATE_ALIGNING, self.STATE_TRACKING]:
            if self._last_bottle_time is not None:
                elapsed_s = (now - self._last_bottle_time).nanoseconds / 1e9
                if elapsed_s > self.BOTTLE_TIMEOUT_S:
                    self.get_logger().warn(f"Bottle lost for {elapsed_s:.1f}s. Reverting to SEARCHING.")
                    self.state = self.STATE_SEARCHING
                    
                    self._filtered_target = None
                    self._commanded_target = None
                    self._pid_integral[:] = 0.0
                    self._pan_integral = 0.0
                    self._pan_prev_error = 0.0
                    self._align_stable_start_time = None

        elif self.state == self.STATE_APPROACHING:
            if self._frozen_target is None:
                return

            wrist_pos = self._get_live_wrist_position()
            if wrist_pos is None:
                return

            diff = wrist_pos - self._frozen_target
            within_tolerance = (
                abs(diff[0]) <= self.GRASP_TOLERANCE_M
                and abs(diff[1]) <= self.GRASP_TOLERANCE_M
                and abs(diff[2]) <= self.GRASP_TOLERANCE_M
            )

            timed_out = False
            if self._approach_start_time is not None:
                elapsed_s = (now - self._approach_start_time).nanoseconds / 1e9
                timed_out = elapsed_s > self.APPROACH_TIMEOUT_S

            if within_tolerance or timed_out:
                if within_tolerance:
                    self.get_logger().info(f"Wrist_P within tolerance -- GRASPING.")
                else:
                    self.get_logger().warn(f"APPROACH_TIMEOUT_S reached -- forcing GRASPING.")
                    
                self.state = self.STATE_GRASPING  
                self._close_gripper_remote()
                time.sleep(self.GRIPPER_CLAMP_WAIT_S)  
                self._go_home()


    def _solve_ik_checked(self, target_position, target_j1):
        self._last_ik_angles[1] = target_j1
        
        ik_result = self.chain.inverse_kinematics(
            target_position, 
            initial_position=self._last_ik_angles
        )
        
        ik_result[1] = target_j1
        self._last_ik_angles = ik_result
        
        fk = self.chain.forward_kinematics(ik_result)
        achieved = fk[:3, 3]
        residual = float(np.linalg.norm(achieved - np.array(target_position)))

        if residual > self.IK_RESIDUAL_SKIP_M:
            return None, residual
        return list(ik_result[1:6]), residual


    def _get_live_wrist_position(self):
        try:
            wrist_tf = self.tf_buffer.lookup_transform(
                self.BASE_FRAME, self.WRIST_FRAME, rclpy.time.Time()
            )
            return np.array([
                wrist_tf.transform.translation.x,
                wrist_tf.transform.translation.y,
                wrist_tf.transform.translation.z,
            ])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None


    def _apply_pre_grasp_offset(self, bottle_centroid: np.ndarray) -> np.ndarray:
        wrist_pos = self._get_live_wrist_position()
        if wrist_pos is None:
            return bottle_centroid

        direction = bottle_centroid - wrist_pos
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return bottle_centroid
        unit_dir = direction / dist
        return bottle_centroid - self.PRE_GRASP_STANDOFF_M * unit_dir


    def run_ik_and_actuate(self, pose_base: PoseStamped, target_j1: float):
        bottle_centroid = np.array([
            pose_base.pose.position.x,
            pose_base.pose.position.y,
            pose_base.pose.position.z,
        ])
        bottle_centroid[1] += self.LATERAL_BIAS_CORRECTION_M
        raw_target = self._apply_pre_grasp_offset(bottle_centroid)

        if self._filtered_target is None:
            self._filtered_target = raw_target.copy()
        else:
            self._filtered_target = (
                self.EMA_ALPHA * raw_target + (1.0 - self.EMA_ALPHA) * self._filtered_target
            )

        if self._commanded_target is None:
            self._commanded_target = self._filtered_target.copy()

        error = self._filtered_target - self._commanded_target
        now = self.get_clock().now()
        dt = self.PID_MIN_DT_S if self._pid_last_time is None else (
            (now - self._pid_last_time).nanoseconds / 1e9
        )
        dt = max(dt, self.PID_MIN_DT_S)
        self._pid_last_time = now

        if abs(error[0]) < self.POSITION_DEADBAND_XZ_M and abs(error[2]) < self.POSITION_DEADBAND_XZ_M:
            self._pid_integral[:] = 0.0
            target_position = self._commanded_target.tolist()
        else:
            self._pid_integral += error * dt
            self._pid_integral = np.clip(
                self._pid_integral, -self.PID_INTEGRAL_CLAMP, self.PID_INTEGRAL_CLAMP
            )
            derivative = (error - self._pid_prev_error) / dt
            self._pid_prev_error = error

            pid_output = self.PID_KP * error + self.PID_KI * self._pid_integral + self.PID_KD * derivative
            pid_output = np.clip(pid_output, -self.PID_MAX_STEP_M, self.PID_MAX_STEP_M)
            self._commanded_target = self._commanded_target + pid_output
            target_position = self._commanded_target.tolist()

        ik_result_checked, _residual = self._solve_ik_checked(target_position, target_j1)
        if ik_result_checked is not None:
            self.send_trajectory(ik_result_checked, self.TRACKING_TIME_FROM_START_S)

        wrist_pos = self._get_live_wrist_position()
        if wrist_pos is not None:
            distance = float(np.linalg.norm(wrist_pos - np.array(target_position)))
            
            if distance < self.APPROACH_DISTANCE_THRESHOLD_M:
                ik_result_checked, residual = self._solve_ik_checked(target_position, target_j1)
                if ik_result_checked is None or residual > self.FREEZE_MAX_RESIDUAL_M:
                    pass
                else:
                    self._frozen_target = np.array(target_position)
                    self.state = self.STATE_APPROACHING
                    self._approach_start_time = self.get_clock().now()
                    self.send_trajectory(ik_result_checked, self.TRAJ_TIME_FROM_START_S)


    def _close_gripper_remote(self):
        """Publishes the close command to the /gripper_command topic."""
        self.get_logger().info(f"Sending Gripper CLOSE command (pos {self.GRIPPER_CLOSE_POS}) via ROS topic...")
        msg = Int32()
        msg.data = self.GRIPPER_CLOSE_POS
        self.gripper_pub.publish(msg)

    def _go_home(self):
        self.state = self.STATE_HOME
        self._goal_in_flight = False
        self.send_trajectory(self.HOME_JOINT_ANGLES, self.HOME_TIME_FROM_START_S)


    def send_trajectory(self, joint_angles, time_from_start_s: float):
        if self._goal_in_flight:
            return

        if not self._action_client.wait_for_server(timeout_sec=0.2):
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = [float(a) for a in joint_angles]
        point.velocities = [0.0] * len(self.JOINT_NAMES)
        point.time_from_start = Duration(seconds=time_from_start_s).to_msg()
        goal_msg.trajectory.points.append(point)
        goal_msg.goal_time_tolerance = Duration(seconds=0.5).to_msg()

        self._goal_in_flight = True
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._goal_in_flight = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)


    def result_callback(self, future):
        self._goal_in_flight = False


    def feedback_callback(self, feedback_msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = DynamicIKTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
