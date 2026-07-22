# plan_probe.py
"""Diagnostic: use plan() (which correctly defaults frame_id, unlike
compute_ik_async in this pymoveit2 version) to test planning to the arm's
own current, FK-verified pose."""
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2

from arm_application.robot import Robot


def main() -> None:
    rclpy.init()
    node = Node("plan_probe_node")
    cbg = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=Robot.JOINT_NAMES,
        base_link_name=Robot.BASE_LINK,
        end_effector_name=Robot.END_EFFECTOR,
        group_name=Robot.PLANNING_GROUP,
        use_move_group_action=False,
        callback_group=cbg,
    )
    moveit2.allowed_planning_time = Robot.PLANNING_TIME
    moveit2.num_planning_attempts = Robot.PLANNING_ATTEMPTS

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    time.sleep(1.0)
    current = moveit2.joint_state
    if current is None:
        node.get_logger().error("No live joint state received - cannot proceed.")
        return
    current_positions = list(current.position)

    fk_result = moveit2.compute_fk(joint_state=current_positions)
    if fk_result is None:
        node.get_logger().error("compute_fk failed.")
        return

    pos = fk_result.pose.position
    quat = fk_result.pose.orientation
    node.get_logger().info(
        f"Planning (via plan(), correct frame_id default) to the arm's own "
        f"current pose: ({pos.x}, {pos.y}, {pos.z}), quat=({quat.x}, {quat.y}, {quat.z}, {quat.w})"
    )

    trajectory = moveit2.plan(
        position=[pos.x, pos.y, pos.z],
        quat_xyzw=[quat.x, quat.y, quat.z, quat.w],
    )

    if trajectory is None:
        node.get_logger().error("plan() FAILED even on the arm's own current, self-consistent pose.")
    else:
        node.get_logger().info("plan() SUCCEEDED.")

    executor.shutdown()
    thread.join(timeout=2.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
