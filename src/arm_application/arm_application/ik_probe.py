# ik_probe.py
"""Diagnostic: seed IK from the arm's actual current joint state (not Home),
to determine whether tf2_echo's pose was captured from this configuration."""
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2

from arm_application.robot import Robot

X, Y, Z = 0.166, 0.000, 0.478
QX, QY, QZ, QW = 0.513, 0.465, 0.563, 0.451


def main() -> None:
    rclpy.init()
    node = Node("ik_probe_node")
    cbg = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=Robot.JOINT_NAMES,
        base_link_name=Robot.BASE_LINK,
        end_effector_name=Robot.END_EFFECTOR,
        group_name=Robot.PLANNING_GROUP,
        callback_group=cbg,
    )

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
    node.get_logger().info(f"Seeding IK from CURRENT live state: {current_positions}")

    # Sanity check first: what pose does FK say the arm is at RIGHT NOW?
    fk_result = moveit2.compute_fk(joint_state=current_positions)
    if fk_result is not None:
        node.get_logger().info(f"FK of current state -> {fk_result}")
    else:
        node.get_logger().warning("compute_fk failed/unavailable.")

    result = moveit2.compute_ik(
        position=[X, Y, Z],
        quat_xyzw=[QX, QY, QZ, QW],
        start_joint_state=current_positions,
    )

    if result is None:
        node.get_logger().error("compute_ik FAILED even seeded from the current live state.")
    else:
        node.get_logger().info(f"compute_ik SUCCEEDED: {result}")

    executor.shutdown()
    thread.join(timeout=2.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
