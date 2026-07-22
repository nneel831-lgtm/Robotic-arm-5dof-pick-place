# ikpy_solver.py
"""Standalone IK solver using ikpy, bypassing MoveIt/OMPL/KDL entirely.
Computes joint angles for a Cartesian position target directly from the
URDF, then sends them through Planner.plan_joint()/execute() - the
joint-space code path already confirmed working via the raw MoveGroup
joint-space action test and go_home()/go_ready().
"""
import os
import sys
import argparse

import rclpy
from ikpy.chain import Chain

from arm_application.planner import Planner
from arm_application.goals import JointGoal
from arm_application.robot import Robot

URDF_PATH = os.path.expanduser(
    "~/your_ros2_ws/src/arm_description/urdf/Reference_Assembly44.urdf"
)


def build_chain():
    # Define which links IKPY should actually calculate angles for.
    # We turn OFF the base link (index 0) and the Gripper/J6 (the last index)
    # Adjust the number of 'True's if necessary, but this matches your error output:
    # ['Base link' (False), 'J1' (True), 'J2' (True), 'J3' (True), 'J4' (True), 'J5' (True), 'J6' (False)]
    mask = [False, True, True, True, True, True, False]
    
    chain = Chain.from_urdf_file(
        your_ros2_ws/src/arm_description/urdf/Reference_Assembly44.urdf, 
        base_elements=["base_link"],
        active_links_mask=mask
    )
    return chain

def solve_ik(chain: Chain, x: float, y: float, z: float) -> list:
    solution = chain.inverse_kinematics(target_position=[x, y, z])
    joint_values = [
        angle
        for angle, active in zip(solution, chain.active_links_mask)
        if active
    ]
    if len(joint_values) != len(Robot.JOINT_NAMES):
        raise RuntimeError(
            f"ikpy returned {len(joint_values)} active joint values, "
            f"expected {len(Robot.JOINT_NAMES)}. Chain links: "
            f"{[l.name for l in chain.links]}"
        )
    return joint_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    args = parser.parse_args()

    chain = build_chain()
    joint_values = solve_ik(chain, args.x, args.y, args.z)
    print(f"ikpy solution for J1-J5: {joint_values}")

    rclpy.init()
    planner = Planner()
    try:
        if not planner.plan_joint(JointGoal(positions=joint_values)):
            planner.get_logger().error("plan_joint failed for ikpy solution.")
            return
        if not planner.execute():
            planner.get_logger().error("execute failed.")
            return
        if not planner.wait_until_executed():
            planner.get_logger().error("wait_until_executed failed.")
            return
        planner.get_logger().info("ikpy-driven move completed successfully.")
    finally:
        planner.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
