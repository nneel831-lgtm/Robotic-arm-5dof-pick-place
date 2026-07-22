# pick_place.py
"""Cartesian pick-and-place sequence using position-only goals (5-DOF, KDL position_only_ik)."""
import rclpy

from arm_application.planner import Planner

HOVER_Z_OFFSET = 0.01

PICK_X, PICK_Y, PICK_Z = 0.266, 0.0028, 0.456
PLACE_X, PLACE_Y, PLACE_Z = 0.166,-0.0028, 0.460

WAYPOINTS = [
    ("pick_hover", PICK_X, PICK_Y, PICK_Z + HOVER_Z_OFFSET),
    ("pick", PICK_X, PICK_Y, PICK_Z),
    ("pick_hover_retreat", PICK_X, PICK_Y, PICK_Z + HOVER_Z_OFFSET),
    ("place_hover", PLACE_X, PLACE_Y, PLACE_Z + HOVER_Z_OFFSET),
    ("place", PLACE_X, PLACE_Y, PLACE_Z),
    ("place_hover_retreat", PLACE_X, PLACE_Y, PLACE_Z + HOVER_Z_OFFSET),
]


def run_waypoint(planner: Planner, name: str, x: float, y: float, z: float) -> bool:
    """Plan, execute, and wait for a single position-only waypoint."""
    if not planner.plan_position(x, y, z):
        planner.get_logger().error(f"Planning failed for waypoint '{name}' ({x}, {y}, {z}).")
        return False
    if not planner.execute():
        planner.get_logger().error(f"Execution failed for waypoint '{name}'.")
        return False
    if not planner.wait_until_executed():
        planner.get_logger().error(f"Execution did not complete successfully for waypoint '{name}'.")
        return False
    return True


def main() -> None:
    """Run a full pick-and-place sequence starting and ending at Home."""
    rclpy.init()

    planner = Planner()

    try:
        # Leave the Home singularity before any Cartesian planning.
        if not planner.go_ready():
            planner.get_logger().error("Failed to reach Ready position. Aborting.")
            return

        for name, x, y, z in WAYPOINTS:
            if not run_waypoint(planner, name, x, y, z):
                planner.get_logger().error("Pick-and-place sequence aborted.")
                return

        if not planner.go_home():
            planner.get_logger().error("Failed to return to Home position.")
            return

        planner.get_logger().info("Pick-and-place sequence completed successfully.")
    finally:
        planner.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
