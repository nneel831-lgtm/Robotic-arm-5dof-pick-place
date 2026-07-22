# reachability_probe.py
"""Diagnostic: sweep Z downward from the known-reachable tf2_echo point to find
the real lower reach limit before committing to pick/place heights."""
import rclpy

from arm_application.planner import Planner

X, Y = 0.166, 0.000
Z_START = 0.478          # verified reachable via tf2_echo at Home
Z_STOP = 0.05
Z_STEP = -0.05            # 5cm decrements


def main() -> None:
    rclpy.init()
    planner = Planner()

    try:
        if not planner.go_ready():
            planner.get_logger().error("Failed to reach Ready position. Aborting probe.")
            return

        z = Z_START
        while z >= Z_STOP:
            ok = planner.plan_position(X, Y, z)
            planner.get_logger().info(f"z={z:.3f} -> {'OK' if ok else 'FAILED'}")
            z += Z_STEP

        planner.go_home()
    finally:
        planner.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
