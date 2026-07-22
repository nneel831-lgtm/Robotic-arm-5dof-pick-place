"""
Example application:
Move EEF (WrR) to a target XYZ position using MoveIt2 planning.
"""

import time
import rclpy

from arm_application.planner import Planner
from arm_application.goals import PoseGoal


def main():

    rclpy.init()

    planner = Planner()

    try:
        planner.get_logger().info("Waiting for robot state...")

        # Give MoveIt time to receive joint_states
        time.sleep(2.0)

        planner.get_logger().info("Starting XYZ motion")

        # Target position
        #
        # Current EEF from tf:
        # x = 0.278
        # y = 0.014
        # z = 0.270
        #
        # Small safe movement:
        goal = PoseGoal(
            x=0.278,
            y=0.014,
            z=0.270,

            # Keep neutral orientation
            # Later we will read current EEF orientation from TF
            qx=0.707,
            qy=0.000,
            qz=0.000,
            qw=0.707,
        )


        planner.get_logger().info(
            f"Planning to XYZ:"
            f" x={goal.x}"
            f" y={goal.y}"
            f" z={goal.z}"
        )


        success = planner.plan_pose(
            x=goal.x,
            y=goal.y,
            z=goal.z,
            qx=goal.qx,
            qy=goal.qy,
            qz=goal.qz,
            qw=goal.qw
        )


        if not success:

            planner.get_logger().error(
                "Planning failed"
            )

            return


        planner.get_logger().info(
            "Planning successful. Executing..."
        )


        if planner.execute():

            finished = planner.wait_until_executed()

            if finished:
                planner.get_logger().info(
                    "Motion completed successfully"
                )

            else:
                planner.get_logger().error(
                    "Execution failed"
                )

        else:

            planner.get_logger().error(
                "Trajectory execution failed"
            )


    except KeyboardInterrupt:

        planner.get_logger().info(
            "Interrupted by user"
        )


    finally:

        planner.shutdown()
        rclpy.shutdown()



if __name__ == "__main__":
    main()
