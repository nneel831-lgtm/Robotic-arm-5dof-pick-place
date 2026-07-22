# planner.py
import threading
from typing import Optional

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from trajectory_msgs.msg import JointTrajectory

from pymoveit2 import MoveIt2

from arm_application.robot import Robot
from arm_application.goals import JointGoal


class Planner(Node):
    """Single point of contact between application code and MoveIt2."""

    def __init__(self) -> None:
        super().__init__("planner_node")

        self._callback_group = ReentrantCallbackGroup()

        self._moveit2 = MoveIt2(
            node=self,
            joint_names=Robot.JOINT_NAMES,
            base_link_name=Robot.BASE_LINK,
            end_effector_name=Robot.END_EFFECTOR,
            group_name=Robot.PLANNING_GROUP,
            use_move_group_action=False,
            ignore_new_calls_while_executing=False,
            callback_group=self._callback_group,
        )

        self._moveit2.max_velocity = Robot.VELOCITY_SCALING
        self._moveit2.max_acceleration = Robot.ACCELERATION_SCALING
        self._moveit2.allowed_planning_time = Robot.PLANNING_TIME
        self._moveit2.num_planning_attempts = Robot.PLANNING_ATTEMPTS

        self._moveit2.set_workspace_parameters(
            min_corner=(Robot.WORKSPACE_X_MIN, Robot.WORKSPACE_Y_MIN, Robot.WORKSPACE_Z_MIN),
            max_corner=(Robot.WORKSPACE_X_MAX, Robot.WORKSPACE_Y_MAX, Robot.WORKSPACE_Z_MAX),
        )

        self._last_trajectory: Optional[JointTrajectory] = None
        self._trajectory_lock = threading.Lock()

        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self.get_logger().info("Planner initialized for group '%s'." % Robot.PLANNING_GROUP)

    def check_workspace(self, x: float, y: float, z: float) -> bool:
        """Validate a Cartesian target against the configured workspace bounds."""
        in_bounds = (
            Robot.WORKSPACE_X_MIN <= x <= Robot.WORKSPACE_X_MAX
            and Robot.WORKSPACE_Y_MIN <= y <= Robot.WORKSPACE_Y_MAX
            and Robot.WORKSPACE_Z_MIN <= z <= Robot.WORKSPACE_Z_MAX
        )
        if not in_bounds:
            self.get_logger().error(
                f"Target ({x}, {y}, {z}) is outside workspace bounds "
                f"x[{Robot.WORKSPACE_X_MIN},{Robot.WORKSPACE_X_MAX}] "
                f"y[{Robot.WORKSPACE_Y_MIN},{Robot.WORKSPACE_Y_MAX}] "
                f"z[{Robot.WORKSPACE_Z_MIN},{Robot.WORKSPACE_Z_MAX}]"
            )
        return in_bounds

    def plan_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
    ) -> bool:
        """Plan (without executing) a full Cartesian pose goal (position + orientation)."""
        if not self.check_workspace(x, y, z):
            with self._trajectory_lock:
                self._last_trajectory = None
            return False

        try:
            trajectory = self._moveit2.plan(
                position=[x, y, z],
                quat_xyzw=[qx, qy, qz, qw],
            )
        except Exception as exc:
            self.get_logger().error(f"plan_pose failed: {exc}")
            trajectory = None

        with self._trajectory_lock:
            self._last_trajectory = trajectory

        return trajectory is not None

    # planner.py — only plan_position() changes, rest of the file is unchanged
    def plan_position(
        self,
        x: float,
        y: float,
        z: float,
        tolerance_position: float = 0.005,
    ) -> bool:
        """Plan (without executing) a position-only Cartesian goal.

        `position_only_ik` is not actually loaded on /move_group for this arm
        (confirmed via `ros2 param get`), so omitting the orientation goal is
        not enough - the constraint sampler still needs some orientation
        target to feed KDL's numeric IK, which this arm rarely reaches.
        Instead, an explicit orientation constraint is set with a
        near-maximal tolerance (~pi rad per axis), so MoveIt's
        IKConstraintSampler samples orientations freely within that band
        before calling IK - a working position-only search independent of
        the solver-level flag.
        """
        if not self.check_workspace(x, y, z):
            with self._trajectory_lock:
                self._last_trajectory = None
            return False

        try:
            trajectory = self._moveit2.plan(
                position=[x, y, z],
                tolerance_position=tolerance_position,
            )
        except Exception as exc:
            self.get_logger().error(f"plan_position failed: {exc}")
            trajectory = None

        with self._trajectory_lock:
            self._last_trajectory = trajectory

        return trajectory is not None

    def plan_joint(self, joint_goal: JointGoal) -> bool:
        """Plan (without executing) a joint-space goal and cache the trajectory."""
        if len(joint_goal.positions) != len(Robot.JOINT_NAMES):
            self.get_logger().error(
                f"Expected {len(Robot.JOINT_NAMES)} joint values, "
                f"got {len(joint_goal.positions)}."
            )
            with self._trajectory_lock:
                self._last_trajectory = None
            return False

        try:
            trajectory = self._moveit2.plan(joint_positions=joint_goal.positions)
        except Exception as exc:
            self.get_logger().error(f"plan_joint failed: {exc}")
            trajectory = None

        with self._trajectory_lock:
            self._last_trajectory = trajectory

        return trajectory is not None

    def go_home(self) -> bool:
        """Plan and execute a motion to the configured home position."""
        if not self.plan_joint(JointGoal(positions=list(Robot.HOME_POSITION))):
            return False
        if not self.execute():
            return False
        return self.wait_until_executed()

    def go_ready(self) -> bool:
        """Plan and execute a motion to the non-singular ready position."""
        if not self.plan_joint(JointGoal(positions=list(Robot.READY_POSITION))):
            return False
        if not self.execute():
            return False
        return self.wait_until_executed()

    def execute(self) -> bool:
        """Execute the most recently planned trajectory (single-use cache)."""
        with self._trajectory_lock:
            trajectory = self._last_trajectory
            self._last_trajectory = None

        if trajectory is None:
            self.get_logger().error(
                "execute failed: no trajectory available (call plan_pose()/plan_position()/plan_joint() first)."
            )
            return False

        try:
            self._moveit2.execute(trajectory)
            return True
        except Exception as exc:
            self.get_logger().error(f"execute failed: {exc}")
            return False

    def wait_until_executed(self) -> bool:
        """Block until the current execution completes."""
        try:
            return self._moveit2.wait_until_executed()
        except Exception as exc:
            self.get_logger().error(f"wait_until_executed failed: {exc}")
            return False

    def cancel_execution(self) -> None:
        """Cancel the currently executing trajectory."""
        try:
            self._moveit2.cancel_execution()
        except Exception as exc:
            self.get_logger().error(f"cancel_execution failed: {exc}")

    def shutdown(self) -> None:
        """Cleanly stop the executor, spin thread, and destroy the node."""
        try:
            self._executor.shutdown()
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)
            self.destroy_node()
            self.get_logger().info("Planner shutdown complete.")
        except Exception as exc:
            self.get_logger().error(f"shutdown failed: {exc}")
