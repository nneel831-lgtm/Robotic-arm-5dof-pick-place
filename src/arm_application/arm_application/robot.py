# robot.py
from typing import List


class Robot:
    """Static configuration constants for the 5-DOF manipulator."""

    NAME: str = "Reference_Assembly_Home1"

    PLANNING_GROUP: str = "arm"
    BASE_LINK: str = "base_link"
    END_EFFECTOR: str = "Wrist_R"

    JOINT_NAMES: List[str] = ["J1", "J2", "J3", "J4", "J5"]

    HOME_POSITION: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Non-singular joint configuration used before Cartesian planning.
    # Small offsets on J2/J3 break the all-zero singularity so KDL's
    # Jacobian-based IK has a well-conditioned seed to converge from.
    # Values are conservative and well within the +-120 deg (~2.094 rad) limits.
    READY_POSITION: List[float] = [0.0, 0.55, -0.35, 0.0, 0.0]

    JOINT_LIMIT_MIN: float = -2.094  # -120 deg
    JOINT_LIMIT_MAX: float = 2.094   # +120 deg

    WORKSPACE_X_MIN: float = -0.6
    WORKSPACE_X_MAX: float = 0.6
    WORKSPACE_Y_MIN: float = -0.6
    WORKSPACE_Y_MAX: float = 0.6
    WORKSPACE_Z_MIN: float = 0.0
    WORKSPACE_Z_MAX: float = 0.8

    PLANNING_TIME: float = 5.0
    PLANNING_ATTEMPTS: int = 10

    VELOCITY_SCALING: float = 0.3
    ACCELERATION_SCALING: float = 0.3
