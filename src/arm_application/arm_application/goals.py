# goals.py
from dataclasses import dataclass
from typing import List


@dataclass
class JointGoal:
    """Joint-space goal."""

    positions: List[float]


@dataclass
class PoseGoal:
    """Cartesian pose goal (position + orientation)."""

    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
