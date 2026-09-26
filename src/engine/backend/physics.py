import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any
from enum import Enum

from src.engine.backend.space import Point2D, CoordinateBounds, Target, Shooter
from src.engine.backend.obstacles import BaseObstacle


class HitType(Enum):
    NO_HIT = "no_hit"
    OUT_OF_BOUNDS = "out_of_bounds"
    OBSTACLE = "obstacle"
    TARGET = "target"
    SELF_HIT = "self_hit"
    MATH_ERROR = "math_error"


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    step_idx: int


@dataclass
class TrajectoryResult:
    formula_str: str
    is_success: bool
    hit_type: HitType
    hit_coordinate: Optional[Point2D]
    hit_obstacle_name: Optional[str]
    hit_target_id: Optional[str]
    trajectory_points: List[Point2D]
    closest_distance_to_target: float
    error_message: Optional[str] = None


class TrajectoryEvaluator:
    """
    Parses mathematical formulas y = f(x) and performs fine-grained
    continuous collision detection against bounds, obstacles, and targets.
    """

    SAFE_ENV: Dict[str, Any] = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
        "min": np.minimum,
        "max": np.maximum,
        "floor": np.floor,
        "ceil": np.ceil,
    }

    @classmethod
    def sanitize_formula(cls, formula_str: str) -> str:
        """Convert standard math syntax (e.g. ^ -> **, implicit multiplication) to valid Python syntax."""
        cleaned = formula_str.strip()
        cleaned = cleaned.replace("^", "**")
        return cleaned

    @classmethod
    def compile_formula(cls, formula_str: str) -> Callable[[float], float]:
        """Compile string formula into a callable function."""
        sanitized = cls.sanitize_formula(formula_str)
        
        # Compile expression safely
        code = compile(sanitized, "<string>", "eval")
        
        def func(x_val: float) -> float:
            env = dict(cls.SAFE_ENV)
            env["x"] = x_val
            val = eval(code, {"__builtins__": {}}, env)
            return float(val)

        return func

    @classmethod
    def evaluate(
        cls,
        formula_str: str,
        shooter: Shooter,
        targets: List[Target],
        obstacles: List[BaseObstacle],
        bounds: CoordinateBounds,
        step_size: float = 0.05,
        max_x: Optional[float] = None,
        direction: int = 1,  # +1 for left-to-right, -1 for right-to-left
    ) -> TrajectoryResult:
        """
        Trace the trajectory starting from shooter.position.x in the specified direction.
        """
        try:
            func = cls.compile_formula(formula_str)
            # Verify starter point evaluates cleanly
            y_start_eval = func(shooter.position.x)
        except Exception as e:
            return TrajectoryResult(
                formula_str=formula_str,
                is_success=False,
                hit_type=HitType.MATH_ERROR,
                hit_coordinate=None,
                hit_obstacle_name=None,
                hit_target_id=None,
                trajectory_points=[],
                closest_distance_to_target=float("inf"),
                error_message=f"Formula evaluation error: {str(e)}",
            )

        trajectory_points: List[Point2D] = []
        x_curr = shooter.position.x
        x_limit = max_x if max_x is not None else (bounds.x_max if direction > 0 else bounds.x_min)
        
        target_positions = [t.position for t in targets if t.is_alive]
        min_dist_to_target = float("inf")

        # Trace trajectory
        step_count = 0
        max_steps = int(abs(x_limit - x_curr) / step_size) + 100

        while step_count < max_steps:
            try:
                y_val = func(x_curr)
                if np.isnan(y_val) or np.isinf(y_val):
                    raise ValueError("Evaluated to NaN or Inf")
            except Exception as e:
                return TrajectoryResult(
                    formula_str=formula_str,
                    is_success=False,
                    hit_type=HitType.MATH_ERROR,
                    hit_coordinate=Point2D(x_curr, 0.0),
                    hit_obstacle_name=None,
                    hit_target_id=None,
                    trajectory_points=trajectory_points,
                    closest_distance_to_target=min_dist_to_target,
                    error_message=f"Math domain error at x={x_curr:.3f}: {str(e)}",
                )

            curr_point = Point2D(x_curr, y_val)
            trajectory_points.append(curr_point)

            # Check distance to all targets
            for t in targets:
                if t.is_alive:
                    dist = curr_point.distance_to(t.position)
                    if dist < min_dist_to_target:
                        min_dist_to_target = dist
                    # Target Hit check (inside target radius)
                    if dist <= t.radius:
                        return TrajectoryResult(
                            formula_str=formula_str,
                            is_success=True,
                            hit_type=HitType.TARGET,
                            hit_coordinate=curr_point,
                            hit_obstacle_name=None,
                            hit_target_id=t.id,
                            trajectory_points=trajectory_points,
                            closest_distance_to_target=0.0,
                        )

            # Check Bounds (Out of bounds)
            if not bounds.contains(curr_point):
                return TrajectoryResult(
                    formula_str=formula_str,
                    is_success=False,
                    hit_type=HitType.OUT_OF_BOUNDS,
                    hit_coordinate=curr_point,
                    hit_obstacle_name=None,
                    hit_target_id=None,
                    trajectory_points=trajectory_points,
                    closest_distance_to_target=min_dist_to_target,
                )

            # Check Obstacle Collisions (skip the very start if shooter touches obstacle boundary)
            if step_count > 2:
                for obs in obstacles:
                    if obs.contains_point(curr_point):
                        return TrajectoryResult(
                            formula_str=formula_str,
                            is_success=False,
                            hit_type=HitType.OBSTACLE,
                            hit_coordinate=curr_point,
                            hit_obstacle_name=obs.name,
                            hit_target_id=None,
                            trajectory_points=trajectory_points,
                            closest_distance_to_target=min_dist_to_target,
                        )

            # Advance x
            x_curr += direction * step_size
            step_count += 1
            if (direction > 0 and x_curr > x_limit) or (direction < 0 and x_curr < x_limit):
                break

        return TrajectoryResult(
            formula_str=formula_str,
            is_success=False,
            hit_type=HitType.NO_HIT,
            hit_coordinate=trajectory_points[-1] if trajectory_points else None,
            hit_obstacle_name=None,
            hit_target_id=None,
            trajectory_points=trajectory_points,
            closest_distance_to_target=min_dist_to_target,
        )
