import ast
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any
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
    INVALID_FORMULA = "invalid_formula"  # disallowed syntax, unparseable, or over the length cap
    INVALID_START = "invalid_start"      # f(x_shooter) is not within the shooter's radius


class FormulaValidationError(ValueError):
    """Raised when a formula uses syntax outside the allowed math subset."""


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


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


class _FloatConstants(ast.NodeTransformer):
    """Turn integer literals into floats so e.g. 9**9**9 overflows instead of hanging on bigint math."""

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, int):
            return ast.copy_location(ast.Constant(value=float(node.value)), node)
        return node


class TrajectoryEvaluator:
    """
    Parses mathematical formulas y = f(x) and performs fine-grained
    continuous collision detection against bounds, obstacles, and targets.
    """

    SAFE_FUNCTIONS: Dict[str, Any] = {
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
        "min": np.minimum,
        "max": np.maximum,
        "floor": np.floor,
        "ceil": np.ceil,
    }
    SAFE_CONSTANTS: Dict[str, float] = {"pi": np.pi, "e": np.e}
    SAFE_ENV: Dict[str, Any] = {**SAFE_FUNCTIONS, **SAFE_CONSTANTS}

    # Max sub-samples inserted between two regular steps when the curve is steep.
    MAX_SUBDIVISIONS = 400

    @classmethod
    def sanitize_formula(cls, formula_str: str) -> str:
        """Convert standard math syntax (e.g. ^ -> **) to valid Python syntax."""
        cleaned = formula_str.strip()
        cleaned = cleaned.replace("^", "**")
        return cleaned

    @classmethod
    def _validate_node(cls, node: ast.AST) -> None:
        """Whitelist walk: only numbers, x, known constants, arithmetic and known function calls."""
        if isinstance(node, ast.Expression):
            cls._validate_node(node.body)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise FormulaValidationError(f"Operator '{type(node.op).__name__}' is not allowed")
            cls._validate_node(node.left)
            cls._validate_node(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise FormulaValidationError(f"Operator '{type(node.op).__name__}' is not allowed")
            cls._validate_node(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in cls.SAFE_FUNCTIONS:
                raise FormulaValidationError("Only calls to the supported math functions are allowed")
            if node.keywords:
                raise FormulaValidationError("Keyword arguments are not allowed")
            if not node.args:
                raise FormulaValidationError(f"Function '{node.func.id}' needs an argument")
            for arg in node.args:
                cls._validate_node(arg)
        elif isinstance(node, ast.Name):
            if node.id != "x" and node.id not in cls.SAFE_CONSTANTS:
                if node.id in cls.SAFE_FUNCTIONS:
                    raise FormulaValidationError(f"Function '{node.id}' must be called, e.g. {node.id}(x)")
                raise FormulaValidationError(f"Unknown name '{node.id}' (the only variable is 'x')")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise FormulaValidationError(f"Literal {node.value!r} is not allowed; only numbers")
        else:
            raise FormulaValidationError(f"Syntax '{type(node).__name__}' is not allowed")

    @classmethod
    def compile_formula(cls, formula_str: str) -> Callable[[float], float]:
        """
        Validate and compile a formula string into a callable.
        Raises FormulaValidationError (or SyntaxError) if the formula is not allowed.
        """
        sanitized = cls.sanitize_formula(formula_str)
        if not sanitized:
            raise FormulaValidationError("Formula is empty")

        tree = ast.parse(sanitized, mode="eval")
        cls._validate_node(tree)
        tree = ast.fix_missing_locations(_FloatConstants().visit(tree))
        code = compile(tree, "<formula>", "eval")

        def func(x_val: float) -> float:
            env = dict(cls.SAFE_ENV)
            env["x"] = float(x_val)
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
        max_formula_characters: Optional[int] = None,
        enforce_start: bool = True,
    ) -> TrajectoryResult:
        """
        Trace the trajectory starting from shooter.position.x in the specified direction.

        - Formulas longer than max_formula_characters (when given) are rejected as INVALID_FORMULA.
        - With enforce_start, f(x_shooter) must lie within the shooter's radius, else INVALID_START.
        - Steep segments are sub-sampled so the curve cannot jump over thin obstacles between steps.
        """
        points: List[Point2D] = []
        min_dist = [float("inf")]

        def make_result(
            hit_type: HitType,
            coord: Optional[Point2D],
            error: Optional[str] = None,
            obstacle_name: Optional[str] = None,
            target_id: Optional[str] = None,
        ) -> TrajectoryResult:
            success = hit_type == HitType.TARGET
            return TrajectoryResult(
                formula_str=formula_str,
                is_success=success,
                hit_type=hit_type,
                hit_coordinate=coord,
                hit_obstacle_name=obstacle_name,
                hit_target_id=target_id,
                trajectory_points=points,
                closest_distance_to_target=0.0 if success else min_dist[0],
                error_message=error,
            )

        # --- Static checks: type, length, syntax ---------------------------------
        if not isinstance(formula_str, str):
            return make_result(HitType.INVALID_FORMULA, None, f"Formula must be a string, got {type(formula_str).__name__}")

        if max_formula_characters is not None and len(formula_str.strip()) > max_formula_characters:
            return make_result(
                HitType.INVALID_FORMULA,
                None,
                f"Formula is {len(formula_str.strip())} characters; the limit is {max_formula_characters}",
            )

        try:
            func = cls.compile_formula(formula_str)
        except (FormulaValidationError, SyntaxError) as e:
            return make_result(HitType.INVALID_FORMULA, None, f"Invalid formula: {e}")

        def safe_eval(x_val: float) -> float:
            with np.errstate(all="ignore"):
                y_val = func(x_val)
            if math.isnan(y_val) or math.isinf(y_val):
                raise ValueError("Evaluated to NaN or Inf")
            return y_val

        x_start = shooter.position.x
        try:
            y_start = safe_eval(x_start)
        except Exception as e:
            return make_result(HitType.MATH_ERROR, None, f"Formula evaluation error at x={x_start:.3f}: {e}")

        # --- Boundary condition at the shooter -----------------------------------
        if enforce_start:
            start_offset = abs(y_start - shooter.position.y)
            if start_offset > shooter.radius:
                points.append(Point2D(x_start, y_start))
                return make_result(
                    HitType.INVALID_START,
                    Point2D(x_start, y_start),
                    f"f({x_start:g}) = {y_start:.3f}, but the shooter is at y={shooter.position.y:g} "
                    f"(radius {shooter.radius:g}); the curve must start within the shooter",
                )

        alive_targets = [t for t in targets if t.is_alive]
        obstacle_grace = 2.5 * step_size  # ignore obstacles touching the shooter for the first few steps

        def check_point(p: Point2D) -> Optional[TrajectoryResult]:
            points.append(p)
            for t in alive_targets:
                dist = p.distance_to(t.position)
                if dist < min_dist[0]:
                    min_dist[0] = dist
                if dist <= t.radius:
                    return make_result(HitType.TARGET, p, target_id=t.id)
            if not bounds.contains(p):
                return make_result(HitType.OUT_OF_BOUNDS, p)
            if abs(p.x - x_start) > obstacle_grace:
                for obs in obstacles:
                    if obs.contains_point(p):
                        return make_result(HitType.OBSTACLE, p, obstacle_name=obs.name)
            return None

        # --- Trace ----------------------------------------------------------------
        x_limit = max_x if max_x is not None else (bounds.x_max if direction > 0 else bounds.x_min)
        n_steps = int(abs(x_limit - x_start) / step_size + 1e-9)

        res = check_point(Point2D(x_start, y_start))
        if res:
            return res

        x_prev, y_prev = x_start, y_start
        for i in range(1, n_steps + 1):
            x_curr = x_start + direction * i * step_size
            try:
                y_curr = safe_eval(x_curr)
            except Exception as e:
                return make_result(HitType.MATH_ERROR, Point2D(x_curr, y_prev), f"Math domain error at x={x_curr:.3f}: {e}")

            # Sub-sample steep segments so vertical jumps can't skip thin obstacles/targets.
            dy = abs(y_curr - y_prev)
            if dy > step_size:
                k = min(int(math.ceil(dy / step_size)), cls.MAX_SUBDIVISIONS)
                for j in range(1, k):
                    x_sub = x_prev + (x_curr - x_prev) * j / k
                    try:
                        y_sub = safe_eval(x_sub)
                    except Exception as e:
                        return make_result(HitType.MATH_ERROR, Point2D(x_sub, y_prev), f"Math domain error at x={x_sub:.3f}: {e}")
                    res = check_point(Point2D(x_sub, y_sub))
                    if res:
                        return res

            res = check_point(Point2D(x_curr, y_curr))
            if res:
                return res
            x_prev, y_prev = x_curr, y_curr

        return make_result(HitType.NO_HIT, points[-1] if points else None)
