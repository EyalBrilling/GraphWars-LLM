# Unit tests for GraphWars-LLM 2D Engine
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.backend.space import Point2D, CoordinateBounds, Shooter, Target
from src.engine.backend.obstacles import CircleObstacle, BoxObstacle, PolygonObstacle
from src.engine.backend.physics import TrajectoryEvaluator, HitType
from src.engine.backend.game_state import GameState


def test_space_and_bounds():
    bounds = CoordinateBounds(-20, 20, -10, 10)
    p_inside = Point2D(5, 5)
    p_outside = Point2D(25, 0)
    
    assert bounds.contains(p_inside)
    assert not bounds.contains(p_outside)
    assert bounds.width == 40
    assert bounds.height == 20


def test_circle_obstacle():
    circle = CircleObstacle(center=Point2D(0, 0), radius=5.0)
    assert circle.contains_point(Point2D(0, 0))
    assert circle.contains_point(Point2D(3, 4))  # exactly 5
    assert not circle.contains_point(Point2D(5.1, 0))


def test_box_obstacle():
    box = BoxObstacle(x_min=-5, x_max=5, y_min=-2, y_max=8)
    assert box.contains_point(Point2D(0, 0))
    assert box.contains_point(Point2D(5, 8))
    assert not box.contains_point(Point2D(6, 0))


def test_direct_hit_trajectory():
    state = GameState.create_preset("direct_shot")
    # Shooter at (-20, 0), Target at (20, 0). Straight line y = 0
    result = state.fire_formula("0")
    assert result.is_success
    assert result.hit_type == HitType.TARGET
    assert result.hit_target_id == "target_1"


def test_obstacle_collision():
    state = GameState.create_preset("pillar")
    # Straight line y = 0 should hit the center pillar
    result = state.fire_formula("0")
    assert not result.is_success
    assert result.hit_type == HitType.OBSTACLE
    assert result.hit_obstacle_name == "Center_Pillar"


def test_arc_over_obstacle():
    state = GameState.create_preset("pillar")
    # Parabola with apex at x=0, y=8 that clears pillar (y_max=6) and lands at (20, 0)
    result = state.fire_formula("-0.02 * (x + 20) * (x - 20)")
    assert result.is_success
    assert result.hit_type == HitType.TARGET
    assert result.hit_target_id == "target_1"


if __name__ == "__main__":
    print("Running engine unit tests...")
    test_space_and_bounds()
    test_circle_obstacle()
    test_box_obstacle()
    test_direct_hit_trajectory()
    test_obstacle_collision()
    test_arc_over_obstacle()
    print("All engine tests passed successfully! [OK]")
