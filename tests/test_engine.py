# Unit tests for GraphWars-LLM 2D Engine
import sys
import os
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.backend.space import Point2D, CoordinateBounds, Shooter, Target
from src.engine.backend.obstacles import CircleObstacle, BoxObstacle, PolygonObstacle
from src.engine.backend.physics import TrajectoryEvaluator, HitType
from src.engine.backend.game_state import GameState
from src.engine.backend.session_logger import TestSessionLogger


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
    result = state.fire_formula("0")
    assert result.is_success
    assert result.hit_type == HitType.TARGET
    assert result.hit_target_id == "target_1"


def test_obstacle_collision():
    state = GameState.create_preset("pillar")
    result = state.fire_formula("0")
    assert not result.is_success
    assert result.hit_type == HitType.OBSTACLE
    assert result.hit_obstacle_name == "Center_Pillar"


def test_arc_over_obstacle():
    state = GameState.create_preset("pillar")
    result = state.fire_formula("-0.02 * (x + 20) * (x - 20)")
    assert result.is_success
    assert result.hit_type == HitType.TARGET
    assert result.hit_target_id == "target_1"


def test_session_logger_history():
    state = GameState.create_preset("pillar")
    output_test_dir = "outputs"
    logger = TestSessionLogger(test_name="test_pillar_history", initial_state_dict=state.to_contract_dict(), output_dir=output_test_dir)
    
    # Attempt 1: Failed shot (hit obstacle)
    llm_output_1 = {
        "reasoning": "Attempt direct line to target",
        "strategy": "direct",
        "planned_waypoints": [{"x": -20.0, "y": 0.0}, {"x": 20.0, "y": 0.0}],
        "formula": "0"
    }
    sim_res_1 = state.fire_formula(llm_output_1["formula"])
    logger.record_attempt(llm_output_1, sim_res_1)
    
    assert len(logger.history) == 1
    assert not logger.is_solved
    assert logger.history[0].simulation_result["hit_type"] == "obstacle"

    # Attempt 2: Successful arc over
    llm_output_2 = {
        "reasoning": "Clear the pillar at y=6 using an inverted parabola with apex at y=8",
        "strategy": "arc_over",
        "planned_waypoints": [{"x": -20.0, "y": 0.0}, {"x": 0.0, "y": 8.0}, {"x": 20.0, "y": 0.0}],
        "formula": "-0.02 * (x + 20) * (x - 20)"
    }
    sim_res_2 = state.fire_formula(llm_output_2["formula"])
    logger.record_attempt(llm_output_2, sim_res_2)
    
    assert len(logger.history) == 2
    assert logger.is_solved
    assert logger.history[1].simulation_result["is_success"] is True

    # Test loading from file
    loaded_logger = TestSessionLogger.load(logger.file_path)
    assert loaded_logger.is_solved is True
    assert len(loaded_logger.history) == 2
    assert loaded_logger.history[0].llm_output["formula"] == "0"
    assert loaded_logger.history[1].llm_output["formula"] == "-0.02 * (x + 20) * (x - 20)"


if __name__ == "__main__":
    print("Running engine unit tests...")
    test_space_and_bounds()
    test_circle_obstacle()
    test_box_obstacle()
    test_direct_hit_trajectory()
    test_obstacle_collision()
    test_arc_over_obstacle()
    test_session_logger_history()
    print("All engine tests passed successfully! [OK]")
