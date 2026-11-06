# Unit tests for GraphWars-LLM 2D Engine
import sys
import os
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.backend.space import Point2D, CoordinateBounds, Shooter, Target
from src.engine.backend.obstacles import CircleObstacle, BoxObstacle, PolygonObstacle, TerrainObstacle
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

    # Test loading from folder and history file
    loaded_logger = TestSessionLogger.load(logger.history_file_path)
    assert loaded_logger.is_solved is True
    assert len(loaded_logger.history) == 2
    assert loaded_logger.history[0].llm_output["formula"] == "0"
    assert loaded_logger.history[1].llm_output["formula"] == "-0.02 * (x + 20) * (x - 20)"
    assert os.path.exists(logger.scenario_file_path)
    assert os.path.exists(logger.history_file_path)


    shutil.rmtree(logger.test_folder, ignore_errors=True)


def test_length_cap_enforced():
    state = GameState.create_preset("direct_shot")
    state.constraints["max_formula_characters"] = 10
    result = state.fire_formula("0 + 0 * x + 0 * x")  # 17 chars
    assert result.hit_type == HitType.INVALID_FORMULA
    assert "limit is 10" in result.error_message
    assert state.fire_formula("0*x").is_success


def test_invalid_start_rejected():
    state = GameState.create_preset("direct_shot")  # shooter at (-20, 0), radius 0.6
    result = state.fire_formula("5")
    assert result.hit_type == HitType.INVALID_START
    assert not result.is_success
    # Within radius is fine
    assert state.fire_formula("0.5 + 0*x").hit_type != HitType.INVALID_START


def test_unsafe_formulas_rejected():
    state = GameState.create_preset("direct_shot")
    for bad in [
        "().__class__.__bases__[0].__subclasses__()",
        "__import__('os')",
        "x.real",
        "'abc'",
        "[x][0]",
        "lambda: 0",
        "sin",
        "foo(x)",
        "y + 1",
        "2x",
    ]:
        res = state.fire_formula(bad)
        assert res.hit_type == HitType.INVALID_FORMULA, (bad, res.hit_type, res.error_message)


def test_huge_power_does_not_hang():
    state = GameState.create_preset("direct_shot")
    res = state.fire_formula("0*x + 9**9**9")
    assert res.hit_type == HitType.MATH_ERROR


def test_no_tunneling_through_thin_obstacle():
    # Steep step from y=0 to y=10 near x=-4.97 crosses a 0.2-thick bar between two regular samples.
    state = GameState(
        bounds=CoordinateBounds(-25, 25, -15, 15),
        shooter=Shooter("Player1", Point2D(-20, 0)),
        targets=[Target("target_1", Point2D(20, 10))],
        obstacles=[BoxObstacle(x_min=-10, x_max=10, y_min=4.9, y_max=5.1, name="Thin_Bar")],
    )
    res = state.fire_formula("5 * (1 + tanh(20 * (x + 4.97)))")
    assert res.hit_type == HitType.OBSTACLE
    assert res.hit_obstacle_name == "Thin_Bar"


def test_scenario_constraints_reach_payload():
    from src.experiments.runner import BenchmarkRunner
    scen = os.path.join(os.path.dirname(__file__), "..", "experiments", "scenarios", "scenario_5_random_gauntlet.json")
    runner = BenchmarkRunner(scen, output_dir=os.path.join("outputs", "_test_tmp"))
    try:
        payload = runner.get_current_llm_payload()
        assert payload["constraints"]["max_formula_characters"] == 180
        assert runner.game_state.max_formula_characters == 180
    finally:
        shutil.rmtree(os.path.join("outputs", "_test_tmp"), ignore_errors=True)


def test_polygon_and_terrain_roundtrip():
    state = GameState(
        obstacles=[
            PolygonObstacle([Point2D(0, 0), Point2D(4, 0), Point2D(2, 5)], name="Tri"),
            TerrainObstacle([Point2D(-25, -10), Point2D(0, -5), Point2D(25, -10)], name="Ground"),
        ]
    )
    d = state.to_contract_dict()
    assert d["obstacles"][0]["type"] == "polygon" and len(d["obstacles"][0]["vertices"]) == 3
    assert d["obstacles"][1]["type"] == "terrain" and len(d["obstacles"][1]["points"]) == 3
    loaded = GameState.from_dict(d)
    assert isinstance(loaded.obstacles[0], PolygonObstacle)
    assert isinstance(loaded.obstacles[1], TerrainObstacle)
    assert loaded.obstacles[0].contains_point(Point2D(2, 1))
    assert loaded.obstacles[1].contains_point(Point2D(0, -6))


def test_unknown_obstacle_type_raises():
    d = GameState.create_preset("direct_shot").to_contract_dict()
    d["obstacles"] = [{"type": "blob", "name": "X", "bounding_box": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}}]
    try:
        GameState.from_dict(d)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "blob" in str(e)


if __name__ == "__main__":
    print("Running engine unit tests...")
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  [OK] {name}")
    print(f"All {len(tests)} engine tests passed successfully! [OK]")
