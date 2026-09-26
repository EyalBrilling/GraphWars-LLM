import os
import json
from typing import Dict, Any, Optional, List

from src.engine.backend.space import Point2D, CoordinateBounds, Shooter, Target
from src.engine.backend.obstacles import BaseObstacle, CircleObstacle, BoxObstacle, PolygonObstacle
from src.engine.backend.game_state import GameState
from src.engine.backend.session_logger import TestSessionLogger
from src.engine.backend.physics import TrajectoryResult


class ScenarioLoader:
    """Loads benchmark scenario definitions from JSON."""

    @staticmethod
    def load_scenario(scenario_path: str) -> GameState:
        with open(scenario_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        bounds = CoordinateBounds(
            x_min=data["bounds"]["x_min"],
            x_max=data["bounds"]["x_max"],
            y_min=data["bounds"]["y_min"],
            y_max=data["bounds"]["y_max"],
        )

        shooter = Shooter(
            name=data["shooter"]["name"],
            position=Point2D(data["shooter"]["x"], data["shooter"]["y"]),
            radius=data["shooter"].get("radius", 0.6),
        )

        targets = [
            Target(
                id=t["id"],
                position=Point2D(t["x"], t["y"]),
                radius=t.get("radius", 0.8),
                is_alive=t.get("is_alive", True),
            )
            for t in data["targets"]
        ]

        obstacles: List[BaseObstacle] = []
        for obs in data.get("obstacles", []):
            obs_type = obs.get("type", "box")
            name = obs.get("name", "obstacle")
            if obs_type == "box":
                bbox = obs["bounding_box"]
                obstacles.append(
                    BoxObstacle(
                        x_min=bbox["x_min"],
                        x_max=bbox["x_max"],
                        y_min=bbox["y_min"],
                        y_max=bbox["y_max"],
                        name=name,
                    )
                )
            elif obs_type == "circle":
                bbox = obs["bounding_box"]
                center_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
                center_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
                radius = (bbox["x_max"] - bbox["x_min"]) / 2.0
                obstacles.append(
                    CircleObstacle(
                        center=Point2D(center_x, center_y),
                        radius=radius,
                        name=name,
                    )
                )

        return GameState(bounds=bounds, shooter=shooter, targets=targets, obstacles=obstacles)


class BenchmarkRunner:
    """
    Executes test runs against scenarios and persists history to outputs/.
    """

    def __init__(self, scenario_path: str, output_dir: str = "outputs"):
        self.scenario_path = scenario_path
        self.scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
        self.game_state = ScenarioLoader.load_scenario(scenario_path)
        self.logger = TestSessionLogger(
            test_name=self.scenario_name,
            initial_state_dict=self.game_state.to_contract_dict(),
            output_dir=output_dir,
        )

    def get_current_llm_payload(self) -> Dict[str, Any]:
        """Get the full payload ready to be sent to an LLM."""
        return self.logger.get_llm_payload()

    def process_llm_response(self, llm_response: Dict[str, Any]) -> TrajectoryResult:
        """
        Takes the LLM's JSON response, evaluates the formula in the engine,
        records the result to the history log in outputs/, and returns the trajectory result.
        """
        formula = llm_response.get("formula", "")
        sim_result = self.game_state.fire_formula(formula)
        self.logger.record_attempt(llm_output=llm_response, sim_result=sim_result)
        return sim_result
