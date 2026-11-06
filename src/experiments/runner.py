import os
import json
from typing import Dict, Any

from src.engine.backend.game_state import GameState
from src.engine.backend.session_logger import TestSessionLogger
from src.engine.backend.physics import HitType, TrajectoryResult


class ScenarioLoader:
    """Loads benchmark scenario definitions from JSON."""

    @staticmethod
    def load_scenario(scenario_path: str) -> GameState:
        with open(scenario_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return GameState.from_dict(data)


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

    def record_no_answer(self, reason: str) -> TrajectoryResult:
        """Record an attempt where the model gave no formula, so it counts as a miss and shows in history."""
        sim_result = TrajectoryResult(
            formula_str="",
            is_success=False,
            hit_type=HitType.NO_ANSWER,
            hit_coordinate=None,
            hit_obstacle_name=None,
            hit_target_id=None,
            trajectory_points=[],
            closest_distance_to_target=float("inf"),
            error_message=reason,
        )
        self.logger.record_attempt(
            llm_output={"reasoning": None, "strategy": None, "planned_waypoints": [], "formula": None},
            sim_result=sim_result,
        )
        return sim_result
