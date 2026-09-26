import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from src.engine.backend.physics import TrajectoryResult, HitType


@dataclass
class TestAttemptRecord:
    attempt: int
    timestamp: str
    llm_output: Dict[str, Any]
    simulation_result: Dict[str, Any]


class TestSessionLogger:
    """
    Manages and persists history logs for test runs into individual JSON files inside the `outputs/` directory.
    Each test log captures both the LLM output and the simulation feedback.
    """

    def __init__(self, test_name: str, initial_state_dict: Dict[str, Any], output_dir: str = "outputs"):
        self.test_name = test_name
        self.output_dir = output_dir
        self.session_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.initial_state = initial_state_dict
        self.history: List[TestAttemptRecord] = []
        self.is_solved: bool = False
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_path = os.path.join(self.output_dir, f"{self.session_id}.json")

    def record_attempt(
        self,
        llm_output: Dict[str, Any],
        sim_result: TrajectoryResult,
    ) -> TestAttemptRecord:
        """Record an LLM attempt along with its physics simulation feedback."""
        attempt_num = len(self.history) + 1
        
        sim_dict = {
            "is_success": sim_result.is_success,
            "hit_type": sim_result.hit_type.value,
            "hit_coordinate": (
                {"x": round(sim_result.hit_coordinate.x, 3), "y": round(sim_result.hit_coordinate.y, 3)}
                if sim_result.hit_coordinate
                else None
            ),
            "hit_obstacle": sim_result.hit_obstacle_name,
            "hit_target_id": sim_result.hit_target_id,
            "closest_distance_to_target": round(sim_result.closest_distance_to_target, 3),
            "error_message": sim_result.error_message,
        }

        record = TestAttemptRecord(
            attempt=attempt_num,
            timestamp=datetime.now().isoformat(),
            llm_output=llm_output,
            simulation_result=sim_dict,
        )

        self.history.append(record)
        if sim_result.is_success:
            self.is_solved = True

        self.save()
        return record

    def get_history_for_llm(self) -> List[Dict[str, Any]]:
        """Returns the history formatted according to the LLM trajectory contract."""
        return [
            {
                "attempt": rec.attempt,
                "llm_output": rec.llm_output,
                "simulation_result": rec.simulation_result,
            }
            for rec in self.history
        ]

    def get_llm_payload(self) -> Dict[str, Any]:
        """Constructs the full input JSON payload for the LLM."""
        payload = dict(self.initial_state)
        payload["history"] = self.get_history_for_llm()
        return payload

    def save(self) -> str:
        """Saves current test session history to disk."""
        data = {
            "session_id": self.session_id,
            "test_name": self.test_name,
            "is_solved": self.is_solved,
            "total_attempts": len(self.history),
            "created_at": self.session_id.split("_", 1)[1] if "_" in self.session_id else "",
            "initial_state": self.initial_state,
            "history": [asdict(rec) for rec in self.history],
        }

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return self.file_path

    @classmethod
    def load(cls, filepath: str) -> "TestSessionLogger":
        """Load an existing test session JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger = cls(test_name=data["test_name"], initial_state_dict=data["initial_state"])
        logger.session_id = data["session_id"]
        logger.file_path = filepath
        logger.is_solved = data.get("is_solved", False)
        
        logger.history = [
            TestAttemptRecord(
                attempt=item["attempt"],
                timestamp=item["timestamp"],
                llm_output=item["llm_output"],
                simulation_result=item["simulation_result"],
            )
            for item in data.get("history", [])
        ]
        return logger
