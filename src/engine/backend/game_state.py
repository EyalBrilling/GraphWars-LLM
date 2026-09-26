from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.engine.backend.space import Point2D, CoordinateBounds, Shooter, Target
from src.engine.backend.obstacles import BaseObstacle, CircleObstacle, BoxObstacle, TerrainObstacle
from src.engine.backend.physics import TrajectoryEvaluator, TrajectoryResult, HitType


class GameState:
    """
    Encapsulates the entire 2D Graphwar arena state:
    bounds, shooter, targets, obstacles, and shot history.
    """

    def __init__(
        self,
        bounds: Optional[CoordinateBounds] = None,
        shooter: Optional[Shooter] = None,
        targets: Optional[List[Target]] = None,
        obstacles: Optional[List[BaseObstacle]] = None,
    ):
        self.bounds = bounds or CoordinateBounds(x_min=-25.0, x_max=25.0, y_min=-15.0, y_max=15.0)
        self.shooter = shooter or Shooter(name="Player1", position=Point2D(-20.0, 0.0))
        self.targets = targets or [Target(id="target_1", position=Point2D(20.0, 0.0))]
        self.obstacles = obstacles or []
        self.history: List[TrajectoryResult] = []

    def add_obstacle(self, obstacle: BaseObstacle) -> None:
        self.obstacles.append(obstacle)

    def fire_formula(self, formula_str: str, step_size: float = 0.05) -> TrajectoryResult:
        """Evaluate a mathematical formula shot and update state."""
        # Determine firing direction based on target position relative to shooter
        main_target = [t for t in self.targets if t.is_alive][0] if any(t.is_alive for t in self.targets) else self.targets[0]
        direction = 1 if main_target.position.x >= self.shooter.position.x else -1

        result = TrajectoryEvaluator.evaluate(
            formula_str=formula_str,
            shooter=self.shooter,
            targets=self.targets,
            obstacles=self.obstacles,
            bounds=self.bounds,
            step_size=step_size,
            direction=direction,
        )

        if result.is_success and result.hit_target_id:
            for t in self.targets:
                if t.id == result.hit_target_id:
                    t.is_alive = False

        self.history.append(result)
        return result

    def get_text_description(self) -> str:
        """Generate a concise textual representation of the arena."""
        lines = [
            f"=== 2D Arena State ===",
            f"Coordinate Bounds: X in [{self.bounds.x_min}, {self.bounds.x_max}], Y in [{self.bounds.y_min}, {self.bounds.y_max}]",
            f"Shooter ({self.shooter.name}): Position = ({self.shooter.position.x:.2f}, {self.shooter.position.y:.2f})",
        ]
        
        lines.append("Targets:")
        for t in self.targets:
            status = "ALIVE" if t.is_alive else "DESTROYED"
            lines.append(f"  - [{t.id}] at ({t.position.x:.2f}, {t.position.y:.2f}) | Status: {status} | Radius: {t.radius}")

        lines.append(f"Obstacles ({len(self.obstacles)} total):")
        for i, obs in enumerate(self.obstacles):
            bbox = obs.get_bounding_box()
            lines.append(f"  - Obstacle {i+1} [{obs.name}]: BBox X=[{bbox[0]:.2f}, {bbox[1]:.2f}], Y=[{bbox[2]:.2f}, {bbox[3]:.2f}]")

        if self.history:
            last = self.history[-1]
            lines.append("Last Shot Result:")
            lines.append(f"  Formula: y = {last.formula_str}")
            lines.append(f"  Outcome: {last.hit_type.value.upper()}")
            if last.hit_coordinate:
                lines.append(f"  Impact Point: ({last.hit_coordinate.x:.2f}, {last.hit_coordinate.y:.2f})")
            if last.hit_obstacle_name:
                lines.append(f"  Collided Obstacle: {last.hit_obstacle_name}")
            lines.append(f"  Closest Distance to Target: {last.closest_distance_to_target:.3f}")

        return "\n".join(lines)

    def to_contract_dict(self, max_formula_characters: int = 120) -> Dict[str, Any]:
        """Serialize current state to the exact LLM trajectory contract schema."""
        return {
            "bounds": {
                "x_min": self.bounds.x_min,
                "x_max": self.bounds.x_max,
                "y_min": self.bounds.y_min,
                "y_max": self.bounds.y_max,
            },
            "shooter": {
                "name": self.shooter.name,
                "x": self.shooter.position.x,
                "y": self.shooter.position.y,
                "radius": self.shooter.radius,
            },
            "targets": [
                {
                    "id": t.id,
                    "x": t.position.x,
                    "y": t.position.y,
                    "radius": t.radius,
                    "is_alive": t.is_alive,
                }
                for t in self.targets
            ],
            "obstacles": [
                {
                    "id": f"obs_{i+1}",
                    "type": obs.__class__.__name__.replace("Obstacle", "").lower(),
                    "name": obs.name,
                    "bounding_box": {
                        "x_min": obs.get_bounding_box()[0],
                        "x_max": obs.get_bounding_box()[1],
                        "y_min": obs.get_bounding_box()[2],
                        "y_max": obs.get_bounding_box()[3],
                    },
                }
                for i, obs in enumerate(self.obstacles)
            ],
            "constraints": {
                "max_formula_characters": max_formula_characters,
                "variable": "x",
                "allowed_operators": [
                    "+", "-", "*", "/", "**", "sin", "cos", "tan", "exp", "log", "sqrt", "abs"
                ],
            },
            "history": [
                {
                    "attempt": idx + 1,
                    "formula": h.formula_str,
                    "hit_type": h.hit_type.value,
                    "hit_coordinate": (
                        {"x": h.hit_coordinate.x, "y": h.hit_coordinate.y}
                        if h.hit_coordinate
                        else None
                    ),
                    "hit_obstacle": h.hit_obstacle_name,
                    "closest_distance_to_target": round(h.closest_distance_to_target, 3),
                    "error_message": h.error_message,
                }
                for idx, h in enumerate(self.history)
            ],
        }

    def to_contract_json(self, indent: int = 2) -> str:
        """Return formatted JSON matching the LLM trajectory contract."""
        import json
        return json.dumps(self.to_contract_dict(), indent=indent)

    @classmethod
    def create_preset(cls, preset_name: str = "pillar") -> "GameState":
        """Generate common test environments."""
        if preset_name == "direct_shot":
            return cls(
                bounds=CoordinateBounds(-25, 25, -15, 15),
                shooter=Shooter("Player1", Point2D(-20, 0)),
                targets=[Target("target_1", Point2D(20, 0))],
                obstacles=[],
            )

        elif preset_name == "pillar":
            # Single tall wall in the center
            return cls(
                bounds=CoordinateBounds(-25, 25, -15, 15),
                shooter=Shooter("Player1", Point2D(-20, 0)),
                targets=[Target("target_1", Point2D(20, 0))],
                obstacles=[
                    BoxObstacle(x_min=-2.0, x_max=2.0, y_min=-15.0, y_max=6.0, name="Center_Pillar", color="#475569")
                ],
            )

        elif preset_name == "slalom":
            # Multiple alternating pillars (S-curve required)
            return cls(
                bounds=CoordinateBounds(-25, 25, -15, 15),
                shooter=Shooter("Player1", Point2D(-22, 0)),
                targets=[Target("target_1", Point2D(22, 0))],
                obstacles=[
                    BoxObstacle(x_min=-12, x_max=-8, y_min=-15, y_max=4, name="Pillar_1_Bottom", color="#475569"),
                    BoxObstacle(x_min=-2, x_max=2, y_min=-4, y_max=15, name="Pillar_2_Top", color="#475569"),
                    BoxObstacle(x_min=8, x_max=12, y_min=-15, y_max=4, name="Pillar_3_Bottom", color="#475569"),
                ],
            )

        elif preset_name == "bunker":
            # Target shielded inside a bunker with an opening
            return cls(
                bounds=CoordinateBounds(-25, 25, -15, 15),
                shooter=Shooter("Player1", Point2D(-20, -5)),
                targets=[Target("target_1", Point2D(18, 5))],
                obstacles=[
                    CircleObstacle(center=Point2D(0, 0), radius=5.0, name="Central_Sphere", color="#475569"),
                    BoxObstacle(x_min=14, x_max=15, y_min=0, y_max=10, name="Bunker_Wall", color="#64748B"),
                    BoxObstacle(x_min=14, x_max=22, y_min=9, y_max=10, name="Bunker_Roof", color="#64748B"),
                ],
            )

        elif preset_name == "terrain_valley":
            # Realistic hilly valley terrain
            profile = [
                Point2D(-25, -8), Point2D(-18, -4), Point2D(-10, -10),
                Point2D(0, 2), Point2D(10, -8), Point2D(18, -2), Point2D(25, -6)
            ]
            return cls(
                bounds=CoordinateBounds(-25, 25, -15, 15),
                shooter=Shooter("Player1", Point2D(-22, -3.0)),
                targets=[Target("target_1", Point2D(22, -1.0))],
                obstacles=[
                    TerrainObstacle(profile, name="Hills", color="#334155")
                ],
            )

        else:
            return cls.create_preset("pillar")
