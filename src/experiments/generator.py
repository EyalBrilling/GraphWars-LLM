import random
import json
import os
from typing import Dict, Any, List


def generate_random_gauntlet(
    seed: int = 42,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -30.0,
    y_max: float = 30.0,
    num_pillars: int = 7,
    num_spheres: int = 4,
) -> Dict[str, Any]:
    """
    Generates a large-scale, randomized, non-periodic 2D Cartesian obstacle course.
    """
    rng = random.Random(seed)

    shooter_x = x_min + 5.0
    shooter_y = rng.uniform(-15.0, -5.0)

    target_x = x_max - 5.0
    target_y = rng.uniform(5.0, 20.0)

    obstacles: List[Dict[str, Any]] = []
    
    # Generate randomized alternating & asymmetric pillars
    x_step = (target_x - shooter_x - 10.0) / (num_pillars + 1)
    for i in range(num_pillars):
        x_center = shooter_x + 5.0 + (i + 1) * x_step + rng.uniform(-1.5, 1.5)
        width = rng.uniform(2.5, 4.5)
        
        # Randomly choose if it hangs from top, rises from bottom, or is a floating gate
        obs_mode = rng.choice(["bottom", "top", "floating_bar"])
        
        if obs_mode == "bottom":
            y_top = rng.uniform(-5.0, 15.0)
            obstacles.append({
                "id": f"pillar_bot_{i+1}",
                "type": "box",
                "name": f"Pillar_Bottom_{i+1}",
                "bounding_box": {
                    "x_min": round(x_center - width / 2.0, 2),
                    "x_max": round(x_center + width / 2.0, 2),
                    "y_min": y_min,
                    "y_max": round(y_top, 2),
                }
            })
        elif obs_mode == "top":
            y_bot = rng.uniform(-15.0, 5.0)
            obstacles.append({
                "id": f"pillar_top_{i+1}",
                "type": "box",
                "name": f"Pillar_Top_{i+1}",
                "bounding_box": {
                    "x_min": round(x_center - width / 2.0, 2),
                    "x_max": round(x_center + width / 2.0, 2),
                    "y_min": round(y_bot, 2),
                    "y_max": y_max,
                }
            })
        else:
            # Floating middle blocker
            y_c = rng.uniform(-8.0, 8.0)
            h = rng.uniform(8.0, 14.0)
            obstacles.append({
                "id": f"floating_bar_{i+1}",
                "type": "box",
                "name": f"Floating_Bar_{i+1}",
                "bounding_box": {
                    "x_min": round(x_center - width / 2.0, 2),
                    "x_max": round(x_center + width / 2.0, 2),
                    "y_min": round(y_c - h / 2.0, 2),
                    "y_max": round(y_c + h / 2.0, 2),
                }
            })

    # Add floating spherical mines in open corridors
    for j in range(num_spheres):
        cx = rng.uniform(shooter_x + 10.0, target_x - 10.0)
        cy = rng.uniform(y_min + 8.0, y_max - 8.0)
        radius = rng.uniform(2.5, 4.0)
        obstacles.append({
            "id": f"mine_sphere_{j+1}",
            "type": "circle",
            "name": f"Mine_Sphere_{j+1}",
            "bounding_box": {
                "x_min": round(cx - radius, 2),
                "x_max": round(cx + radius, 2),
                "y_min": round(cy - radius, 2),
                "y_max": round(cy + radius, 2),
            }
        })

    # Sort obstacles by X coordinate
    obstacles.sort(key=lambda o: o["bounding_box"]["x_min"])

    scenario = {
        "name": f"random_gauntlet_seed{seed}",
        "difficulty": "grandmaster",
        "description": f"Large-scale (100x60) randomized non-periodic gauntlet with {len(obstacles)} asymmetric obstacles.",
        "bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "shooter": {
            "name": "Player1",
            "x": round(shooter_x, 2),
            "y": round(shooter_y, 2),
            "radius": 0.8,
        },
        "targets": [
            {
                "id": "target_1",
                "x": round(target_x, 2),
                "y": round(target_y, 2),
                "radius": 1.0,
                "is_alive": True,
            }
        ],
        "obstacles": obstacles,
        "constraints": {
            "max_formula_characters": 180,
            "variable": "x",
            "allowed_operators": [
                "+", "-", "*", "/", "**", "sin", "cos", "tan", "exp", "log", "sqrt", "abs"
            ],
        },
    }

    return scenario


if __name__ == "__main__":
    scen = generate_random_gauntlet(seed=777)
    out_path = "experiments/scenarios/scenario_5_random_gauntlet.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scen, f, indent=2)
    print(f"Generated {out_path} with {len(scen['obstacles'])} randomized obstacles!")
