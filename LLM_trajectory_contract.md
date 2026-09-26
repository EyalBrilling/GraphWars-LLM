# LLM Trajectory Generation Protocol & Contract

This document defines the formal specification and interface contract for Large Language Models (LLMs) generating mathematical trajectories in GraphWars-LLM.

Any LLM acting as a trajectory generator **MUST** adhere strictly to the input schema, mathematical rules, and output contract defined below.

---

## 1. System Role & Goal

You are the **Trajectory Planner** in a continuous 2D Cartesian plane ($x, y \in \mathbb{R}$).
Your goal is to formulate a mathematical trajectory $y = f(x)$ that starts at the shooter's coordinate $(x_0, y_0)$, navigates around obstacles without colliding, and hits the target $(x_T, y_T)$ within the target's radius.

---

## 2. Input Specification (What the LLM Receives)

The current arena state and the previous history of attempts (loaded from the test session history JSON) are provided to the LLM:

### Input Schema

```json
{
  "bounds": {
    "x_min": -25.0,
    "x_max": 25.0,
    "y_min": -15.0,
    "y_max": 15.0
  },
  "shooter": {
    "name": "Player1",
    "x": -20.0,
    "y": 0.0,
    "radius": 0.6
  },
  "targets": [
    {
      "id": "target_1",
      "x": 20.0,
      "y": 0.0,
      "radius": 0.8,
      "is_alive": true
    }
  ],
  "obstacles": [
    {
      "id": "obs_1",
      "type": "box",
      "name": "Center_Pillar",
      "bounding_box": {
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": -15.0,
        "y_max": 6.0
      }
    }
  ],
  "constraints": {
    "max_formula_characters": 120,
    "variable": "x",
    "allowed_operators": ["+", "-", "*", "/", "**", "sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
  },
  "history": [
    {
      "attempt": 1,
      "llm_output": {
        "reasoning": "Direct shot toward target at (20, 0).",
        "strategy": "direct",
        "planned_waypoints": [{ "x": -20.0, "y": 0.0 }, { "x": 20.0, "y": 0.0 }],
        "formula": "0"
      },
      "simulation_result": {
        "hit_type": "obstacle",
        "hit_coordinate": { "x": -2.0, "y": 0.0 },
        "hit_obstacle": "Center_Pillar",
        "closest_distance_to_target": 22.0,
        "error_message": null
      }
    }
  ]
}
```

---

## 3. Mathematical & Syntax Rules for Formulas

1. **Explicit Function of $x$:** The formula must compute $y$ as an explicit mathematical function of $x$.
2. **Standard Python / Math Syntax:**
   - Use `*` for multiplication (e.g. `2 * x`, NOT `2x`).
   - Use `**` or `^` for exponentiation (e.g. `x**2` or `x^2`).
   - Supported functions: `sin(x)`, `cos(x)`, `tan(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `abs(x)`, `pi`, `e`.
3. **Boundary Condition:** At $x = x_{\text{shooter}}$, $f(x_{\text{shooter}})$ must be close to $y_{\text{shooter}}$ (within shooter radius).
4. **Target Condition:** At $x = x_{\text{target}}$, $f(x_{\text{target}})$ should arrive inside the target's bounding radius $[y_T - r_T, y_T + r_T]$.
5. **Length Cap:** The formula string must not exceed `max_formula_characters` (default: 120 chars).

---

## 4. Output Contract (What the LLM Must Return)

The LLM response **MUST** be a valid JSON object matching the following structure:

### Output JSON Schema

```json
{
  "reasoning": "Brief spatial analysis of obstacle clearance and waypoint logic based on arena and history",
  "strategy": "arc_over | s_curve | direct | low_tunnel | trigonometric_wave",
  "planned_waypoints": [
    { "x": -20.0, "y": 0.0 },
    { "x": 0.0, "y": 8.0 },
    { "x": 20.0, "y": 0.0 }
  ],
  "formula": "-0.02 * (x + 20) * (x - 20)"
}
```

### Field Definitions:

| Field | Type | Description | Required |
| :--- | :--- | :--- | :--- |
| `reasoning` | `string` | 1–3 sentences diagnosing obstacles / past attempts and justifying the clearance path. | **Yes** |
| `strategy` | `string` | High-level trajectory archetype (e.g. `arc_over`, `s_curve`, `wave`). | **Yes** |
| `planned_waypoints` | `array of {x, y}` | Key anchor coordinates the curve passes through. | **Yes** |
| `formula` | `string` | The exact mathematical formula string for $y = f(x)$. | **Yes** |

---

## 5. History Files & Test Folder Storage

- Each test run generates its own dedicated folder inside `outputs/`:
  ```
  outputs/<test_name>_<timestamp>/
  ├── scenario.json   # Initial scenario state given to the LLM
  └── history.json    # Complete attempt-by-attempt log with LLM outputs & simulation results
  ```
- Each record inside `history.json` contains:
  1. The **`llm_output`** (the reasoning, strategy, waypoints, and formula produced by the LLM).
  2. The **`simulation_result`** (exact impact coordinate, collision type, obstacle hit, and distance to target).
- The LLM receives this cumulative history on each successive attempt to inform its future spatial decisions.
