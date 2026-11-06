# GraphWars-LLM Implementation Plan

## Objective
Build an experimental framework and simulation environment to explore **LLM-Guided Symbolic Trajectory Generation** in Graphwar 2D Cartesian obstacle spaces.

---

## 🎯 Phased Roadmap

### Phase 1: Game Engine & Simulation Environment (`src/engine/`)
- [x] **1.1 Backend Engine (`src/engine/backend/`)**:
  - Continuous Cartesian coordinate system ($x \in [x_{\min}, x_{\max}], y \in [y_{\min}, y_{\max}]$).
  - Configurable players / shooters ($P_{\text{start}}$) and targets ($P_{\text{target}}$).
  - Obstacle shapes (circles, boxes, polygons, terrain heightmaps), loadable from scenario JSON.
  - Trajectory evaluation & collision detection for $y = f(x)$, with sub-sampling of steep segments (no tunneling).
  - Exact impact calculation (hit coordinate, obstacle hit, target hit, out of bounds, invalid start/formula).
- [x] **1.2 UI & Visualizer (`src/engine/ui/`, `tools/video_generator.py`)**:
  - Tkinter + Matplotlib GUI with live formula tester and attempt-by-attempt history playback.
  - GIF/MP4 export of a test run's attempts.

---

### Phase 2: Trajectory Generation & LLM Reasoning
- [x] **2.1 Symbolic Trajectory Representation**:
  - AST-whitelisted parsing and evaluation of formula strings.
  - Enforced per-scenario character limit and shooter start condition.
- [x] **2.2 LLM Search & Spatial Reasoning** (`src/llm_agent/run_single_test_agent.py`):
  - JSON contract payload (`LLM_trajectory_contract.md`) for Gemini / OpenAI / Anthropic / mock.
  - Iterative attempts with attempt budget and collision feedback history.
- [ ] **2.3 Improvements**: exposing more geometry/feedback (e.g. clearance margins), multi-target scenarios.

---

### Phase 3: Benchmarks & Experiments
- [x] **3.1 Test Scenarios**: direct shot, center pillar, slalom, bunker, random gauntlet (`src/experiments/generator.py`).
- [ ] **3.2 Evaluation**:
  - Batch runner across scenarios × models × seeds.
  - Success rates, attempts-to-hit, trajectory efficiency, and formula compactness.

---

## 📌 Current Status & Next Steps
1. ✅ Engine, UI, playback and video export.
2. ✅ Single-test LLM agent runner with logged history, thinking and raw API traffic.
3. 🔄 Next: batch benchmark runner and aggregate metrics (Phase 3.2).
