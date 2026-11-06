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
  - GIF export of a test run's attempts (MP4 needs `imageio` + `imageio-ffmpeg`, not installed yet).
  - Robot emoji start (labeled with the model that ran) and Portal cake goal (`assets/portal_cake.png`), with the real hit radius shown as a dashed circle.

---

### Phase 2: Trajectory Generation & LLM Reasoning
- [x] **2.1 Symbolic Trajectory Representation**:
  - AST-whitelisted parsing and evaluation of formula strings.
  - Enforced per-scenario character limit and shooter start condition.
- [x] **2.2 LLM Search & Spatial Reasoning** (`src/llm_agent/run_single_test_agent.py`):
  - JSON contract payload (`LLM_trajectory_contract.md`) for Gemini / OpenAI / Anthropic / mock.
  - Iterative attempts with attempt budget and collision feedback history.
  - Per-attempt logs: raw request/response and thinking summary (Anthropic `display: "summarized"`; the raw chain of thought is never returned).
  - No tools are sent with requests, so models solve without running code (verifiable in `attempt_N_raw_request.json`).
- [ ] **2.3 Improvements**:
  - Thinking summaries for Gemini / OpenAI reasoning models.
  - Exposing more geometry/feedback (e.g. clearance margins), multi-target scenarios.

---

### Phase 3: Benchmarks & Experiments
- [x] **3.1 Test Scenarios**: direct shot, center pillar, slalom, bunker, random gauntlet (`src/experiments/generator.py`), needle gauntlet (scenario 6, hand-designed, verified solvable with a 136-char reference formula).
- [ ] **3.2 Evaluation**:
  - Batch runner across scenarios × models × seeds.
  - Success rates, attempts-to-hit, trajectory efficiency, and formula compactness.
- [ ] **3.3 Harder Scenarios**: single runs so far are solved on attempt 1, so the benchmark needs more headroom:
  - Tighter formula limits (e.g. 100 chars) to force compact math.
  - Obstacle shapes that boxes and circles can't describe (polygons, terrain).
  - Scenarios where a first attempt is likely to fail, to exercise the feedback loop.

---

## 📊 Results So Far (single runs, `claude-sonnet-5-5`, 5-attempt budget)

| Scenario | Result | Formula length | Thinking tokens |
|---|---|---|---|
| 2 · Center pillar | Hit on attempt 1 | 22 / 120 | 0 |
| 5 · Random gauntlet | Hit on attempt 1 | 109 / 180 | ~4,200 |
| 6 · Needle gauntlet | Hit on attempt 1 | 179 / 180 | ~14,000 |

One run each, so these are examples, not success rates.

---

## 📌 Current Status & Next Steps
1. ✅ Engine, UI, playback and GIF export with themed icons.
2. ✅ Single-test LLM agent runner with logged history, thinking summaries and raw API traffic.
3. ✅ Needle gauntlet example in the README.
4. 🔄 Next: batch benchmark runner and aggregate metrics (Phase 3.2).
5. ⏳ Harder scenarios (Phase 3.3).
6. ⏳ LinkedIn video: MP4 story cut (title card, arena, plan, shot, result).
