# GraphWars-LLM Implementation Plan

## Objective
Build an experimental framework and simulation environment to explore **LLM-Guided Symbolic Trajectory Generation** in Graphwar 2D Cartesian obstacle spaces.

---

## 🎯 Phased Roadmap

### Phase 1: Game Engine & Simulation Environment (`src/engine/`)
- [ ] **1.1 Backend Engine (`src/engine/backend/`)**:
  - Continuous Cartesian coordinate system ($x \in [x_{\min}, x_{\max}], y \in [y_{\min}, y_{\max}]$).
  - Configurable players / shooters ($P_{\text{start}}$) and targets ($P_{\text{target}}$).
  - Obstacle shapes (circles, boxes, polygons, terrain heightmaps).
  - Trajectory evaluation & continuous collision detection for arbitrary mathematical functions $y = f(x)$.
  - Exact impact calculation (hit coordinate, obstacle hit, target hit, or out of bounds).
- [ ] **1.2 UI & Visualizer (`src/engine/ui/`)**:
  - Real-time interactive 2D graphical display (Pygame / Matplotlib GUI).
  - Visual rendering of terrain, obstacles, shooting nodes, targets, grid lines, and trajectory curves.
  - Interactive formula tester input to visually test functions live.

---

### Phase 2: Trajectory Generation & LLM Reasoning
- [ ] **2.1 Symbolic Trajectory Representation**:
  - Parsing, evaluating, and validating mathematical string formulas.
  - Character limit constraints and domain safety checking.
- [ ] **2.2 LLM Search & Spatial Reasoning**:
  - Representing 2D game state for LLMs.
  - Iterative search, refinement, and collision feedback handling.

---

### Phase 3: Benchmarks & Experiments
- [ ] **3.1 Test Scenarios**:
  - Varied difficulty levels (direct sight, single obstacle, narrow corridors, complex terrain).
- [ ] **3.2 Evaluation**:
  - Success rates, trajectory efficiency, and formula compactness.

---

## 📌 Current Status & Next Steps
1. ✅ Updated clean README and roadmap.
2. 🔄 Building `src/engine/` (Backend + Interactive UI).
