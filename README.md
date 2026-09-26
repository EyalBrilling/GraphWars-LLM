# GraphWars-LLM

**Exploring LLMs for Symbolic Trajectory Generation in Graphwar**

---

## What is Graphwar?

**Graphwar** is an artillery game on a 2D Cartesian plane where projectiles are driven by mathematical functions ($y = f(x)$ or differential equations) rather than standard angle and power. Players must formulate equations that navigate around terrain and obstacles to hit opponent targets.

---

## The Challenge

1. **Continuous 2D Spatial Geometry:** Navigating complex obstacles and terrain without clipping.
2. **Formula Constraints:** Mathematical functions must satisfy strict game constraints such as character length limits and valid mathematical operators.
3. **Pure Math Limitations:** Traditional numerical interpolation and curve-fitting methods (like Fourier series or DCT) lack spatial reasoning, often resulting in severe oscillations (Runge's phenomenon) or obstacle collisions without human guidance.

---

## Project Goal

This project explores how Large Language Models (LLMs) can be leveraged for spatial search, reasoning, and generating symbolic mathematical trajectories in constrained 2D environments.

---

## Repository Structure

```
graphwars_LLM/
├── README.md               # Project overview
├── plan.md                 # Implementation steps & roadmap
├── src/
│   ├── engine/             # 2D simulation environment
│   │   ├── backend/        # Coordinate space, obstacles, trajectory physics & collision
│   │   └── ui/             # Real-time visual interface & rendering
│   ├── llm_agent/          # LLM reasoning & agent components
│   └── experiments/        # Scenarios, benchmarks & tests
└── tests/                  # Unit and integration tests
```
