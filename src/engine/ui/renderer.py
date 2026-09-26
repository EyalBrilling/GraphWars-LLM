import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import numpy as np
from typing import Optional, List

from src.engine.backend.space import CoordinateBounds, Shooter, Target, Point2D
from src.engine.backend.obstacles import BaseObstacle, CircleObstacle, BoxObstacle, PolygonObstacle, TerrainObstacle
from src.engine.backend.physics import TrajectoryResult, HitType
from src.engine.backend.game_state import GameState


class ArenaRenderer:
    """
    Renders the 2D Cartesian Graphwar arena, obstacles, players, and trajectories onto a Matplotlib axes.
    """

    def __init__(self, ax: plt.Axes, bounds: CoordinateBounds):
        self.ax = ax
        self.bounds = bounds

    def render(self, state: GameState, current_result: Optional[TrajectoryResult] = None):
        """Draw the entire scene."""
        self.ax.clear()
        
        # Grid & Background
        self.ax.set_facecolor("#0F172A")  # Deep slate dark mode
        self.ax.grid(True, which="both", color="#1E293B", linestyle="--", linewidth=0.7, alpha=0.8)
        
        # Coordinate axes (x=0, y=0)
        self.ax.axhline(0, color="#475569", linewidth=1.2, linestyle="-")
        self.ax.axvline(0, color="#475569", linewidth=1.2, linestyle="-")

        # Set limits
        self.ax.set_xlim(state.bounds.x_min, state.bounds.x_max)
        self.ax.set_ylim(state.bounds.y_min, state.bounds.y_max)
        self.ax.set_aspect("equal", adjustable="box")

        # Set axis labels & styling
        self.ax.tick_params(colors="#94A3B8", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#334155")

        # 1. Draw Obstacles
        for obs in state.obstacles:
            self._draw_obstacle(obs)

        # 2. Draw Shooter
        shooter_circle = Circle(
            (state.shooter.position.x, state.shooter.position.y),
            state.shooter.radius,
            facecolor="#38BDF8",  # Cyan/Blue
            edgecolor="#FFFFFF",
            linewidth=1.5,
            zorder=5,
            label=f"Shooter ({state.shooter.name})"
        )
        self.ax.add_patch(shooter_circle)
        self.ax.text(
            state.shooter.position.x,
            state.shooter.position.y + state.shooter.radius + 0.5,
            state.shooter.name,
            color="#38BDF8",
            fontsize=9,
            fontweight="bold",
            ha="center",
            zorder=6
        )

        # 3. Draw Targets
        for target in state.targets:
            color = "#EF4444" if target.is_alive else "#6B7280"
            alpha = 1.0 if target.is_alive else 0.4
            target_circle = Circle(
                (target.position.x, target.position.y),
                target.radius,
                facecolor=color,
                edgecolor="#FCA5A5" if target.is_alive else "#9CA3AF",
                linewidth=1.8,
                alpha=alpha,
                zorder=5,
            )
            self.ax.add_patch(target_circle)
            # Bullseye ring
            inner_ring = Circle(
                (target.position.x, target.position.y),
                target.radius * 0.4,
                facecolor="#FFFFFF",
                alpha=alpha,
                zorder=6,
            )
            self.ax.add_patch(inner_ring)
            self.ax.text(
                target.position.x,
                target.position.y + target.radius + 0.5,
                f"{target.id} {'[ALIVE]' if target.is_alive else '[HIT]'}",
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="center",
                zorder=6
            )

        # 4. Draw Trajectory History (Faded)
        for old_result in state.history[:-1]:
            if old_result.trajectory_points:
                xs = [p.x for p in old_result.trajectory_points]
                ys = [p.y for p in old_result.trajectory_points]
                self.ax.plot(xs, ys, color="#64748B", linestyle=":", linewidth=1.0, alpha=0.4, zorder=3)

        # 5. Draw Active / Last Trajectory
        active_res = current_result or (state.history[-1] if state.history else None)
        if active_res and active_res.trajectory_points:
            xs = [p.x for p in active_res.trajectory_points]
            ys = [p.y for p in active_res.trajectory_points]
            
            traj_color = "#22C55E" if active_res.is_success else "#F59E0B"
            self.ax.plot(xs, ys, color=traj_color, linewidth=2.2, alpha=0.9, zorder=4, label=f"y = {active_res.formula_str}")

            # Impact point marker
            if active_res.hit_coordinate:
                marker_color = "#22C55E" if active_res.is_success else "#EF4444"
                marker_symbol = "*" if active_res.is_success else "X"
                self.ax.plot(
                    active_res.hit_coordinate.x,
                    active_res.hit_coordinate.y,
                    marker=marker_symbol,
                    markersize=12,
                    markeredgecolor="white",
                    markerfacecolor=marker_color,
                    markeredgewidth=1.5,
                    zorder=10
                )

    @classmethod
    def save_snapshot(
        cls,
        state: GameState,
        filepath: str,
        current_result: Optional[TrajectoryResult] = None,
        figsize=(8, 5),
    ):
        """Save high-resolution snapshot to disk."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
        fig.patch.set_facecolor("#0F172A")
        renderer = cls(ax, state.bounds)
        renderer.render(state, current_result)
        fig.tight_layout()
        fig.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    def _draw_obstacle(self, obs: BaseObstacle):
        if isinstance(obs, CircleObstacle):
            patch = Circle(
                (obs.center.x, obs.center.y),
                obs.radius,
                facecolor="#334155",
                edgecolor="#64748B",
                linewidth=1.2,
                zorder=2
            )
            self.ax.add_patch(patch)
        elif isinstance(obs, BoxObstacle):
            width = obs.x_max - obs.x_min
            height = obs.y_max - obs.y_min
            patch = Rectangle(
                (obs.x_min, obs.y_min),
                width,
                height,
                facecolor="#334155",
                edgecolor="#64748B",
                linewidth=1.2,
                zorder=2
            )
            self.ax.add_patch(patch)
        elif isinstance(obs, PolygonObstacle):
            xy = [[v.x, v.y] for v in obs.vertices]
            patch = Polygon(
                xy,
                closed=True,
                facecolor="#334155",
                edgecolor="#64748B",
                linewidth=1.2,
                zorder=2
            )
            self.ax.add_patch(patch)
        elif isinstance(obs, TerrainObstacle):
            xs = [p.x for p in obs.points]
            ys = [p.y for p in obs.points]
            # Extend polygon down to y_min
            poly_xs = [xs[0]] + xs + [xs[-1]]
            poly_ys = [self.bounds.y_min - 5.0] + ys + [self.bounds.y_min - 5.0]
            xy = list(zip(poly_xs, poly_ys))
            patch = Polygon(
                xy,
                closed=True,
                facecolor="#1E293B",
                edgecolor="#475569",
                linewidth=1.5,
                zorder=2
            )
            self.ax.add_patch(patch)
