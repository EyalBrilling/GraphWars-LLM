"""
GraphWars-LLM Video Generator Tool

Renders a standalone animation (GIF/MP4) showing a projectile/robot traversing
attempt trajectories sequentially from an output test folder.

Usage:
    python tools/video_generator.py outputs/scenario_5_random_gauntlet_20261010_135647 --speed 30.0 --fps 30 --output video.gif
"""

import os
import sys
import json
import argparse
import math
from typing import Dict, Any, List, Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine.backend.space import CoordinateBounds, Shooter, Target
from src.engine.backend.obstacles import BoxObstacle, CircleObstacle, PolygonObstacle, TerrainObstacle
from src.engine.backend.game_state import GameState


def draw_arena(
    ax: plt.Axes,
    bounds: CoordinateBounds,
    shooter: Shooter,
    targets: List[Target],
    obstacles: List[Any],
):
    """Draw the static arena environment (background, grid, obstacles, shooter, targets)."""
    ax.clear()
    ax.set_facecolor("#0F172A")  # Slate 900 background

    # Title & Bounds
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")

    # Draw Obstacles
    for obs in obstacles:
        if isinstance(obs, BoxObstacle):
            w = obs.x_max - obs.x_min
            h = obs.y_max - obs.y_min
            rect = patches.Rectangle(
                (obs.x_min, obs.y_min),
                w,
                h,
                linewidth=1.5,
                edgecolor="#64748B",
                facecolor="#334155",
                alpha=0.85,
                zorder=2,
            )
            ax.add_patch(rect)
            # Label
            ax.text(
                (obs.x_min + obs.x_max) / 2.0,
                (obs.y_min + obs.y_max) / 2.0,
                obs.name,
                color="#94A3B8",
                fontsize=7,
                ha="center",
                va="center",
                zorder=3,
                clip_on=True,
            )
        elif isinstance(obs, CircleObstacle):
            circle = patches.Circle(
                (obs.center.x, obs.center.y),
                obs.radius,
                linewidth=1.5,
                edgecolor="#64748B",
                facecolor="#334155",
                alpha=0.85,
                zorder=2,
            )
            ax.add_patch(circle)
            ax.text(
                obs.center.x,
                obs.center.y,
                obs.name,
                color="#94A3B8",
                fontsize=7,
                ha="center",
                va="center",
                zorder=3,
                clip_on=True,
            )
        elif isinstance(obs, PolygonObstacle):
            ax.add_patch(patches.Polygon(
                [[v.x, v.y] for v in obs.vertices],
                closed=True,
                linewidth=1.5,
                edgecolor="#64748B",
                facecolor="#334155",
                alpha=0.85,
                zorder=2,
            ))
        elif isinstance(obs, TerrainObstacle):
            xs = [p.x for p in obs.points]
            ys = [p.y for p in obs.points]
            floor_y = bounds.y_min - 5.0
            ax.add_patch(patches.Polygon(
                list(zip([xs[0]] + xs + [xs[-1]], [floor_y] + ys + [floor_y])),
                closed=True,
                linewidth=1.5,
                edgecolor="#475569",
                facecolor="#1E293B",
                zorder=2,
            ))

    # Draw Shooter
    shooter_circle = patches.Circle(
        (shooter.position.x, shooter.position.y),
        shooter.radius,
        linewidth=2,
        edgecolor="#38BDF8",
        facecolor="#0284C7",
        zorder=5,
    )
    ax.add_patch(shooter_circle)
    ax.text(
        shooter.position.x,
        shooter.position.y - shooter.radius - 1.2,
        shooter.name,
        color="#38BDF8",
        fontsize=8,
        fontweight="bold",
        ha="center",
        zorder=5,
    )

    # Draw Targets
    for t in targets:
        color = "#22C55E" if t.is_alive else "#EF4444"
        target_circle = patches.Circle(
            (t.position.x, t.position.y),
            t.radius,
            linewidth=2,
            edgecolor=color,
            facecolor="#15803D" if t.is_alive else "#991B1B",
            zorder=5,
        )
        ax.add_patch(target_circle)
        ax.text(
            t.position.x,
            t.position.y - t.radius - 1.2,
            f"Target: {t.id}",
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            zorder=5,
        )


def generate_trajectory_video(
    output_dir: str,
    speed: float = 30.0,
    fps: int = 30,
    output_file: Optional[str] = None,
    pause_sec: float = 1.0,
):
    """
    Generate an animated GIF/MP4 video from a GraphWars output folder.

    Args:
        output_dir: Path to directory containing scenario.json and history.json
        speed: Robot speed along the x-axis in coordinate units per second
        fps: Frames per second of the output video
        output_file: Target file path (.gif or .mp4)
        pause_sec: Seconds to pause at impact before next attempt
    """
    history_path = os.path.join(output_dir, "history.json")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Could not find history.json in: {output_dir}")

    with open(history_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    initial_state = session_data.get("initial_state", {})
    attempts = session_data.get("history", [])

    if not attempts:
        raise ValueError("No attempt history found in history.json!")

    # Reconstruct Environment (same loader as the runner, so rules and obstacles match)
    state = GameState.from_dict(initial_state)
    bounds, shooter, targets, obstacles = state.bounds, state.shooter, state.targets, state.obstacles

    # Setup Plot Figure
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor("#0F172A")

    frames: List[Image.Image] = []
    dx_per_frame = speed / float(fps)
    pause_frames = int(pause_sec * fps)

    print(f"🎬 GraphWars Video Generator")
    print(f"Output Directory: {output_dir}")
    print(f"Attempts to Render: {len(attempts)}")
    print(f"Robot Speed: {speed} units/sec | FPS: {fps} (dx/frame: {dx_per_frame:.3f})")

    total_attempts_count = len(attempts)

    for attempt_entry in attempts:
        attempt_num = attempt_entry["attempt"]
        llm_out = attempt_entry.get("llm_output", {})
        sim_res = attempt_entry.get("simulation_result", {})

        formula_str = llm_out.get("formula", "0")
        strategy = llm_out.get("strategy", "N/A")
        reasoning = llm_out.get("reasoning", "")
        waypoints = llm_out.get("planned_waypoints", [])

        # Evaluate continuous physics trajectory (non-mutating, with the scenario's constraints)
        traj_result = state.evaluate_formula(formula_str, step_size=0.05)

        pts = traj_result.trajectory_points
        if not pts:
            pts = [shooter.position]

        x_vals = [p.x for p in pts]
        y_vals = [p.y for p in pts]

        # Calculate animation steps
        total_pts = len(pts)
        step_increment = max(1, int(dx_per_frame / 0.05))
        indices = list(range(0, total_pts, step_increment))
        if indices[-1] != total_pts - 1:
            indices.append(total_pts - 1)

        # Animate trajectory traversal
        for idx_step, idx_end in enumerate(indices):
            draw_arena(ax, bounds, shooter, targets, obstacles)

            curr_x = x_vals[: idx_end + 1]
            curr_y = y_vals[: idx_end + 1]

            # 1. Planned Waypoints (if available)
            if waypoints:
                wx = [w["x"] for w in waypoints]
                wy = [w["y"] for w in waypoints]
                ax.plot(wx, wy, "o--", color="#F59E0B", alpha=0.5, linewidth=1, markersize=4, label="Waypoints")

            # 2. Trajectory Line Trail
            ax.plot(curr_x, curr_y, color="#38BDF8", linewidth=2.5, alpha=0.9, zorder=4)

            # 3. Robot Node (Traveling Marker)
            robot_x = curr_x[-1]
            robot_y = curr_y[-1]

            # Glowing robot marker
            ax.plot(robot_x, robot_y, "o", color="#F43F5E", markersize=10, zorder=6)
            ax.plot(robot_x, robot_y, "o", color="#FFFFFF", markersize=5, zorder=7)

            # 4. HUD Banner Overlay
            status_color = "#38BDF8"
            ax.text(
                0.02,
                0.95,
                f"ATTEMPT {attempt_num} / {total_attempts_count}  [{strategy.upper()}]",
                transform=ax.transAxes,
                color="#F8FAFC",
                fontsize=11,
                fontweight="bold",
                va="top",
            )
            ax.text(
                0.02,
                0.90,
                f"Formula: y = {formula_str}",
                transform=ax.transAxes,
                color=status_color,
                fontsize=10,
                fontfamily="monospace",
                va="top",
            )

            if reasoning:
                short_reasoning = reasoning[:85] + ("..." if len(reasoning) > 85 else "")
                ax.text(
                    0.02,
                    0.85,
                    f"Plan: {short_reasoning}",
                    transform=ax.transAxes,
                    color="#94A3B8",
                    fontsize=8.5,
                    style="italic",
                    va="top",
                )

            # Convert figure to PIL Image
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            img = Image.fromarray(rgba)
            frames.append(img)

        # Handle Impact Frame Banner
        impact_x = x_vals[-1]
        impact_y = y_vals[-1]

        is_success = sim_res.get("is_success", False)
        hit_type = sim_res.get("hit_type", "unknown").upper()
        hit_obstacle = sim_res.get("hit_obstacle")

        # Build final pause frames with outcome text
        for _ in range(pause_frames):
            draw_arena(ax, bounds, shooter, targets, obstacles)
            ax.plot(x_vals, y_vals, color="#38BDF8", linewidth=2.5, alpha=0.9, zorder=4)

            if waypoints:
                wx = [w["x"] for w in waypoints]
                wy = [w["y"] for w in waypoints]
                ax.plot(wx, wy, "o--", color="#F59E0B", alpha=0.4, linewidth=1, markersize=4)

            if is_success:
                # Target Hit Celebration
                ax.plot(impact_x, impact_y, "*", color="#22C55E", markersize=20, zorder=8)
                banner_text = "TARGET DESTROYED!"
                banner_color = "#22C55E"
            else:
                # Collision / Out of bounds
                ax.plot(impact_x, impact_y, "X", color="#EF4444", markersize=14, zorder=8)
                if hit_obstacle:
                    banner_text = f"COLLISION WITH: {hit_obstacle}"
                else:
                    banner_text = f"SHOT OUTCOME: {hit_type}"
                banner_color = "#EF4444"

            # HUD Status Banner
            ax.text(
                0.02,
                0.95,
                f"ATTEMPT {attempt_num} / {total_attempts_count}  [{strategy.upper()}]",
                transform=ax.transAxes,
                color="#F8FAFC",
                fontsize=11,
                fontweight="bold",
                va="top",
            )
            ax.text(
                0.02,
                0.90,
                f"Formula: y = {formula_str}",
                transform=ax.transAxes,
                color="#38BDF8",
                fontsize=10,
                fontfamily="monospace",
                va="top",
            )
            ax.text(
                0.50,
                0.50,
                banner_text,
                transform=ax.transAxes,
                color=banner_color,
                fontsize=16,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor=banner_color, alpha=0.9),
                zorder=10,
            )

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            img = Image.fromarray(rgba)
            frames.append(img)

    plt.close(fig)

    # Determine Output File Path
    if not output_file:
        output_file = os.path.join(output_dir, "trajectory_video.gif")

    print(f"💾 Saving animation ({len(frames)} frames) to: {output_file}")
    
    if output_file.lower().endswith(".gif"):
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
    else:
        # Fallback / MP4 via Pillow or imageio/ffmpeg if requested
        try:
            import imageio
            imageio.mimwrite(output_file, [np.array(f) for f in frames], fps=fps)
        except Exception as e:
            print(f"Warning: Failed to export mp4 using imageio ({e}). Saving as GIF instead.")
            gif_file = output_file.rsplit(".", 1)[0] + ".gif"
            duration_ms = int(1000 / fps)
            frames[0].save(
                gif_file,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )
            output_file = gif_file

    print(f"✨ Video generation complete! File saved: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="GraphWars Standalone Trajectory Video Generator Tool.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Path to output test folder (containing scenario.json and history.json)",
    )
    parser.add_argument(
        "--output-dir", "-d",
        default=None,
        help="Alternative argument for output test folder",
    )
    parser.add_argument(
        "--speed", "-v",
        type=float,
        default=30.0,
        help="Robot travel speed along x-axis in units per second (default: 30.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate FPS (default: 30)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output video file path (.gif or .mp4). Default: trajectory_video.gif inside output folder",
    )

    args = parser.parse_args()
    target_dir = args.folder or args.output_dir

    if not target_dir:
        print("Error: Please provide an output test directory path.", file=sys.stderr)
        print("Example: python tools/video_generator.py outputs/scenario_5_random_gauntlet_20261010_135647", file=sys.stderr)
        sys.exit(1)

    try:
        generate_trajectory_video(
            output_dir=target_dir,
            speed=args.speed,
            fps=args.fps,
            output_file=args.output,
        )
    except Exception as e:
        print(f"Error generating video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
