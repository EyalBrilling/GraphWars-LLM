"""
GraphWars-LLM Video Generator Tool

Renders a standalone animation (GIF/MP4) showing a projectile traversing
attempt trajectories sequentially from an output test folder.

Usage:
    python tools/video_generator.py outputs/scenario_5_random_gauntlet_20261010_135647 --speed 30.0 --fps 30 --output video.gif
"""

import os
import sys
import json
import argparse
import textwrap
from typing import Any, Dict, List, Optional

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
from src.engine.ui.icons import draw_goal_icon, draw_hit_area, draw_start_icon, icon_size, model_label_for_run


BG = "#0F172A"
MUTED = "#94A3B8"
TEXT = "#F8FAFC"
ACCENT = "#38BDF8"
SUCCESS = "#22C55E"
FAIL = "#EF4444"
FORMULA_WRAP = 110  # characters per line for the formula in the header


def draw_arena(
    ax: plt.Axes,
    bounds: CoordinateBounds,
    shooter: Shooter,
    targets: List[Target],
    obstacles: List[Any],
    shooter_label: Optional[str] = None,
):
    """Draw the static arena environment (background, grid, obstacles, shooter, targets)."""
    ax.clear()
    ax.set_facecolor(BG)

    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.15, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    for obs in obstacles:
        if isinstance(obs, BoxObstacle):
            ax.add_patch(patches.Rectangle(
                (obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                linewidth=1.5, edgecolor="#64748B", facecolor="#334155", alpha=0.85, zorder=2,
            ))
        elif isinstance(obs, CircleObstacle):
            ax.add_patch(patches.Circle(
                (obs.center.x, obs.center.y), obs.radius,
                linewidth=1.5, edgecolor="#64748B", facecolor="#334155", alpha=0.85, zorder=2,
            ))
        elif isinstance(obs, PolygonObstacle):
            ax.add_patch(patches.Polygon(
                [[v.x, v.y] for v in obs.vertices], closed=True,
                linewidth=1.5, edgecolor="#64748B", facecolor="#334155", alpha=0.85, zorder=2,
            ))
        elif isinstance(obs, TerrainObstacle):
            xs = [p.x for p in obs.points]
            ys = [p.y for p in obs.points]
            floor_y = bounds.y_min - 5.0
            ax.add_patch(patches.Polygon(
                list(zip([xs[0]] + xs + [xs[-1]], [floor_y] + ys + [floor_y])), closed=True,
                linewidth=1.5, edgecolor="#475569", facecolor="#1E293B", zorder=2,
            ))

    # Start: robot emoji labeled with the model; dashed circle = real start radius
    sx, sy = shooter.position.x, shooter.position.y
    s_size = icon_size(bounds, shooter.radius)
    draw_hit_area(ax, sx, sy, shooter.radius, ACCENT)
    draw_start_icon(ax, sx, sy, s_size, zorder=6)
    ax.text(sx, sy - 0.6 * s_size, shooter_label or shooter.name,
            color=ACCENT, fontsize=9, fontweight="bold", ha="center", va="top", zorder=12)

    # Goal: the cake; dashed circle = real hit radius
    for t in targets:
        tx, ty = t.position.x, t.position.y
        t_size = icon_size(bounds, t.radius)
        draw_hit_area(ax, tx, ty, t.radius, SUCCESS)
        draw_goal_icon(ax, tx, ty, t_size, zorder=6)
        ax.text(tx, ty - 0.6 * t_size, "Cake",
                color=SUCCESS, fontsize=9, fontweight="bold", ha="center", va="top", zorder=12)


def draw_header(fig: plt.Figure, attempt_num: int, total: int, strategy: str, formula: str,
                outcome_text: Optional[str] = None, outcome_color: str = TEXT) -> List[Any]:
    """Header above the plot: attempt, strategy, wrapped formula and (at impact) the outcome."""
    artists = [
        fig.text(0.04, 0.955, f"ATTEMPT {attempt_num} / {total}   ·   {strategy.upper()}",
                 color=TEXT, fontsize=12, fontweight="bold", va="top"),
    ]
    wrapped = textwrap.fill(f"y = {formula}", FORMULA_WRAP, subsequent_indent="    ")
    artists.append(fig.text(0.04, 0.915, wrapped, color=ACCENT, fontsize=9.5,
                            fontfamily="monospace", va="top", linespacing=1.4))
    if outcome_text:
        artists.append(fig.text(
            0.96, 0.955, outcome_text, color=outcome_color, fontsize=13, fontweight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1E293B", edgecolor=outcome_color),
        ))
    return artists


def capture(fig: plt.Figure) -> Image.Image:
    """Render the figure and return an independent RGB copy of the pixels.

    np.array (not np.asarray) is essential: buffer_rgba() is a view into the canvas
    buffer, so without a copy every captured frame aliases the last rendered one.
    """
    fig.canvas.draw()
    return Image.fromarray(np.array(fig.canvas.buffer_rgba())).convert("RGB")


def frame_end_indices(x_vals: List[float], dx_per_frame: float) -> List[int]:
    """Pick trajectory indices so the projectile advances a constant x-distance per frame."""
    dist = np.abs(np.asarray(x_vals) - x_vals[0])
    total = float(dist[-1])
    n_frames = max(1, int(np.ceil(total / dx_per_frame)))
    idx = [int(np.searchsorted(dist, k * dx_per_frame, side="right") - 1) for k in range(1, n_frames + 1)]
    idx = sorted(set(max(0, i) for i in idx))
    if not idx or idx[-1] != len(x_vals) - 1:
        idx.append(len(x_vals) - 1)
    return idx


def generate_trajectory_video(
    output_dir: str,
    speed: float = 30.0,
    fps: int = 30,
    output_file: Optional[str] = None,
    pause_sec: float = 1.5,
):
    """
    Generate an animated GIF/MP4 video from a GraphWars output folder.

    Args:
        output_dir: Path to directory containing scenario.json and history.json
        speed: Projectile speed along the x-axis in coordinate units per second
        fps: Frames per second of the output video
        output_file: Target file path (.gif or .mp4)
        pause_sec: Seconds to hold on the impact frame before the next attempt
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

    # Same loader as the runner, so rules and obstacles match
    state = GameState.from_dict(initial_state)
    bounds, shooter, targets, obstacles = state.bounds, state.shooter, state.targets, state.obstacles
    shooter_label = model_label_for_run(output_dir)

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=100)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.06, top=0.80)

    frames: List[Image.Image] = []
    dx_per_frame = speed / float(fps)
    pause_frames = max(1, int(pause_sec * fps))
    total_attempts = len(attempts)

    print("🎬 GraphWars Video Generator")
    print(f"Output Directory: {output_dir}")
    print(f"Attempts to Render: {total_attempts}")
    print(f"Speed: {speed} units/sec | FPS: {fps} (dx/frame: {dx_per_frame:.3f})")

    for attempt_entry in attempts:
        attempt_num = attempt_entry["attempt"]
        llm_out = attempt_entry.get("llm_output", {})
        sim_res = attempt_entry.get("simulation_result", {})

        formula_str = llm_out.get("formula") or ""
        strategy = llm_out.get("strategy") or ("no answer" if sim_res.get("hit_type") == "no_answer" else "N/A")
        waypoints = llm_out.get("planned_waypoints", []) or []
        wx = [w["x"] for w in waypoints if "x" in w and "y" in w]
        wy = [w["y"] for w in waypoints if "x" in w and "y" in w]

        traj = state.evaluate_formula(formula_str, step_size=0.05)
        pts = traj.trajectory_points or [shooter.position]
        x_vals = [p.x for p in pts]
        y_vals = [p.y for p in pts]

        # Animate the projectile at constant x-speed
        for idx_end in frame_end_indices(x_vals, dx_per_frame):
            draw_arena(ax, bounds, shooter, targets, obstacles, shooter_label)
            if wx:
                ax.plot(wx, wy, "o--", color="#F59E0B", alpha=0.45, linewidth=1, markersize=4, zorder=3)
            ax.plot(x_vals[: idx_end + 1], y_vals[: idx_end + 1], color=ACCENT, linewidth=2.5, alpha=0.95, zorder=4)
            ax.plot(x_vals[idx_end], y_vals[idx_end], "o", color="#F43F5E", markersize=10, zorder=6)
            ax.plot(x_vals[idx_end], y_vals[idx_end], "o", color="#FFFFFF", markersize=5, zorder=7)
            header = draw_header(fig, attempt_num, total_attempts, strategy, formula_str)
            frames.append(capture(fig))
            for a in header:
                a.remove()

        # Impact: hold on the full trajectory with the outcome in the header
        is_success = sim_res.get("is_success", False)
        hit_obstacle = sim_res.get("hit_obstacle")
        if is_success:
            outcome_text, outcome_color, marker, msize = "TARGET HIT", SUCCESS, "*", 22
        else:
            label = hit_obstacle or sim_res.get("hit_type", "miss").replace("_", " ")
            outcome_text, outcome_color, marker, msize = f"MISS · {label}", FAIL, "X", 14

        draw_arena(ax, bounds, shooter, targets, obstacles, shooter_label)
        if wx:
            ax.plot(wx, wy, "o--", color="#F59E0B", alpha=0.35, linewidth=1, markersize=4, zorder=3)
        ax.plot(x_vals, y_vals, color=outcome_color if is_success else ACCENT, linewidth=2.8, alpha=0.95, zorder=4)
        if not is_success:  # on a hit the path ending in the cake says it; a star would cover the icon
            ax.plot(x_vals[-1], y_vals[-1], marker, color=outcome_color, markersize=msize,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=8)
        header = draw_header(fig, attempt_num, total_attempts, strategy, formula_str, outcome_text, outcome_color)
        hold = capture(fig)
        for a in header:
            a.remove()
        frames.extend([hold] * pause_frames)

    plt.close(fig)

    if not output_file:
        output_file = os.path.join(output_dir, "trajectory_video.gif")

    print(f"💾 Saving animation ({len(frames)} frames, {len(frames) / fps:.1f}s) to: {output_file}")

    def save_gif(path: str) -> None:
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(round(1000 / fps)), loop=0, optimize=False)

    if output_file.lower().endswith(".gif"):
        save_gif(output_file)
    else:
        try:
            import imageio
            imageio.mimwrite(output_file, [np.array(f) for f in frames], fps=fps)
        except Exception as e:
            print(f"Warning: Failed to export video using imageio ({e}). Saving as GIF instead.")
            output_file = output_file.rsplit(".", 1)[0] + ".gif"
            save_gif(output_file)

    print(f"✨ Video generation complete! File saved: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="GraphWars Standalone Trajectory Video Generator Tool.")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Path to output test folder (containing scenario.json and history.json)")
    parser.add_argument("--output-dir", "-d", default=None, help="Alternative argument for output test folder")
    parser.add_argument("--speed", "-v", type=float, default=30.0,
                        help="Projectile speed along x-axis in units per second (default: 30.0)")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate FPS (default: 30)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output video file path (.gif or .mp4). Default: trajectory_video.gif inside output folder")

    args = parser.parse_args()
    target_dir = args.folder or args.output_dir
    if not target_dir:
        print("Error: Please provide an output test directory path.", file=sys.stderr)
        print("Example: python tools/video_generator.py outputs/scenario_5_random_gauntlet_20261010_135647", file=sys.stderr)
        sys.exit(1)

    try:
        generate_trajectory_video(output_dir=target_dir, speed=args.speed, fps=args.fps, output_file=args.output)
    except Exception as e:
        print(f"Error generating video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
