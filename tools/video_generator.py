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
from src.engine.backend.physics import TrajectoryEvaluator
from src.engine.ui.icons import draw_goal_icon, draw_hit_area, draw_start_icon, icon_size, model_label_for_run


BG = "#0F172A"
MUTED = "#94A3B8"
TEXT = "#F8FAFC"
ACCENT = "#38BDF8"
SUCCESS = "#22C55E"
FAIL = "#EF4444"
FORMULA_WRAP = 110  # characters per line for the formula in the header

# Social layout: two stacked halves make a 1080x1350 (4:5) post. Each half is 10in wide with a
# 0.62in header strip on top of the plot; the plot shows y in SOCIAL_Y_RANGE (walls run past it,
# nothing happens there) so the arena fills the width.
SOCIAL_FIG_W, SOCIAL_FIG_H, SOCIAL_HEADER_IN = 10.0, 6.222, 0.622
SOCIAL_HEADER_Y = 1 - (SOCIAL_HEADER_IN / 2) / SOCIAL_FIG_H
SOCIAL_Y_RANGE = (-26.0, 30.0)


def draw_arena(
    ax: plt.Axes,
    bounds: CoordinateBounds,
    shooter: Shooter,
    targets: List[Target],
    obstacles: List[Any],
    shooter_label: Optional[str] = None,
    social: bool = False,
):
    """Draw the static arena environment (background, grid, obstacles, shooter, targets).

    social=True is the phone-readable style: no grid/ticks/labels, thicker edges, bigger icons.
    """
    ax.clear()
    ax.set_facecolor(BG)

    ax.set_xlim(bounds.x_min, bounds.x_max)
    if social:
        ax.set_ylim(max(bounds.y_min, SOCIAL_Y_RANGE[0]), min(bounds.y_max, SOCIAL_Y_RANGE[1]))
    else:
        ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_aspect("auto" if social else "equal")
    if social:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.grid(True, linestyle="--", alpha=0.15, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    edge_w = 2.5 if social else 1.5
    for obs in obstacles:
        if isinstance(obs, BoxObstacle):
            ax.add_patch(patches.Rectangle(
                (obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                linewidth=edge_w, edgecolor="#64748B", facecolor="#334155", alpha=0.85, zorder=2,
            ))
        elif isinstance(obs, CircleObstacle):
            ax.add_patch(patches.Circle(
                (obs.center.x, obs.center.y), obs.radius,
                linewidth=edge_w, edgecolor="#64748B", facecolor="#334155", alpha=0.85, zorder=2,
            ))
        elif isinstance(obs, PolygonObstacle):
            ax.add_patch(patches.Polygon(
                [[v.x, v.y] for v in obs.vertices], closed=True,
                linewidth=edge_w, edgecolor="#64748B", facecolor="#334155", alpha=0.85, zorder=2,
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
    icon_scale = 1.6 if social else 1.0
    sx, sy = shooter.position.x, shooter.position.y
    s_size = icon_size(bounds, shooter.radius) * icon_scale
    draw_hit_area(ax, sx, sy, shooter.radius, ACCENT)
    draw_start_icon(ax, sx, sy, s_size, zorder=6)
    if not social:  # in social mode the model name is in the header instead
        ax.text(sx, sy - 0.6 * s_size, shooter_label or shooter.name,
                color=ACCENT, fontsize=9, fontweight="bold", ha="center", va="top", zorder=12)

    # Goal: the cake; dashed circle = real hit radius
    for t in targets:
        tx, ty = t.position.x, t.position.y
        t_size = icon_size(bounds, t.radius) * icon_scale
        draw_hit_area(ax, tx, ty, t.radius, SUCCESS)
        draw_goal_icon(ax, tx, ty, t_size, zorder=6)
        if not social:
            ax.text(tx, ty - 0.6 * t_size, "Cake",
                    color=SUCCESS, fontsize=9, fontweight="bold", ha="center", va="top", zorder=12)


def pretty_model_name(model_id: str) -> str:
    """'claude-sonnet-5-5' -> 'CLAUDE SONNET 5.5' (falls back to the raw id)."""
    parts = (model_id or "").split("-")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return " ".join(parts[:-2]).upper() + f" {parts[-2]}.{parts[-1]}"
    return (model_id or "").upper()


def draw_header(fig: plt.Figure, attempt_num: int, total: int, model_title: str, strategy: str, formula: str,
                outcome_text: Optional[str] = None, outcome_color: str = TEXT,
                social: bool = False) -> List[Any]:
    """Header above the plot: attempt, wrapped formula and (at impact) the outcome.

    The model's self-reported strategy label is not shown (it can contradict the formula,
    e.g. "trigonometric_wave" on a sum of bumps); only "no answer" is surfaced.
    """
    title = f"ATTEMPT {attempt_num} / {total}"
    if model_title:
        title = f"{model_title}   ·   {title}"
    if strategy == "no answer":
        title += "   ·   NO ANSWER"
    if social:
        # Phone-readable: one header line (model + attempt) and a big outcome badge, no formula
        y = SOCIAL_HEADER_Y
        artists = [
            fig.text(0.02, y, f"{model_title}  ·  {attempt_num}/{total}", color=TEXT, fontsize=24,
                     fontweight="bold", va="center"),
        ]
        if outcome_text:
            artists.append(fig.text(
                0.98, y, outcome_text, color=outcome_color, fontsize=24, fontweight="bold",
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor=outcome_color, linewidth=3),
            ))
        return artists
    artists = [
        fig.text(0.04, 0.955, title,
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
    social: bool = False,
    anchor_invalid_start: bool = False,
):
    """
    Generate an animated GIF/MP4 video from a GraphWars output folder.

    Args:
        output_dir: Path to directory containing scenario.json and history.json
        speed: Projectile speed along the x-axis in coordinate units per second
        fps: Frames per second of the output video
        output_file: Target file path (.gif or .mp4)
        pause_sec: Seconds to hold on the impact frame before the next attempt
        social: Phone-readable style for social media (big header, no formula, thick lines)
        anchor_invalid_start: For invalid-start attempts, draw the curve shifted to start at the
            shooter (presentation only; the attempt is still shown as a miss)
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
    model_title = pretty_model_name(shooter_label)

    if social:
        fig = plt.figure(figsize=(SOCIAL_FIG_W, SOCIAL_FIG_H), dpi=100)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1 - SOCIAL_HEADER_IN / SOCIAL_FIG_H])
    else:
        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=100)
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.06, top=0.80)
    fig.patch.set_facecolor(BG)
    line_w = 6.0 if social else 2.5
    head_big, head_small = (20, 10) if social else (10, 5)

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
        if getattr(traj.hit_type, "value", traj.hit_type) == "invalid_start":
            raw = state.evaluate_formula(formula_str, step_size=0.05, enforce_start=False)
            if anchor_invalid_start and raw.trajectory_points:
                # Presentation option: shift the model's curve so it starts at the shooter
                # (same shape, constant offset removed). The outcome shown stays a miss.
                offset = raw.trajectory_points[0].y - shooter.position.y
                traj = TrajectoryEvaluator.evaluate(
                    formula_str=f"({formula_str})-({offset!r})", shooter=shooter, targets=targets,
                    obstacles=obstacles, bounds=bounds, step_size=0.05,
                    direction=1 if targets[0].position.x >= shooter.position.x else -1,
                    max_formula_characters=10**6,
                )
            else:
                # Show the curve the model actually wrote (it starts away from the shooter).
                traj = raw
        pts = traj.trajectory_points or [shooter.position]
        x_vals = [p.x for p in pts]
        y_vals = [p.y for p in pts]

        # Animate the projectile at constant x-speed
        for idx_end in frame_end_indices(x_vals, dx_per_frame):
            draw_arena(ax, bounds, shooter, targets, obstacles, shooter_label, social)
            if wx:  # the model's own planned waypoints
                ax.plot(wx, wy, "o--", color="#F59E0B", alpha=0.7 if social else 0.45,
                        linewidth=2.5 if social else 1, markersize=8 if social else 4, zorder=3)
            ax.plot(x_vals[: idx_end + 1], y_vals[: idx_end + 1], color=ACCENT, linewidth=line_w, alpha=0.95, zorder=4)
            ax.plot(x_vals[idx_end], y_vals[idx_end], "o", color="#F43F5E", markersize=head_big, zorder=6)
            ax.plot(x_vals[idx_end], y_vals[idx_end], "o", color="#FFFFFF", markersize=head_small, zorder=7)
            header = draw_header(fig, attempt_num, total_attempts, model_title, strategy, formula_str, social=social)
            frames.append(capture(fig))
            for a in header:
                a.remove()

        # Impact: hold on the full trajectory with the outcome in the header
        is_success = sim_res.get("is_success", False)
        hit_obstacle = sim_res.get("hit_obstacle")
        if is_success:
            outcome_text, outcome_color, marker, msize = "TARGET HIT", SUCCESS, "*", 22
        else:
            # Name the obstacle when there is one; other misses (e.g. invalid start) read as plain "MISS".
            outcome_text = f"MISS · {hit_obstacle}" if hit_obstacle and not social else "MISS"
            outcome_color, marker, msize = FAIL, "X", (30 if social else 14)

        draw_arena(ax, bounds, shooter, targets, obstacles, shooter_label, social)
        if wx:
            ax.plot(wx, wy, "o--", color="#F59E0B", alpha=0.6 if social else 0.35,
                    linewidth=2.5 if social else 1, markersize=8 if social else 4, zorder=3)
        ax.plot(x_vals, y_vals, color=outcome_color if is_success else ACCENT,
                linewidth=line_w + 0.5, alpha=0.95, zorder=4)
        if not is_success:  # on a hit the path ending in the cake says it; a star would cover the icon
            ax.plot(x_vals[-1], y_vals[-1], marker, color=outcome_color, markersize=msize,
                    markeredgecolor="white", markeredgewidth=2 if social else 1.2, zorder=8)
        header = draw_header(fig, attempt_num, total_attempts, model_title, strategy, formula_str,
                             outcome_text, outcome_color, social=social)
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
    parser.add_argument("--social", action="store_true",
                        help="Phone-readable style for social media: big header, no formula, thick lines")
    parser.add_argument("--anchor-invalid-start", action="store_true",
                        help="Draw invalid-start attempts shifted to start at the shooter (still shown as a miss)")

    args = parser.parse_args()
    target_dir = args.folder or args.output_dir
    if not target_dir:
        print("Error: Please provide an output test directory path.", file=sys.stderr)
        print("Example: python tools/video_generator.py outputs/scenario_5_random_gauntlet_20261010_135647", file=sys.stderr)
        sys.exit(1)

    try:
        generate_trajectory_video(output_dir=target_dir, speed=args.speed, fps=args.fps,
                                  output_file=args.output, social=args.social,
                                  anchor_invalid_start=args.anchor_invalid_start)
    except Exception as e:
        print(f"Error generating video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
