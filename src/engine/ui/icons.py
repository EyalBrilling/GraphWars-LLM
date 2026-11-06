"""
Start / goal icons shared by the UI renderer and the video tool.

- Start: the robot emoji, rendered to an image with the Windows color emoji font
  (matplotlib text can't draw color emoji). Falls back to a drawn robot.
- Goal: assets/portal_cake.png if present (user-supplied image). Falls back to a drawn cake.

All sizes are in data units; the axes must use an equal aspect ratio and fixed limits.
"""

import os
import json
from functools import lru_cache
from typing import Optional

import numpy as np
import matplotlib.patches as patches
from matplotlib.axes import Axes

from src.engine.backend.space import CoordinateBounds

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GOAL_IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets", "portal_cake.png")
EMOJI_FONT_PATHS = [
    r"C:\Windows\Fonts\seguiemj.ttf",                         # Windows
    "/System/Library/Fonts/Apple Color Emoji.ttc",            # macOS
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",      # Linux
]
ROBOT_EMOJI = "\U0001F916"

CORAL = "#D97757"
ROBOT_BODY = "#E2E8F0"
ROBOT_EDGE = "#475569"
ROBOT_EYE = "#38BDF8"


def icon_size(bounds: CoordinateBounds, radius: float) -> float:
    """Icon height: readable at any arena scale, never smaller than the object's hit area."""
    return max(2.6 * radius, 0.05 * bounds.width)


def _rounded(x0, y0, w, h, r, **kw) -> patches.FancyBboxPatch:
    return patches.FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw)


def draw_robot(ax: Axes, x: float, y: float, size: float, zorder: int = 6) -> None:
    """A friendly robot head centered at (x, y), `size` tall (antenna included)."""
    s = size
    head_w, head_h = 0.9 * s, 0.68 * s
    hx0, hy0 = x - head_w / 2, y - 0.42 * s
    z = zorder

    # Ears
    ear_w, ear_h = 0.1 * s, 0.26 * s
    for ex in (hx0 - ear_w * 0.8, hx0 + head_w - ear_w * 0.2):
        ax.add_patch(_rounded(ex, hy0 + head_h * 0.3, ear_w, ear_h, 0.03 * s,
                              facecolor=CORAL, edgecolor=ROBOT_EDGE, linewidth=1, zorder=z))
    # Antenna
    ax.plot([x, x], [hy0 + head_h, hy0 + head_h + 0.16 * s], color=ROBOT_EDGE, linewidth=1.5, zorder=z)
    ax.add_patch(patches.Circle((x, hy0 + head_h + 0.2 * s), 0.07 * s,
                                facecolor=CORAL, edgecolor=ROBOT_EDGE, linewidth=1, zorder=z + 1))
    # Head
    ax.add_patch(_rounded(hx0, hy0, head_w, head_h, 0.16 * s,
                          facecolor=ROBOT_BODY, edgecolor=ROBOT_EDGE, linewidth=1.5, zorder=z + 1))
    # Face plate
    fw, fh = head_w * 0.76, head_h * 0.58
    ax.add_patch(_rounded(x - fw / 2, hy0 + head_h * 0.22, fw, fh, 0.1 * s,
                          facecolor="#1E293B", edgecolor="none", zorder=z + 2))
    # Eyes
    eye_y = hy0 + head_h * 0.56
    for ex in (x - 0.17 * s, x + 0.17 * s):
        ax.add_patch(patches.Circle((ex, eye_y), 0.075 * s, facecolor=ROBOT_EYE, edgecolor="none", zorder=z + 3))
    # Smile
    ax.add_patch(patches.Arc((x, hy0 + head_h * 0.42), 0.26 * s, 0.14 * s, theta1=200, theta2=340,
                             color=CORAL, linewidth=1.8, zorder=z + 3))


def draw_cake(ax: Axes, x: float, y: float, size: float, zorder: int = 6) -> None:
    """A chocolate cake with cream, cherries and a lit candle, centered at (x, y), `size` tall."""
    s = size
    z = zorder
    body_w, body_h = 0.92 * s, 0.42 * s
    bx0, by0 = x - body_w / 2, y - 0.4 * s

    # Plate
    ax.add_patch(patches.Ellipse((x, by0), body_w * 1.25, 0.1 * s,
                                 facecolor="#CBD5E1", edgecolor="#94A3B8", linewidth=1, zorder=z))
    # Chocolate layers with a cream filling stripe
    ax.add_patch(_rounded(bx0, by0, body_w, body_h, 0.05 * s,
                          facecolor="#4A2C1F", edgecolor="#2B1810", linewidth=1.2, zorder=z + 1))
    ax.add_patch(patches.Rectangle((bx0, by0 + body_h * 0.42), body_w, body_h * 0.14,
                                   facecolor="#F5E6D3", edgecolor="none", zorder=z + 2))
    # Cream top with drips
    top_y = by0 + body_h
    ax.add_patch(_rounded(bx0 - 0.02 * s, top_y - 0.07 * s, body_w + 0.04 * s, 0.12 * s, 0.05 * s,
                          facecolor="#FFF7ED", edgecolor="#E7D7C1", linewidth=1, zorder=z + 3))
    for dx in (-0.3, -0.05, 0.22):
        ax.add_patch(patches.Ellipse((x + dx * s, top_y - 0.08 * s), 0.08 * s, 0.1 * s,
                                     facecolor="#FFF7ED", edgecolor="none", zorder=z + 3))
    # Cherries
    for dx in (-0.3, -0.1, 0.1, 0.3):
        if abs(dx) < 0.05:
            continue
        ax.add_patch(patches.Circle((x + dx * s, top_y + 0.07 * s), 0.055 * s,
                                    facecolor="#DC2626", edgecolor="#7F1D1D", linewidth=0.8, zorder=z + 4))
    # Candle and flame
    cw, ch = 0.07 * s, 0.26 * s
    ax.add_patch(patches.Rectangle((x - cw / 2, top_y + 0.02 * s), cw, ch,
                                   facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=0.8, zorder=z + 4))
    for i in range(3):
        ax.add_patch(patches.Rectangle((x - cw / 2, top_y + 0.06 * s + i * 0.08 * s), cw, 0.03 * s,
                                       facecolor="#EF4444", edgecolor="none", zorder=z + 5))
    flame_y = top_y + 0.02 * s + ch + 0.07 * s
    ax.add_patch(patches.Ellipse((x, flame_y), 0.1 * s, 0.17 * s, facecolor="#F59E0B", edgecolor="none", zorder=z + 5))
    ax.add_patch(patches.Ellipse((x, flame_y - 0.02 * s), 0.05 * s, 0.09 * s, facecolor="#FDE68A", edgecolor="none", zorder=z + 6))


@lru_cache(maxsize=None)
def _emoji_image(char: str) -> Optional[np.ndarray]:
    """Render a color emoji to a tightly cropped RGBA array, or None if no emoji font works."""
    from PIL import Image, ImageDraw, ImageFont
    for font_path in EMOJI_FONT_PATHS:
        if not os.path.exists(font_path):
            continue
        try:
            # Color bitmap fonts (Apple/Noto) only load at their native size of 109/137 px
            font = ImageFont.truetype(font_path, 109)
            img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((100, 100), char, font=font, embedded_color=True, anchor="mm")
            bbox = img.getbbox()
            if bbox:
                return np.array(img.crop(bbox))
        except Exception:
            continue
    return None


@lru_cache(maxsize=None)
def _file_image(path: str, mtime: float) -> Optional[np.ndarray]:
    """Load an image file as RGBA (cache keyed on mtime so a replaced file is picked up)."""
    from PIL import Image
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    bbox = img.getbbox()  # trim fully transparent margins
    return np.array(img.crop(bbox) if bbox else img)


def _draw_image(ax: Axes, img: np.ndarray, x: float, y: float, size: float, zorder: int) -> None:
    """Draw an RGBA image centered at (x, y), `size` tall in data units, keeping its aspect ratio."""
    h, w = img.shape[:2]
    width = size * w / h
    ax.imshow(img, extent=(x - width / 2, x + width / 2, y - size / 2, y + size / 2),
              zorder=zorder, interpolation="antialiased")


def draw_start_icon(ax: Axes, x: float, y: float, size: float, zorder: int = 6) -> None:
    img = _emoji_image(ROBOT_EMOJI)
    if img is not None:
        _draw_image(ax, img, x, y, size, zorder)
    else:
        draw_robot(ax, x, y, size, zorder)


def draw_goal_icon(ax: Axes, x: float, y: float, size: float, zorder: int = 6) -> None:
    img = None
    if os.path.exists(GOAL_IMAGE_PATH):
        img = _file_image(GOAL_IMAGE_PATH, os.path.getmtime(GOAL_IMAGE_PATH))
    if img is not None:
        _draw_image(ax, img, x, y, size, zorder)
    else:
        draw_cake(ax, x, y, size, zorder)


def draw_hit_area(ax: Axes, x: float, y: float, radius: float, color: str, zorder: int = 4) -> None:
    """Faint dashed circle showing the real collision radius under an icon."""
    ax.add_patch(patches.Circle((x, y), radius, fill=False, edgecolor=color, linewidth=1,
                                linestyle=(0, (3, 3)), alpha=0.6, zorder=zorder))


def model_label_for_run(folder: Optional[str]) -> Optional[str]:
    """Model name used in a run, read from attempt_1_raw_request.json if present (e.g. 'claude-sonnet-5-5')."""
    if not folder:
        return None
    path = os.path.join(folder, "attempt_1_raw_request.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            req = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(req, dict):
        if isinstance(req.get("model"), str):
            return req["model"]
        if req.get("provider") == "mock":
            return "mock agent"
    return None
