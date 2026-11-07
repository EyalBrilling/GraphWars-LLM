"""
Stack two trajectory GIFs vertically into one (e.g. model A over model B on the same course).

Both GIFs play in sync on a shared timeline; the one that finishes first holds its last
frame, and the combined result holds the final frame for ~2 seconds before looping.
The output is portrait (good for LinkedIn / mobile feeds).

Usage:
    python tools/stack_gifs.py top.gif bottom.gif combined.gif --width 1080
"""

import argparse

from PIL import Image, ImageSequence

TICK_MS = 33          # output frame duration (~30 fps)
GAP_PX = 8            # white separator between the two halves
HOLD_TICKS = 60       # extra frames on the final combined frame (~2s)


def load_frames(path: str) -> list:
    """Load a GIF and expand each frame by its own duration onto a fixed TICK_MS timeline."""
    im = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(im):
        duration = frame.info.get("duration", TICK_MS) or TICK_MS
        rgb = frame.convert("RGB")
        frames.extend([rgb] * max(1, round(duration / TICK_MS)))
    return frames


def stack_gifs(top_path: str, bottom_path: str, out_path: str, width: int = 1080) -> None:
    top = load_frames(top_path)
    bottom = load_frames(bottom_path)
    total = max(len(top), len(bottom)) + HOLD_TICKS

    def fit(frame):
        return frame.resize((width, int(frame.height * width / frame.width)), Image.LANCZOS)

    out = []
    for i in range(total):
        a = fit(top[min(i, len(top) - 1)])
        b = fit(bottom[min(i, len(bottom) - 1)])
        canvas = Image.new("RGB", (width, a.height + GAP_PX + b.height), (255, 255, 255))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (0, a.height + GAP_PX))
        out.append(canvas.quantize(colors=128, method=Image.MEDIANCUT))

    out[0].save(out_path, save_all=True, append_images=out[1:], duration=TICK_MS, loop=0, optimize=True)
    print(f"Saved {out_path}: {out[0].size[0]}x{out[0].size[1]}, {len(out)} frames "
          f"({len(out) * TICK_MS / 1000:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Stack two GIFs vertically, synced in time.")
    parser.add_argument("top", help="GIF shown on top")
    parser.add_argument("bottom", help="GIF shown below")
    parser.add_argument("output", help="Output GIF path")
    parser.add_argument("--width", "-w", type=int, default=1080, help="Output width in px (default: 1080)")
    args = parser.parse_args()
    stack_gifs(args.top, args.bottom, args.output, args.width)


if __name__ == "__main__":
    main()
