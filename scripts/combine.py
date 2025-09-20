#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
from PIL import Image


ITER_PATTERNS = [
    re.compile(r"step[_-]?(\d+)"),         # e.g., regression_step_000100.png
    re.compile(r"_(\d{4,})\."),            # fallback: ..._000100.png
]


def parse_iteration_from_name(path: str) -> int | None:
    """Return an integer iteration parsed from filename, or None if not found."""
    fname = os.path.basename(path)
    for pat in ITER_PATTERNS:
        m = pat.search(fname)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None  # e.g., "final.png"


def collect_images(input_dir: str, pattern: str):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise SystemExit(f"No files found in {input_dir!r} matching pattern {pattern!r}")

    # Sort primarily by parsed iteration (None goes last), secondarily by name.
    annotated = []
    for f in files:
        it = parse_iteration_from_name(f)
        annotated.append((f, it))
    annotated.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else float("inf"), x[0]))
    return annotated


def main():
    ap = argparse.ArgumentParser(description="Combine images into a grid with iteration labels.")
    ap.add_argument("--input_dir", type=str, required=True, help="Directory with images")
    ap.add_argument("--pattern", type=str, default="regression_step_*.png",
                    help="Glob pattern for images (default: regression_step_*.png)")
    ap.add_argument("--cols", type=int, default=5, help="Number of columns in grid (default: 5)")
    ap.add_argument("--cell_size", type=float, default=3.0,
                    help="Size (inches) for each subplot cell (default: 3.0)")
    ap.add_argument("--out", type=str, default="combined_grid.png", help="Output image path")
    args = ap.parse_args()

    items = collect_images(args.input_dir, args.pattern)
    n = len(items)
    cols = max(1, args.cols)
    rows = math.ceil(n / cols)

    # Create figure sized by grid
    fig_w = cols * args.cell_size
    fig_h = rows * args.cell_size
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # Fill grid
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n:
                path, iteration = items[idx]
                # Load via PIL to ensure compatibility, then to array for imshow
                img = Image.open(path).convert("RGBA")
                ax.imshow(img)
                ax.axis("off")
                # Title text
                if iteration is not None:
                    title = f"Meta Iteration Step: {iteration}"
                else:
                    # e.g., 'final.png' → strip extension for something readable
                    title = os.path.splitext(os.path.basename(path))[0]
                ax.set_title(title, fontsize=10)
                idx += 1
            else:
                ax.axis("off")

    plt.tight_layout()
    # Save high-res but reasonable size
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved grid to: {args.out}")


if __name__ == "__main__":
    main()

# python combine.py \
#   --input_dir experiments/mnist_20250920_1037_a11efc10/plots \
#   --pattern "regression_step_*.png" \
#   --cols 5 \
#   --out experiments/mnist_20250920_1037_a11efc10/plots/combined_grid.png
