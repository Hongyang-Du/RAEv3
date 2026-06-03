#!/usr/bin/env python3
"""
Build progression grids from overfit experiment outputs.

For each experiment (RAEv1 / RAEv2), produces:
  progression_v1.png  : rows=widths, cols=steps, cell = original|recon
  progression_v2.png  : same for RAEv2
  progression_combined.png : RAEv1 (top) and RAEv2 (bottom) stacked,
                              only at the widths both experiments share

Usage
-----
python make_progression_grid.py \
    --v1-dir RAE/output/overfit_results \
    --v2-dir RAEv2/output/overfit_results_v2 \
    --out-dir comparison_results
"""

import argparse
import os
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── helpers ──────────────────────────────────────────────────────────────────

def load_step_image(img_dir: str, hidden_size: int, step: int,
                    cell_w: int = 256) -> np.ndarray | None:
    """
    Load step_{step:06d}.png and return only the rightmost (first DiT recon) column.
    The saved panel is: original | RAE-recon | DiT-recon × N (each tile = H px wide).
    Tile width = H (images are square 256×256).
    """
    path = os.path.join(img_dir, f"d{hidden_size}", f"step_{step:06d}.png")
    if not os.path.exists(path):
        return None
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape
    tile = H  # each sub-image is square, so tile width = height

    # panel: original(tile) | RAE-recon(tile) | DiT-recon_1(tile) | ...
    # we want the 3rd column (index 2) = first DiT recon
    col = 2
    if W < (col + 1) * tile:
        col = W // tile - 1  # fallback: last available column
    recon = img[:, col * tile: (col + 1) * tile, :]
    return np.array(Image.fromarray(recon).resize((cell_w, cell_w), Image.LANCZOS))


def add_label(arr: np.ndarray, text: str, font_size: int = 14,
              bg: tuple = (30, 30, 30), fg: tuple = (255, 255, 255)) -> np.ndarray:
    """Paste a small text label onto the top-left of a numpy image array."""
    img = Image.fromarray(arr.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  font_size)
    except Exception:
        font = ImageFont.load_default()
    # background rect
    bbox = draw.textbbox((2, 2), text, font=font)
    draw.rectangle(bbox, fill=bg)
    draw.text((2, 2), text, fill=fg, font=font)
    return np.array(img)


def make_grid(rows: list[np.ndarray | None],
              row_labels: list[str],
              col_labels: list[str],
              cell_w: int,
              cell_h: int,
              label_col_w: int = 70,
              label_row_h: int = 24,
              gap: int = 2,
              placeholder_color: tuple = (60, 60, 60)) -> Image.Image:
    """
    Assemble a 2-D grid of images.
    rows: list of lists of np.ndarray or None (H × W × 3)
    """
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    total_w = label_col_w + n_cols * (cell_w + gap)
    total_h = label_row_h + n_rows * (cell_h + gap)

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = font_bold = ImageFont.load_default()

    # column headers
    for ci, clabel in enumerate(col_labels):
        x = label_col_w + ci * (cell_w + gap) + cell_w // 2
        draw.text((x - 20, 4), clabel, fill=(200, 200, 200), font=font)

    # row labels + cells
    for ri, (row_cells, rl) in enumerate(zip(rows, row_labels)):
        y0 = label_row_h + ri * (cell_h + gap)
        draw.text((4, y0 + cell_h // 2 - 8), rl, fill=(220, 220, 220),
                  font=font_bold)
        for ci, cell in enumerate(row_cells):
            x0 = label_col_w + ci * (cell_w + gap)
            if cell is None:
                canvas.paste(Image.new("RGB", (cell_w, cell_h), placeholder_color),
                             (x0, y0))
            else:
                img = Image.fromarray(cell).resize((cell_w, cell_h), Image.LANCZOS)
                canvas.paste(img, (x0, y0))

    return canvas


# ── per-experiment grid ────────────────────────────────────────────────────────

def build_progression(
    results_dir: str,
    val_every: int = 100,
    num_steps: int = 1000,
    cell_w: int = 128,   # width of EACH sub-image (original + recon → 2*cell_w)
    label_col_w: int = 80,
):
    steps = list(range(val_every, num_steps + val_every, val_every))
    img_dir = os.path.join(results_dir, "images")
    if not os.path.exists(img_dir):
        return None, [], []

    width_dirs = sorted(
        int(d[1:]) for d in os.listdir(img_dir)
        if d.startswith("d") and os.path.isdir(os.path.join(img_dir, d))
    )
    if not width_dirs:
        return None, [], []

    cell_total_w = cell_w   # single DiT recon column
    cell_h = cell_w

    rows = []
    row_labels = []
    for hs in width_dirs:
        row = [load_step_image(img_dir, hs, s, cell_w) for s in steps]
        rows.append(row)
        row_labels.append(f"d={hs}")

    col_labels = [f"s={s}" for s in steps]
    grid = make_grid(rows, row_labels, col_labels,
                     cell_w=cell_total_w, cell_h=cell_h,
                     label_col_w=label_col_w)
    return grid, width_dirs, steps


# ── combined grid (shared widths only) ───────────────────────────────────────

def build_combined(v1_dir: str, v2_dir: str,
                   val_every: int = 100, num_steps: int = 1000,
                   cell_w: int = 128, label_col_w: int = 80):
    steps = list(range(val_every, num_steps + val_every, val_every))
    v1_img = os.path.join(v1_dir, "images")
    v2_img = os.path.join(v2_dir, "images")

    def get_widths(idir):
        if not os.path.exists(idir): return []
        return sorted(int(d[1:]) for d in os.listdir(idir)
                      if d.startswith("d") and os.path.isdir(os.path.join(idir, d)))

    v1_ws = get_widths(v1_img)
    v2_ws = get_widths(v2_img)
    shared = sorted(set(v1_ws) & set(v2_ws))
    if not shared:
        return None

    cell_total_w = cell_w
    cell_h = cell_w
    rows, row_labels = [], []

    for hs in shared:
        for src_dir, tag, C in [(v1_img, "RAEv1 C=768", 768),
                                (v2_img, "RAEv2 C=1024", 1024)]:
            row = [load_step_image(src_dir, hs, s, cell_w) for s in steps]
            converges = "≥C" if hs >= C else "<C"
            row_labels.append(f"d={hs} {tag} ({converges})")
            rows.append(row)
        # blank separator row
        rows.append([None] * len(steps))
        row_labels.append("")

    col_labels = [f"s={s}" for s in steps]
    return make_grid(rows, row_labels, col_labels,
                     cell_w=cell_total_w, cell_h=cell_h,
                     label_col_w=label_col_w + 40)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir",    required=True)
    parser.add_argument("--v2-dir",    default=None)
    parser.add_argument("--out-dir",   default="comparison_results")
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--cell-w",    type=int, default=128,
                        help="Width of each sub-image in a cell (original+recon = 2×)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # RAEv1 progression
    grid_v1, widths_v1, steps = build_progression(
        args.v1_dir, args.val_every, args.num_steps, args.cell_w)
    if grid_v1 is not None:
        out = os.path.join(args.out_dir, "progression_v1.png")
        grid_v1.save(out)
        print(f"Saved {out}  ({grid_v1.width}×{grid_v1.height})")
    else:
        print("No RAEv1 results found.")

    # RAEv2 progression
    if args.v2_dir:
        grid_v2, widths_v2, _ = build_progression(
            args.v2_dir, args.val_every, args.num_steps, args.cell_w)
        if grid_v2 is not None:
            out = os.path.join(args.out_dir, "progression_v2.png")
            grid_v2.save(out)
            print(f"Saved {out}  ({grid_v2.width}×{grid_v2.height})")
        else:
            print("No RAEv2 results found.")


if __name__ == "__main__":
    main()
