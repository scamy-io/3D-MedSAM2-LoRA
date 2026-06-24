#!/usr/bin/env python3
"""01_preprocess.py — preprocess a single BTCV case to PNG frames + overlay.

Usage (run from project root):
    python scripts/01_preprocess.py --case img0001 --organ 6

What this script does:
1. Loads a NIfTI CT volume + its organ label file.
2. Converts every axial slice to a PNG frame folder (SAM2 video input format).
3. Finds the best prompt slice for the chosen organ, draws the bounding box,
   and saves an overlay PNG so you can visually verify your pipeline.

This script is already complete — your job is to implement the functions it
calls in src/data/nifti_io.py, src/data/volume_to_frames.py, and
src/data/prompts.py.  Once those are implemented, run this script and check
that the overlay looks correct (organ highlighted in red, green bbox around it).
"""
import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import imageio.v2 as imageio

# ── Functions YOU implement in Week 2 ────────────────────────────────────────
from src.data.nifti_io import load_volume, apply_hu_window, to_rgb       # Week 1
from src.data.volume_to_frames import write_frames_png
from src.data.prompts import bbox_from_mask, best_start_slice


def draw_rect(img, x0, y0, x1, y1, color=(0, 255, 0), thickness=2):
    """Draw a filled-border rectangle on an (H, W, 3) uint8 array in-place."""
    t = thickness
    img[y0:y0 + t, x0:x1] = color
    img[y1 - t:y1, x0:x1] = color
    img[y0:y1, x0:x0 + t] = color
    img[y0:y1, x1 - t:x1] = color


def main():
    parser = argparse.ArgumentParser(description="Preprocess one BTCV case")
    parser.add_argument("--case",   required=True, help="e.g. img0001")
    parser.add_argument("--organ",  type=int, default=6,
                        help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_path = Path(cfg["paths"]["raw_images"]) / f"{args.case}.nii"
    lbl_path = Path(cfg["paths"]["raw_labels"]) / f"{args.case.replace('img', 'label')}.nii"
    out_dir  = Path(cfg["paths"]["processed"]) / args.case

    # ── Step 1: Load the CT volume ───────────────────────────────────────────
    # load_volume() returns (vol, affine, spacing)
    # vol shape: (H, W, Z) float32   spacing: (sx, sy, sz) mm
    print(f"Loading {img_path}")
    vol, affine, spacing = load_volume(str(img_path))
    print(f"  shape={vol.shape}  spacing={tuple(f'{s:.2f}' for s in spacing)} mm")

    # ── Step 2: Write PNG frames (one per axial slice) ───────────────────────
    # write_frames_png() creates out_dir/00000.png, 00001.png, ...
    # Returns the number of frames written.
    do_resample = cfg.get("preprocess", {}).get("resample_isotropic", False)
    target_spacing = cfg.get("preprocess", {}).get("target_spacing_mm", 1.5)

    n = write_frames_png(
        str(img_path),
        str(out_dir),
        do_resample=do_resample,
        target_spacing=target_spacing,
    )

    print(f"  Wrote {n} PNG frames -> {out_dir}")

    if not lbl_path.exists():
        print("  No label file found — skipping overlay.")
        return

    # ── Step 3: Find best prompt slice and draw overlay ──────────────────────
    lbl, _, _ = load_volume(str(lbl_path))

    try:
        # best_start_slice() finds the axial slice with the most organ pixels
        start_z = best_start_slice(lbl, args.organ)
    except ValueError as e:
        print(f"  {e}")
        return

    vol_u8   = apply_hu_window(vol)
    slice_u8 = vol_u8[:, :, start_z]
    rgb      = to_rgb(slice_u8).copy()
    gt_slice = (lbl[:, :, start_z] == args.organ).astype(np.uint8)

    # Highlight organ region in red
    rgb[gt_slice == 1] = (
        rgb[gt_slice == 1] * 0.5 + np.array([220, 60, 60]) * 0.5
    ).astype(np.uint8)

    # Draw green bounding box — this is the SAM2 prompt
    bbox = bbox_from_mask(gt_slice, pad=4)
    if bbox is not None:
        x0, y0, x1, y1 = bbox.astype(int)
        draw_rect(rgb, x0, y0, x1, y1, color=(0, 255, 0), thickness=2)

    overlay_path = out_dir / "overlay.png"
    imageio.imwrite(str(overlay_path), rgb)
    print(f"  Best start slice: z={start_z}  bbox={bbox}")
    print(f"  Saved overlay -> {overlay_path}")
    print("\nDONE — open overlay.png and check that the green box surrounds the organ.")


if __name__ == "__main__":
    main()
