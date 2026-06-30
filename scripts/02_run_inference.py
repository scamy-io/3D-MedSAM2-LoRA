#!/usr/bin/env python3
"""02_run_inference.py — SAM2 bidirectional propagation on one BTCV case.

Usage (run from project root):
    python scripts/02_run_inference.py --case img0001 --organ 6
    python scripts/02_run_inference.py --case img0001 --organ 6 --lora checkpoints/lora_organ6.pt

What this script does:
1. Loads a CT volume and its labels.
2. Writes PNG frames if they don't exist yet.
3. Finds the prompt slice + bounding box.
4. Runs SAM2 bidirectional propagation to get a 3D organ mask.
5. Saves the predicted mask as .npy and a GIF overlay for visual inspection.

This script is already complete — your job is to implement the functions it
calls in src/engine/predictor.py and src/engine/propagate.py.
Once those are implemented, run this script to get your zero-shot result.

Expected output:
  - outputs/<case>/mask_organ<N>.npy   (3D bool array, shape H×W×Z)
  - outputs/<case>/overlay_organ<N>.gif  (green=GT, red=prediction)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np

# Week 1 & 2 functions (already implemented)
from src.data.nifti_io import load_volume, apply_hu_window
from src.data.volume_to_frames import write_frames_png
from src.data.prompts import bbox_from_mask, best_start_slice
from src.utils.viz import save_overlay_gif

# ── Functions YOU implement in Week 3 ────────────────────────────────────────
from src.engine.predictor import build_predictor, init_state              # Week 3
from src.engine.propagate import propagate_bidirectional                  # Week 3


def main():
    parser = argparse.ArgumentParser(description="SAM2 zero-shot / LoRA inference")
    parser.add_argument("--case",   required=True, help="e.g. img0001")
    parser.add_argument("--organ",  type=int, required=True,
                        help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--lora",   default=None,
                        help="Optional: path to LoRA adapter .pt from Week 4 training")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_path   = Path(cfg["paths"]["raw_images"]) / f"{args.case}.nii"
    lbl_path   = Path(cfg["paths"]["raw_labels"]) / f"{args.case.replace('img', 'label')}.nii"
    frames_dir = Path(cfg["paths"]["processed"]) / args.case
    out_dir    = Path("outputs") / args.case
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Write PNG frames (if not already on disk) ────────────────────
    if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
        print("Writing PNG frames...")
        write_frames_png(str(img_path), str(frames_dir))

    # ── Step 2: Load volume + labels ─────────────────────────────────────────
    vol, _, spacing = load_volume(str(img_path))
    lbl, _, _       = load_volume(str(lbl_path))
    vol_u8 = apply_hu_window(vol)          # (H, W, Z) uint8

    try:
        start_z = best_start_slice(lbl, args.organ)
    except ValueError as e:
        print(e); return

    gt_slice = (lbl[:, :, start_z] == args.organ).astype(np.uint8)
    bbox = bbox_from_mask(gt_slice, pad=4)
    if bbox is None:
        print(f"Organ {args.organ} not found in case {args.case}."); return

    # ── Step 3: Build SAM2 predictor ─────────────────────────────────────────
    # build_predictor() creates a SAM2VideoPredictor from the config + checkpoint.
    print(f"Building predictor  start_z={start_z}  bbox={bbox}")
    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])

    # Optionally load LoRA adapter (produced by Week 4 training)
    if args.lora:
        import torch
        ckpt = torch.load(args.lora, map_location="cuda")
        
        # Dynamically detect and inject active adapters based on checkpoint keys
        has_sd = any("sd_adapter" in k for k in ckpt.keys())
        has_peft = any("base_model" in k for k in ckpt.keys())
        has_custom_qv = any(not "base_model" in k and (".A" in k or ".B" in k) for k in ckpt.keys())

        if has_sd:
            from src.train.adapters import inject_sd_adapters
            predictor.model.image_encoder = inject_sd_adapters(predictor.model.image_encoder)
        
        if has_peft:
            from src.train.lora import add_lora_peft
            predictor.model.image_encoder = add_lora_peft(
                predictor.model.image_encoder,
                r=cfg["train"]["rank"],
                alpha=cfg["train"]["alpha"]
            )
        elif has_custom_qv:
            from src.train.lora import inject_lora_qv
            predictor.model.image_encoder = inject_lora_qv(
                predictor.model.image_encoder,
                r=cfg["train"]["rank"],
                alpha=cfg["train"]["alpha"]
            )

        predictor.model.image_encoder.load_state_dict(ckpt, strict=False)
        predictor.model.to("cuda")
        print(f"Loaded LoRA adapter from {args.lora}")

    # ── Step 4: Initialise video state ───────────────────────────────────────
    # init_state() calls predictor.init_state(video_path=frames_dir, ...)
    # IMPORTANT: offload_video_to_cpu=True and offload_state_to_cpu=True are
    # mandatory on 8 GB VRAM — include them inside init_state().
    state = init_state(predictor, str(frames_dir))

    # ── Step 5: Bidirectional propagation ────────────────────────────────────
    # propagate_bidirectional() adds the bbox prompt at start_z, then propagates
    # forward and backward through all slices, returning a (H, W, Z) bool array.
    target_hw = (vol.shape[0], vol.shape[1])
    pred3d = propagate_bidirectional(predictor, state, start_z, bbox,
                                     target_hw=target_hw)

    # ── Step 6: Save results ─────────────────────────────────────────────────
    npy_path = out_dir / f"mask_organ{args.organ}.npy"
    np.save(str(npy_path), pred3d)
    print(f"Saved mask  shape={pred3d.shape}  -> {npy_path}")

    gt3d = (lbl == args.organ).astype(bool)
    gif_path = out_dir / f"overlay_organ{args.organ}.gif"
    save_overlay_gif(vol_u8, pred3d, str(gif_path), gt3d)
    print(f"Saved GIF -> {gif_path}")
    print("\nDONE — open the GIF and check that the mask follows the organ across slices.")


if __name__ == "__main__":
    main()
