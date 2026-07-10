#!/usr/bin/env python3
"""03_train_lora.py — fine-tune SAM2 with LoRA on BTCV organ slices.

Usage (run from project root):
    python scripts/03_train_lora.py --organ 6
    python scripts/03_train_lora.py --organ 1 --qv-only

What this script does:
1. Builds a SAM2 predictor and injects LoRA adapters into the image encoder.
2. Builds a dataset of prompted 2D organ slices from the training split.
3. Runs the AMP + gradient-accumulation training loop (bfloat16, Fix #6/#7).
4. Saves the LoRA adapter weights to checkpoints/lora_organ<N>.pt.

This script is already complete — your job is to implement the functions it
calls:
  src/data/dataset.py      → BTCVSliceDataset           (Week 2)
  src/engine/predictor.py  → build_predictor             (Week 3)
  src/train/lora.py        → add_lora_peft, inject_lora_qv, trainable_report
  src/train/losses.py      → dice_loss, total_loss
  src/train/train.py       → run_training

Once training finishes, checkpoints/lora_organ<N>.pt is saved.
Pass it to 02_run_inference.py via --lora to see the improvement.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix #9: seed before ANY torch / numpy import
from src.utils.seed import set_seed
set_seed(42)

import yaml

# Week 2 dataset
from src.data.dataset import BTCVSliceDataset

# Week 3 predictor
from src.engine.predictor import build_predictor

# ── Functions YOU implement in Week 4 ────────────────────────────────────────
from src.train.lora import add_lora_peft, inject_lora_qv, trainable_report
from src.train.adapters import inject_sd_adapters
from src.train.train import run_training



def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune SAM2 on BTCV")
    parser.add_argument("--organ",   type=int, required=True,
                        help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--qv-only", action="store_true",
                        help="Use custom Q&V-only LoRA (Option B) instead of peft")
    parser.add_argument("--sd-adapter", action="store_true",
                        help="Inject Space-Depth (SD-Trans) Adapters into vision encoder")
    parser.add_argument("--config",  default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Step 1: Build model and inject LoRA ──────────────────────────────────
    # build_predictor() returns a SAM2VideoPredictor; .model is the SAM2Base.
    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    model     = predictor

    if args.sd_adapter:
        model.image_encoder = inject_sd_adapters(model.image_encoder)

    if args.qv_only:
        # Option B: custom LoRALinear — Q and V projections only (stretch goal)
        inject_lora_qv(model.image_encoder,
                       r=cfg["train"]["rank"], alpha=cfg["train"]["alpha"])
    else:
        # Option A (recommended): peft LoRA on the fused qkv linear layer
        model.image_encoder = add_lora_peft(
            model.image_encoder,
            r=cfg["train"]["rank"],
            alpha=cfg["train"]["alpha"],
        )

    # Sanity check: trainable params should be well under 1% of total
    trainable_report(model)


    # ── Step 2: Build dataset (training split only, no val cases) ────────────
    # BTCVSliceDataset yields (image, gt_mask, bbox) for each slice with organ.
    val_set   = set(cfg["split"]["val_cases"])
    img_dir   = Path(cfg["paths"]["raw_images"])
    lbl_dir   = Path(cfg["paths"]["raw_labels"])
    all_cases = [p.stem for p in sorted(img_dir.glob("*.nii"))
                 if p.stem not in val_set]

    print(f"Training cases: {len(all_cases)}  |  organ={args.organ}")
    dataset = BTCVSliceDataset(
        all_cases, args.organ,
        str(img_dir), str(lbl_dir),
        image_size=cfg["model"]["image_size"],
    )
    print(f"Dataset slices: {len(dataset)}")

    if len(dataset) == 0:
        print("No slices found for this organ. Check that imagesTr/*.nii exist.")
        return

    # ── Step 3: Train ────────────────────────────────────────────────────────
    # run_training() uses bfloat16 AMP (Fix #6), set_to_none (Fix #7),
    # and gradient accumulation (Fix #3). See train.py TODO for the full loop.
    save_path = Path(cfg["paths"]["checkpoints"]) / f"lora_organ{args.organ}.pt"
    losses = run_training(cfg, model, dataset, args.organ, str(save_path))

    print(f"\nTraining complete. Final epoch loss: {losses[-1]:.4f}")
    print(f"LoRA adapter saved -> {save_path}")
    print("Run 02_run_inference.py --lora", save_path, "to test the fine-tuned model.")


if __name__ == "__main__":
    main()
