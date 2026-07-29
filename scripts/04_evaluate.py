#!/usr/bin/env python3
"""04_evaluate.py — evaluate zero-shot vs LoRA Dice + HD95 on the val split.

Usage (run from project root):
    python scripts/04_evaluate.py --organ 6
    python scripts/04_evaluate.py --organ 6 --lora checkpoints/lora_organ6.pt

What this script does:
1. Runs bidirectional SAM2 propagation on every val-split CT volume.
2. Computes DSC and HD95 for both zero-shot and LoRA-tuned predictions.
3. Prints a summary table (zero-shot vs LoRA) and saves a per-case CSV.
4. Saves overlay GIFs for visual inspection.

This script is already complete — your job is to implement the functions it
calls in src/eval/evaluate.py (Week 5):
  _load_predictor(cfg, lora_path)     → builds predictor, loads adapter
  evaluate_organ(cfg, organ_id, ...)  → runs inference on all val cases
  print_table(rows, organ_id)         → prints DSC / HD95 summary table

And the metrics in src/eval/metrics.py (Week 4):
  dice_score(pred3d, gt3d)            → volumetric Dice coefficient
  hd95(pred3d, gt3d, spacing)         → 95th-percentile Hausdorff in mm

Expected output (example):
  ============================================================
    Organ 6  |  6 val cases
  ============================================================
    Metric         Zero-shot           LoRA
    --------------------------------------------------
    DSC            0.912 ± 0.023       0.941 ± 0.018
    HD95 (mm)      4.2   ± 1.1         2.9   ± 0.8
  ============================================================
  Per-case CSV saved -> outputs/eval_organ6.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix #9: seed before any torch imports
from src.utils.seed import set_seed
set_seed(42)

import yaml

from src.eval.evaluate import evaluate_organ, print_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM2 on BTCV val split")
    parser.add_argument("--organ",  type=int, required=True,
                        help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--lora",   default=None,
                        help="LoRA adapter .pt — auto-detected from checkpoints/ if absent")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Auto-detect LoRA adapter for this organ if not provided
    lora_path = args.lora
    if lora_path is None:
        auto = Path(cfg["paths"]["checkpoints"]) / f"lora_organ{args.organ}.pt"
        if auto.exists():
            lora_path = str(auto)
            print(f"Auto-detected LoRA adapter: {lora_path}")
        else:
            print("No LoRA adapter found — running zero-shot only.")

    print(f"\nEvaluating organ={args.organ}  val cases={cfg['split']['val_cases']}\n")

    # ── evaluate_organ() runs inference on all val cases and returns a list
    # of dicts: {case, dsc_zs, hd95_zs, dsc_lora, hd95_lora}
    rows = evaluate_organ(cfg, args.organ, lora_path)

    if not rows:
        print("No results — check that val cases exist in the data directory.")
        return

    # ── print_table() aggregates mean ± std and prints the summary table
    summary = print_table(rows, args.organ)

    # Save per-case CSV
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_organ{args.organ}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-case CSV saved -> {csv_path}")

    # Print reference baselines from the literature
    baselines = {6: "~0.93", 1: "~0.94", 11: "~0.72"}
    if args.organ in baselines:
        print(f"\nPublished SAM2 zero-shot baseline for organ {args.organ}: "
              f"DSC {baselines[args.organ]}")


if __name__ == "__main__":
    main()
