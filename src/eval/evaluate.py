# [Week 5] src/eval/evaluate.py
# Dependencies: all previous weeks (weeks 1-4)
from pathlib import Path
import numpy as np
import torch

from src.data.nifti_io import apply_hu_window, load_volume
from src.data.prompts import bbox_from_mask, best_start_slice
from src.data.volume_to_frames import write_frames_png
from src.engine.predictor import build_predictor, init_state
from src.engine.propagate import propagate_bidirectional
from src.eval.metrics import dice_score, hd95, vpe
from src.utils.viz import save_overlay_gif


def _load_predictor(cfg, lora_path=None):
    """Build a SAM 2 video predictor and optionally load a LoRA adapter.

    Specification:
    - predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    - If lora_path is not None and the file exists:
          adapter = torch.load(lora_path, map_location="cuda")
          predictor.model.image_encoder.load_state_dict(adapter, strict=False)
    - Return predictor.

    Args:
        cfg       (dict):      Config dict from configs/default.yaml.
        lora_path (str|None):  Path to a saved LoRA .pt file, or None.

    Returns:
        SAM2VideoPredictor (with LoRA weights loaded if lora_path given).
    """
    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])

    if lora_path is not None:
        target_path = Path(lora_path)
        if target_path.exists():
            weights = torch.load(target_path, map_location="cuda")
            # Dynamically detect and inject active adapters based on checkpoint keys
            has_sd = any("sd_adapter" in k for k in weights.keys())
            has_peft = any("base_model" in k for k in weights.keys())
            has_custom_qv = any(not "base_model" in k and (".A" in k or ".B" in k) for k in weights.keys())

            if has_sd:
                from src.train.adapters import inject_sd_adapters
                predictor.model.image_encoder = inject_sd_adapters(predictor.model.image_encoder)

            if has_peft:
                from src.train.lora import add_lora_peft
                predictor.model.image_encoder = add_lora_peft(
                    predictor.model.image_encoder,
                    r=cfg["train"]["rank"],
                    alpha=cfg["train"]["alpha"],
                )
            elif has_custom_qv:
                from src.train.lora import inject_lora_qv
                predictor.model.image_encoder = inject_lora_qv(
                    predictor.model.image_encoder,
                    r=cfg["train"]["rank"],
                    alpha=cfg["train"]["alpha"],
                )

            predictor.model.image_encoder.load_state_dict(weights, strict=False)
            predictor.model.to("cuda")

    return predictor


def evaluate_organ(cfg, organ_id, lora_path=None):
    """Run zero-shot and optional LoRA inference on all validation cases.

    Specification:
    - val_cases = cfg["split"]["val_cases"]
    - img_dir   = Path(cfg["paths"]["raw_images"])
    - lbl_dir   = Path(cfg["paths"]["raw_labels"])
    - pred_zs   = _load_predictor(cfg)
    - pred_lora = _load_predictor(cfg, lora_path) if lora_path else None

    For each case in val_cases:
      a. Paths:
             img_path   = img_dir / f"{case}.nii"
             lbl_path   = lbl_dir / f"{case.replace('img','label')}.nii"
             frames_dir = Path(cfg["paths"]["processed"]) / case
             out_dir    = Path("outputs") / case ; out_dir.mkdir(parents=True, exist_ok=True)
         Skip if either file does not exist.

      b. Write PNG frames if not already done.

      c. Load volume and labels:
             vol, _, spacing = load_volume(img_path)
             lbl, _, _       = load_volume(lbl_path)
             vol_u8          = apply_hu_window(vol)
             gt3d            = (lbl == organ_id).astype(bool)
         Skip if gt3d has no True pixels (organ absent in this case).

      d. Find prompt:
             start_z  = best_start_slice(lbl, organ_id)
             gt_slice = (lbl[:,:,start_z] == organ_id).astype(np.uint8)
             bbox     = bbox_from_mask(gt_slice, pad=4)
         Skip if bbox is None.

      e. Zero-shot inference:
             state_zs = init_state(pred_zs, str(frames_dir))
             mask_zs  = propagate_bidirectional(pred_zs, state_zs, start_z, bbox,
                                                target_hw=(vol.shape[0], vol.shape[1]))
             dsc_zs = dice_score(mask_zs, gt3d)
             hd_zs  = hd95(mask_zs, gt3d, spacing)
         Save GIF to out_dir / f"zeroshot_{organ_id}.gif".

      f. LoRA inference (only if pred_lora is not None):
             state_lo = init_state(pred_lora, str(frames_dir))
             mask_lo  = propagate_bidirectional(pred_lora, state_lo, start_z, bbox,
                                                target_hw=(vol.shape[0], vol.shape[1]))
             dsc_lo = dice_score(mask_lo, gt3d)
             hd_lo  = hd95(mask_lo, gt3d, spacing)
         If pred_lora is None, set dsc_lo = hd_lo = float("nan").

      g. Append dict(case=case, dsc_zs=dsc_zs, hd95_zs=hd_zs,
                     dsc_lora=dsc_lo, hd95_lora=hd_lo) to results.

    Args:
        cfg       (dict):     Config dict from configs/default.yaml.
        organ_id  (int):      BTCV integer label for the target organ.
        lora_path (str|None): LoRA adapter .pt path, or None for zero-shot only.

    Returns:
        list[dict]: One dict per processed val case with keys:
                    case, dsc_zs, hd95_zs, dsc_lora, hd95_lora.
    """
    val_cases = cfg["split"]["val_cases"]
    img_dir = Path(cfg["paths"]["raw_images"])
    lbl_dir = Path(cfg["paths"]["raw_labels"])

    pred_zs = _load_predictor(cfg)
    pred_lora = _load_predictor(cfg, lora_path) if lora_path else None

    results = []

    for case in val_cases:
        img_path = img_dir / f"{case}.nii"
        lbl_file_name = f"{case.replace('img', 'label')}.nii"
        lbl_path = lbl_dir / lbl_file_name
        frames_dir = Path(cfg["paths"]["processed"]) / case
        out_dir = Path("outputs") / case
        out_dir.mkdir(parents=True, exist_ok=True)

        if not img_path.exists() or not lbl_path.exists():
            continue

        if not frames_dir.exists() or not any(frames_dir.iterdir()):
            write_frames_png(img_path, frames_dir)

        vol, _, spacing = load_volume(img_path)
        lbl, _, _ = load_volume(lbl_path)
        vol_u8 = apply_hu_window(vol)
        gt3d = (lbl == organ_id).astype(bool)

        if not gt3d.any():
            continue

        start_z = best_start_slice(lbl, organ_id)
        gt_slice = (lbl[:, :, start_z] == organ_id).astype(np.uint8)
        bbox = bbox_from_mask(gt_slice, pad=4)

        if bbox is None:
            continue

        target_hw = (vol.shape[0], vol.shape[1])

        state_zs = init_state(pred_zs, str(frames_dir))
        mask_zs = propagate_bidirectional(
            pred_zs, state_zs, start_z, bbox, target_hw=target_hw
        )
        dsc_zs = dice_score(mask_zs, gt3d)
        hd_zs = hd95(mask_zs, gt3d, spacing)
        vpe_zs = vpe(mask_zs, gt3d, spacing)

        save_overlay_gif(
            vol_u8,
            mask_zs,
            str(out_dir / f"zeroshot_{organ_id}.gif"),
            gt3d,
        )

        if pred_lora is not None:
            state_lo = init_state(pred_lora, str(frames_dir))
            mask_lo = propagate_bidirectional(
                pred_lora, state_lo, start_z, bbox, target_hw=target_hw
            )
            dsc_lo = dice_score(mask_lo, gt3d)
            hd_lo = hd95(mask_lo, gt3d, spacing)
            vpe_lo = vpe(mask_lo, gt3d, spacing)

            save_overlay_gif(
                vol_u8,
                mask_lo,
                str(out_dir / f"lora_{organ_id}.gif"),
                gt3d,
            )
        else:
            dsc_lo = float("nan")
            hd_lo = float("nan")
            vpe_lo = float("nan")

        results.append(
            dict(
                case=case,
                dsc_zs=dsc_zs,
                hd95_zs=hd_zs,
                vpe_zs=vpe_zs,
                dsc_lora=dsc_lo,
                hd95_lora=hd_lo,
                vpe_lora=vpe_lo,
            )
        )

    return results


def print_table(rows, organ_id):
    """Print a formatted DSC / HD95 summary table and return aggregated metrics.

    Specification:
    - Compute mean and std for each metric across all rows, ignoring NaN values
      (use np.nanmean and np.nanstd).
    - Print in this exact format:
          ============================================================
            Organ {organ_id}  |  {n} val cases
          ============================================================
            Metric         Zero-shot           LoRA
            --------------------------------------------------
            DSC            {mean:.3f} ± {std:.3f}   {mean:.3f} ± {std:.3f}
            HD95 (mm)      {mean:.1f}  ± {std:.1f}   {mean:.1f}  ± {std:.1f}
          ============================================================
    - Return a dict with keys:
          organ, dsc_zs_mean, dsc_zs_std, dsc_lora_mean, dsc_lora_std,
          hd95_zs_mean, hd95_zs_std, hd95_lora_mean, hd95_lora_std

    Args:
        rows     (list[dict]): Output of evaluate_organ().
        organ_id (int):        Organ label integer (for display).

    Returns:
        dict: Aggregated metric statistics.
    """
    n = len(rows)

    dsc_zs_vals = [r["dsc_zs"] for r in rows]
    hd95_zs_vals = [r["hd95_zs"] for r in rows]
    vpe_zs_vals = [r["vpe_zs"] for r in rows]

    dsc_lo_vals = [r["dsc_lora"] for r in rows]
    hd95_lo_vals = [r["hd95_lora"] for r in rows]
    vpe_lo_vals = [r["vpe_lora"] for r in rows]

    dsc_zs_m, dsc_zs_s = np.nanmean(dsc_zs_vals), np.nanstd(dsc_zs_vals)
    hd95_zs_m, hd95_zs_s = np.nanmean(hd95_zs_vals), np.nanstd(hd95_zs_vals)
    vpe_zs_m, vpe_zs_s = np.nanmean(vpe_zs_vals), np.nanstd(vpe_zs_vals)

    dsc_lo_m, dsc_lo_s = np.nanmean(dsc_lo_vals), np.nanstd(dsc_lo_vals)
    hd95_lo_m, hd95_lo_s = np.nanmean(hd95_lo_vals), np.nanstd(hd95_lo_vals)
    vpe_lo_m, vpe_lo_s = np.nanmean(vpe_lo_vals), np.nanstd(vpe_lo_vals)

    print("=" * 60)
    print(f"  Organ {organ_id}  |  {n} val cases")
    print("=" * 60)
    print(f"  Metric         Zero-shot           LoRA")
    print("  " + "-" * 50)
    print(
        f"  DSC            {dsc_zs_m:.3f} ± {dsc_zs_s:.3f}   {dsc_lo_m:.3f} ± {dsc_lo_s:.3f}"
    )
    print(
        f"  HD95 (mm)      {hd95_zs_m:.1f}  ± {hd95_zs_s:.1f}   {hd95_lo_m:.1f}  ± {hd95_lo_s:.1f}"
    )
    print(
        f"  VPE (cm³)      {vpe_zs_m:.2f}  ± {vpe_zs_s:.2f}   {vpe_lo_m:.2f}  ± {vpe_lo_s:.2f}"
    )
    print("=" * 60)

    return dict(
        organ=organ_id,
        dsc_zs_mean=dsc_zs_m,
        dsc_zs_std=dsc_zs_s,
        dsc_lora_mean=dsc_lo_m,
        dsc_lora_std=dsc_lo_s,
        hd95_zs_mean=hd95_zs_m,
        hd95_zs_std=hd95_zs_s,
        hd95_lora_mean=hd95_lo_m,
        hd95_lora_std=hd95_lo_s,
        vpe_zs_mean=vpe_zs_m,
        vpe_zs_std=vpe_zs_s,
        vpe_lora_mean=vpe_lo_m,
        vpe_lora_std=vpe_lo_s,
    )