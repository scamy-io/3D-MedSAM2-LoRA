# [Week 2] src/utils/viz.py
from pathlib import Path
import numpy as np
import imageio.v2 as imageio


def save_overlay_gif(vol_u8, pred3d, out_path, gt3d=None):
    """Write an animated GIF overlaying segmentation masks onto the CT volume.

    Specification:
    - Iterate over every axial slice (Z axis) of vol_u8.
    - Convert each greyscale (H, W) slice to RGB by repeating the channel.
    - Where pred3d[:, :, z] is True, apply a RED tint  (blend 50 % with [220,60,60]).
    - Where gt3d[:, :, z] is True (if gt3d is not None), apply a GREEN tint
      (blend 50 % with [60,220,60]).  GT is rendered beneath the prediction.
    - Collect all (H, W, 3) uint8 frames and write them as an animated GIF.
    - Create the parent directory of out_path if it does not exist.
    - Recommended frame duration: 80 ms.

    Args:
        vol_u8   (np.ndarray):      uint8 CT volume, shape (H, W, Z).
        pred3d   (np.ndarray):      bool/uint8 predicted mask, shape (H, W, Z).
        out_path (str | Path):      Destination .gif path.
        gt3d     (np.ndarray|None): Ground-truth mask same shape, or None.
    """
    save_path = Path(out_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_slices = vol_u8.shape[2]
    frames_list = []
    red = np.array([220, 60, 60], dtype=np.float32)
    green = np.array([60, 220, 60], dtype=np.float32)

    for z_idx in range(num_slices):
        cur_slice = vol_u8[:, :, z_idx]
        rgb = np.stack([cur_slice, cur_slice, cur_slice], axis=-1).astype(np.float32)

        if gt3d is not None:
            gt_active = gt3d[:, :, z_idx].astype(bool)
            rgb[gt_active] = 0.5 * rgb[gt_active] + 0.5 * green

        pred_active = pred3d[:, :, z_idx].astype(bool)
        rgb[pred_active] = 0.5 * rgb[pred_active] + 0.5 * red

        frames_list.append(rgb.astype(np.uint8))

    imageio.mimsave(str(save_path), frames_list, duration=0.08, loop=0)
