# [Week 2] src/data/prompts.py
import numpy as np
from scipy import ndimage


def bbox_from_mask(mask2d, pad=4):
    """Return a SAM 2-format bounding box around the foreground of a 2-D mask.

    Specification:
    - Find all True/nonzero pixels; return None if there are none.
    - Compute the tight axis-aligned bounding box (min/max row and column).
    - Expand every side by pad pixels, clamped to image bounds.
    - Return format: float32 array [x0, y0, x1, y1]  (x = column, y = row).

    Args:
        mask2d (np.ndarray): 2-D bool or uint8 mask, shape (H, W).
        pad    (int):        Extra pixels on every side of the tight box.

    Returns:
        np.ndarray | None: float32 [x0, y0, x1, y1], or None if mask is empty.
    """
    pts = np.argwhere(mask2d)
    if pts.size == 0:
        return None
    ymin, xmin = pts.min(axis=0)
    ymax, xmax = pts.max(axis=0)
    h, w = mask2d.shape
    x0 = max(0, int(xmin) - pad)
    y0 = max(0, int(ymin) - pad)
    x1 = min(w - 1, int(xmax) + pad)
    y1 = min(h - 1, int(ymax) + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def best_start_slice(label_vol, organ_id):
    """Return the axial index of the slice with the largest organ cross-section.

    Specification:
    - Binarise label_vol to the target organ (label_vol == organ_id).
    - Raise ValueError if the organ is absent.
    - For each z-slice compute the foreground pixel count.
    - Return the z index with the maximum count (argmax across the Z axis).
    - Hint: scipy.ndimage can help identify connected components if needed.

    Args:
        label_vol (np.ndarray): Integer label volume, shape (H, W, Z).
        organ_id  (int):        BTCV label integer for the target organ.

    Returns:
        int: Z-axis index of the slice with the most organ pixels.

    Raises:
        ValueError: If organ_id is not present anywhere in label_vol.
    """
    mask = (label_vol == organ_id)
    if not mask.any():
        raise ValueError(
            f"Organ {organ_id} is not present anywhere in the label volume."
        )
    areas = mask.sum(axis=(0, 1))
    return int(np.argmax(areas))
