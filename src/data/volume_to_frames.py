# [Week 2] src/data/volume_to_frames.py
# Dependency: src/data/nifti_io.py (Week 1)
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb, resample_isotropic


def write_frames_png(nifti_path, out_dir, do_resample=False, target_spacing=1.5):
    """Convert every axial slice of a CT volume to a numbered PNG file on disk."""
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    volume, _, voxel_spacing = load_volume(nifti_path)
    if do_resample:
        volume, _ = resample_isotropic(volume, voxel_spacing, target=target_spacing, order=1)
    u8_vol = apply_hu_window(volume, lo=-150, hi=250, as_uint8=True)
    total_slices = u8_vol.shape[2]
    for idx in range(total_slices):
        rgb_frame = to_rgb(u8_vol[:, :, idx])
        save_file = dst / f"{idx:05d}.png"
        Image.fromarray(rgb_frame).save(save_file)
    return total_slices


def frames_to_arrays(frames_dir):
    """Load all PNG frames from a folder into a single uint8 numpy array.

    Specification:
    - Read every .png in frames_dir in sorted (filename) order.
    - Stack along a new leading axis → shape (Z, H, W, 3).

    Args:
        frames_dir (str | Path): Folder produced by write_frames_png.

    Returns:
        np.ndarray: uint8 array of shape (Z, H, W, 3).
    """
    folder = Path(frames_dir)
    files = sorted(folder.glob("*.png"))
    if not files:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    frames = [np.array(Image.open(f), dtype=np.uint8) for f in files]
    return np.stack(frames, axis=0)
