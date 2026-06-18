import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_volume_sitk(path):
    """Fallback reader to load a NIfTI file using SimpleITK.

    Args:
        path: path to a .nii or .nii.gz file

    Returns:
        vol     : float32 numpy array of shape (H, W, Z)
        affine  : 4x4 identity matrix fallback
        spacing : tuple (sx, sy, sz) — voxel size in mm
    """
    try:
        import SimpleITK as sitk

        sitk_img = sitk.ReadImage(str(path))
        vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        # SimpleITK loads in (Z, Y, X) order -> transpose to (X, Y, Z) / (H, W, Z)
        vol = np.transpose(vol, (2, 1, 0))
        spacing = tuple(float(s) for s in sitk_img.GetSpacing()[:3])
        affine = np.eye(4, dtype=np.float32)
        return vol, affine, spacing
    except Exception as e:
        raise RuntimeError(f"SimpleITK failed to load {path}: {e}")


def load_volume(path):
    """Load a NIfTI file and return the 3D volume, affine, and voxel spacing.

    Args:
        path: path to a .nii or .nii.gz file

    Returns:
        vol     : float32 numpy array of shape (H, W, Z)
        affine  : 4x4 affine matrix from the NIfTI header
        spacing : tuple (sx, sy, sz) — voxel size in mm
    """
    try:
        img = nib.load(str(path))
        img = nib.as_closest_canonical(img)
        vol = img.get_fdata().astype(np.float32)
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        affine = img.affine
    except Exception as err:
        print(f"nibabel failed to load {path} ({err}). Falling back to SimpleITK...")
        vol, affine, spacing = load_volume_sitk(path)

    print(f"Loaded volume: {path} | Spacing: {spacing}")
    return vol, affine, spacing


def apply_hu_window(vol, lo=-150, hi=250, as_uint8=True):
    """Clip a CT volume to a soft-tissue HU window and rescale to [0, 255] or [0, 1].

    TODO:
    1. Clip:     v = np.clip(vol, lo, hi)
    2. Rescale:  v = (v - lo) / (hi - lo)   →  values now in [0, 1]
    3. If as_uint8 is True:  return (v * 255).astype(np.uint8)
       If as_uint8 is False: return v.astype(np.float32)

    Args:
        vol      : float32 volume array (H, W, Z) in Hounsfield Units
        lo       : lower clip bound  (default -150 HU)
        hi       : upper clip bound  (default  250 HU)
        as_uint8 : True  → return uint8 [0, 255]  (for PNG frames)
                   False → return float [0, 1]     (for direct array feeding)

    Returns:
        Windowed array, same shape as vol
    """
    clipped = np.clip(vol, lo, hi)
    norm = (clipped - lo) / float(hi - lo)
    if as_uint8:
        return (norm * 255.0).astype(np.uint8)
    return norm.astype(np.float32)


def to_rgb(slice2d_u8):
    """Convert a (H, W) uint8 greyscale slice to (H, W, 3) by repeating the channel.

    TODO:
    1. Add a channel dimension: slice2d_u8[..., None]  →  shape (H, W, 1)
    2. Repeat 3 times along axis 2:
           np.repeat(slice2d_u8[..., None], 3, axis=2)  →  shape (H, W, 3)
    3. Return the result

    SAM 2's ViT image encoder expects 3-channel RGB input even for greyscale CT.

    Args:
        slice2d_u8: 2D uint8 array of shape (H, W)

    Returns:
        uint8 array of shape (H, W, 3)
    """
    return np.stack([slice2d_u8, slice2d_u8, slice2d_u8], axis=-1)


def resample_isotropic(vol, spacing, target=1.5, order=1):
    """Resample a volume to isotropic voxel spacing using scipy zoom.

    OPTIONAL — only call this if cfg.preprocess.resample_isotropic is True.

    TODO:
    1. Compute zoom factors:  factors = [s / target for s in spacing]
    2. Apply zoom:  resampled = zoom(vol, factors, order=order)
       - Use order=1 for image volumes (bilinear interpolation)
       - Use order=0 for label/mask volumes (nearest-neighbour — no blurring)
    3. Return (resampled, (target, target, target))

    Args:
        vol     : input volume array (H, W, Z)
        spacing : current voxel spacing in mm — e.g. (0.97, 0.97, 3.0)
        target  : desired isotropic spacing in mm (default 1.5 mm)
        order   : interpolation order (1 for images, 0 for masks)

    Returns:
        resampled volume, new spacing tuple (target, target, target)
    """
    zoom_factors = [s / target for s in spacing]
    resampled = zoom(vol, zoom_factors, order=order)
    return resampled, (target, target, target)
