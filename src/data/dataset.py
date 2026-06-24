# [Week 2] src/data/dataset.py
# Dependency: src/data/nifti_io.py (Week 1), src/data/prompts.py (Week 2)
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb, resample_isotropic
from src.data.prompts import bbox_from_mask


class BTCVSliceDataset(Dataset):
    """2-D prompted dataset of organ-containing axial slices for LoRA training."""

    def __init__(
        self,
        cases,
        organ_id,
        image_dir,
        label_dir,
        image_size=1024,
        do_resample=False,
        target_spacing=1.5,
    ):
        self.samples = []
        img_dir = Path(image_dir)
        lbl_dir = Path(label_dir)
        target_size = (image_size, image_size)

        for case_id in cases:
            stem = case_id.replace("img", "")
            img_file = img_dir / f"img{stem}.nii"
            if not img_file.exists():
                img_file = img_dir / f"img{stem}.nii.gz"

            lbl_file = lbl_dir / f"label{stem}.nii"
            if not lbl_file.exists():
                lbl_file = lbl_dir / f"label{stem}.nii.gz"

            img_data, _, sp = load_volume(img_file)
            lbl_data, _, _ = load_volume(lbl_file)

            if do_resample:
                img_data, sp = resample_isotropic(img_data, sp, target=target_spacing, order=1)
                lbl_data, _ = resample_isotropic(lbl_data, sp, target=target_spacing, order=0)

            img_u8 = apply_hu_window(img_data, lo=-150, hi=250, as_uint8=True)
            depth = img_u8.shape[2]

            for z in range(depth):
                slice_lbl = lbl_data[:, :, z]
                if np.any(slice_lbl == organ_id):
                    mask = (slice_lbl == organ_id).astype(np.uint8)
                    slice_rgb = to_rgb(img_u8[:, :, z])

                    scaled_img = np.array(
                        Image.fromarray(slice_rgb).resize(target_size, resample=Image.BILINEAR),
                        dtype=np.uint8,
                    )
                    scaled_mask = np.array(
                        Image.fromarray(mask).resize(target_size, resample=Image.NEAREST),
                        dtype=np.uint8,
                    )
                    box = bbox_from_mask(scaled_mask, pad=4)
                    if box is not None:
                        self.samples.append((scaled_img, scaled_mask, box))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_arr, gt_arr, box_arr = self.samples[idx]
        image_t = torch.from_numpy(img_arr).float() / 255.0
        mask_t = torch.from_numpy(gt_arr).float()
        box_t = torch.from_numpy(box_arr).float()
        return image_t, mask_t, box_t