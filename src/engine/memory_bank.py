"""src/engine/memory_bank.py — Short-Long Memory Bank for SAM 2 3D propagation.

Implements the Dual-Memory architecture specified in idea.md:
1. Long-Term Memory: Permanently anchors to the user's initial prompt slice (start_z)
   and ground truth bounding box to serve as a spatial regularizer.
2. Short-Term Memory: Maintains a sliding window of the immediately preceding k slices
   to handle slice-to-slice shape transitions.
"""

import numpy as np


class ShortLongMemoryBank:
    """Dual-Memory Bank for SAM 2 Volumetric Segmentation.

    Args:
        k_slices (int): Maximum number of short-term preceding slices to keep.
    """

    def __init__(self, k_slices=3):
        self.k_slices = k_slices
        self.prompt_z = None
        self.prompt_bbox = None
        self.prompt_mask = None
        self.short_term_masks = {}

    def register_prompt(self, start_z, bbox, mask=None):
        """Register the initial prompt slice as the permanent long-term anchor.

        Args:
            start_z (int): Axial slice index of the initial prompt.
            bbox (np.ndarray): Bounding box [x0, y0, x1, y1].
            mask (np.ndarray | None): 2D ground truth mask on start_z if available.
        """
        self.prompt_z = start_z
        self.prompt_bbox = bbox
        self.prompt_mask = mask
        if mask is not None:
            self.short_term_masks[start_z] = mask.astype(bool)

    def add_slice_mask(self, slice_z, mask_bool):
        """Store a slice prediction into the short-term memory bank.

        Args:
            slice_z (int): Axial slice index.
            mask_bool (np.ndarray): 2D boolean segmentation mask.
        """
        self.short_term_masks[slice_z] = mask_bool

        # Prune short-term memory to keep only the k most recent slices
        if len(self.short_term_masks) > self.k_slices + 1:
            sorted_keys = sorted(
                self.short_term_masks.keys(),
                key=lambda z: abs(z - slice_z),
            )
            # Retain prompt_z (long-term) + k closest slices
            keep = set(sorted_keys[: self.k_slices])
            if self.prompt_z is not None:
                keep.add(self.prompt_z)
            for z_idx in list(self.short_term_masks.keys()):
                if z_idx not in keep:
                    del self.short_term_masks[z_idx]

    def reset_short_term(self):
        """Purge short-term memory while retaining the long-term prompt anchor."""
        long_term_mask = (
            self.short_term_masks.get(self.prompt_z) if self.prompt_z is not None else None
        )
        self.short_term_masks.clear()
        if self.prompt_z is not None and long_term_mask is not None:
            self.short_term_masks[self.prompt_z] = long_term_mask
