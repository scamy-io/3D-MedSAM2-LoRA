import numpy as np
from PIL import Image
from src.engine.memory_bank import ShortLongMemoryBank


def propagate_bidirectional(
    predictor, state, start_z, bbox, target_hw=None, min_area_pixels=5, k_slices=3
):
    """Run SAM 2 bidirectional propagation with Short-Long Memory & Early Halting.

    Args:
        predictor       (SAM2VideoPredictor): Built and initialised predictor.
        state           (dict):               Inference state from init_state.
        start_z         (int):                Z-index of the prompt slice.
        bbox            (np.ndarray):         float32 [x0, y0, x1, y1] bounding box.
        target_hw       (tuple | None):       (H, W) to resize masks to.
        min_area_pixels (int):                Minimum mask pixel count threshold for early halting.
        k_slices        (int):                Short-term memory window size.

    Returns:
        np.ndarray: bool array of shape (H, W, Z) — the full 3-D organ mask.
    """
    predictor.add_new_points_or_box(
        state,
        frame_idx=start_z,
        obj_id=1,
        box=bbox,
    )

    memory_bank = ShortLongMemoryBank(k_slices=k_slices)
    memory_bank.register_prompt(start_z, bbox)

    slice_masks = {}
    num_frames = state["num_frames"] if "num_frames" in state else len(state.get("images", []))
    if num_frames == 0:
        # Fallback in case state uses video_height / num_frames differently
        num_frames = getattr(state, "num_frames", 0)

    def process_and_store_mask(frame_idx, mask_tensor):
        """Helper to threshold, optionally resize, and store mask in slice_masks."""
        mask_bool = (mask_tensor > 0).cpu().numpy()  # shape: (h, w)

        if target_hw is not None and mask_bool.shape != target_hw:
            target_h, target_w = target_hw
            img = Image.fromarray(mask_bool)
            resized_img = img.resize((target_w, target_h), resample=Image.NEAREST)
            mask_bool = np.array(resized_img, dtype=bool)

        slice_masks[frame_idx] = mask_bool
        memory_bank.add_slice_mask(frame_idx, mask_bool)
        return mask_bool

    # Step 1: Forward Pass (start_z -> top slice)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
        state, start_frame_idx=start_z, reverse=False
    ):
        mask_bool = process_and_store_mask(frame_idx, masks[0, 0])
        # Early halting check: if organ cross-section drops below pixel threshold
        if frame_idx != start_z and mask_bool.sum() < min_area_pixels:
            print(
                f"  Forward propagation halted early at slice z={frame_idx} "
                f"(area {mask_bool.sum()} < threshold {min_area_pixels} px)"
            )
            break

    # Reset short-term memory for backward pass while maintaining long-term prompt anchor
    memory_bank.reset_short_term()

    # Step 2: Backward Pass (start_z -> slice 0)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
        state, start_frame_idx=start_z, reverse=True
    ):
        if frame_idx not in slice_masks:
            mask_bool = process_and_store_mask(frame_idx, masks[0, 0])
            # Early halting check
            if frame_idx != start_z and mask_bool.sum() < min_area_pixels:
                print(
                    f"  Backward propagation halted early at slice z={frame_idx} "
                    f"(area {mask_bool.sum()} < threshold {min_area_pixels} px)"
                )
                break

    # Determine spatial dimensions from any valid mask or target_hw
    if target_hw is not None:
        H, W = target_hw
    elif len(slice_masks) > 0:
        sample_mask = next(iter(slice_masks.values()))
        H, W = sample_mask.shape
    else:
        H, W = 512, 512

    # Determine total Z depth
    max_z = max(slice_masks.keys()) if len(slice_masks) > 0 else 0
    total_z = max(num_frames, max_z + 1)

    # Build full (H, W, Z) volume, filling unreached/halted slices with empty masks
    empty_mask = np.zeros((H, W), dtype=bool)
    sorted_frames = [slice_masks.get(z, empty_mask) for z in range(total_z)]
    mask_3d = np.stack(sorted_frames, axis=2)

    return mask_3d