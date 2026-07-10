# [Week 4] src/train/losses.py
import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    """Soft Dice loss for binary segmentation.

    Specification:
    - Convert logits to probabilities: p = torch.sigmoid(logits).
    - Soft intersection:  inter = (p * targets).sum()
    - Soft union:         union = p.sum() + targets.sum()
    - Dice coefficient:   (2 * inter + eps) / (union + eps)
    - Return 1 − dice_coefficient  (0 = perfect, 1 = worst)
    - Result must be a scalar tensor with a gradient.

    Args:
        logits  (Tensor): Raw model output, any shape (NOT sigmoid-activated).
        targets (Tensor): Binary float32 ground truth, same shape as logits.
        eps     (float):  Smoothing constant to avoid division by zero.

    Returns:
        Tensor: Scalar soft Dice loss with gradient.
    """
    probs = logits.sigmoid()
    intersection = torch.sum(probs * targets)
    cardinality = torch.sum(probs) + torch.sum(targets)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice


def total_loss(logits, targets):
    """Combined Dice + Binary Cross-Entropy loss for organ segmentation.

    Specification:
    - loss = dice_loss(logits, targets)
           + F.binary_cross_entropy_with_logits(logits, targets)
    - Dice drives spatial overlap; BCE drives per-pixel calibration.
    - Return the sum as a scalar tensor.

    Args:
        logits  (Tensor): Raw model output (NOT sigmoid-activated).
        targets (Tensor): Binary float32 ground truth, same shape as logits.

    Returns:
        Tensor: Scalar combined loss with gradient.
    """
    seg_dice = dice_loss(logits, targets)
    seg_bce = F.binary_cross_entropy_with_logits(logits, targets)
    return seg_dice + seg_bce