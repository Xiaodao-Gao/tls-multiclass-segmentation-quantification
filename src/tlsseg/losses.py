"""Cross entropy, mean foreground soft Dice and feature-noise MSE."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def segmentation_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    noise_mse: torch.Tensor | float,
    diffusion_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Use classes 0..3, ignore 255, and average soft Dice over classes 1..3.

    A wholly ignored batch returns a differentiable zero total (including no
    auxiliary update), matching the original training helper's skip policy.
    """
    if logits.ndim != 4 or logits.shape[1] != 4 or logits.shape[0] < 1:
        raise ValueError("Expected nonempty Bx4xHxW segmentation logits.")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Segmentation logits must be finite floating-point values.")
    if mask.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("Mask shape must match BxHxW logits geometry.")
    if mask.is_floating_point() or mask.is_complex():
        raise ValueError("Mask must contain integer class indices.")
    mask = mask.to(device=logits.device, dtype=torch.long)
    valid = mask != 255
    if ((mask[valid] < 0) | (mask[valid] > 3)).any():
        raise ValueError("Mask labels must be 0, 1, 2, 3 or 255.")
    weight = float(diffusion_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("diffusion_weight must be finite and nonnegative.")
    noise = torch.as_tensor(noise_mse, device=logits.device, dtype=logits.dtype)
    if noise.ndim != 0 or not torch.isfinite(noise) or noise < 0:
        raise ValueError("noise_mse must be a finite nonnegative scalar.")
    if not valid.any():
        zero = logits.sum() * 0.0
        return zero, {"ce": zero, "dice": zero, "noise_mse": noise}
    ce = F.cross_entropy(logits, mask, ignore_index=255)
    probability = logits.softmax(1) * valid[:, None]
    safe = mask.masked_fill(~valid, 0)
    truth = F.one_hot(safe, num_classes=4).permute(0, 3, 1, 2).to(logits.dtype)
    truth = truth * valid[:, None]
    axes = (0, 2, 3)
    intersection = (probability * truth).sum(axes)
    denominator = probability.sum(axes) + truth.sum(axes)
    soft_dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1.0 - soft_dice[1:].mean()
    total = ce + dice_loss + weight * noise
    return total, {"ce": ce, "dice": dice_loss, "noise_mse": noise}
