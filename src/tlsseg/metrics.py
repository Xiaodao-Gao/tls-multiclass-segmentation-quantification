"""Global pixel-aggregated confusion metrics with explicit absent-class rules."""
from __future__ import annotations

import numbers

import numpy as np
import torch

_TLS_CLASSES = ("Background", "Agg", "FL1", "FL2")


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class ConfusionMatrix:
    """Accumulate rows=truth, columns=prediction; target value 255 is ignored."""

    def __init__(self, num_classes: int = 4, ignore_index: int = 255):
        if isinstance(num_classes, bool) or not isinstance(num_classes, numbers.Integral) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two.")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        if 0 <= self.ignore_index < self.num_classes:
            raise ValueError("ignore_index must not overlap a class index.")
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.matrix.fill(0)

    def update(self, logits_or_prediction, mask) -> None:
        values, target = _numpy(logits_or_prediction), _numpy(mask)
        if target.ndim not in (2, 3) or target.dtype.kind not in "uib":
            raise ValueError("Mask must be integer HxW or BxHxW indices.")
        valid = target != self.ignore_index
        if ((target[valid] < 0) | (target[valid] >= self.num_classes)).any():
            raise ValueError("Mask contains an invalid class index.")
        if values.shape == target.shape:
            if values.dtype.kind not in "uib":
                raise ValueError("Predictions must contain integer class indices.")
            prediction = values
        else:
            channel_axis = 0 if target.ndim == 2 else 1
            expected = list(target.shape)
            expected.insert(channel_axis, self.num_classes)
            if tuple(values.shape) != tuple(expected):
                raise ValueError("Logits/predictions and mask geometry do not match.")
            if not np.isfinite(values).all():
                raise ValueError("Logits must be finite.")
            prediction = values.argmax(axis=channel_axis)
        if ((prediction[valid] < 0) | (prediction[valid] >= self.num_classes)).any():
            raise ValueError("Prediction contains an invalid class index.")
        encoded = target[valid].astype(np.int64) * self.num_classes + prediction[valid].astype(np.int64)
        self.matrix += np.bincount(encoded, minlength=self.num_classes**2).reshape(self.matrix.shape)

    def compute(self) -> dict:
        total = int(self.matrix.sum())
        if total == 0:
            raise ValueError("No valid pixels were evaluated (all pixels are ignored or no batches were added).")
        names = list(_TLS_CLASSES) if self.num_classes == 4 else [f"class_{i}" for i in range(self.num_classes)]
        counts = self.matrix.astype(np.float64)
        true_positive = np.diag(counts)
        support, predicted = counts.sum(1), counts.sum(0)
        per_class = {}
        for index, name in enumerate(names):
            def ratio(numerator, denominator):
                return float(numerator / denominator) if denominator > 0 else None

            tp = true_positive[index]
            per_class[name] = {
                "class_index": index,
                "dice": ratio(2 * tp, support[index] + predicted[index]),
                "iou": ratio(tp, support[index] + predicted[index] - tp),
                "precision": ratio(tp, predicted[index]),
                "recall": ratio(tp, support[index]),
                "gt_pixels": int(support[index]), "predicted_pixels": int(predicted[index]),
            }
        macro = {}
        for metric in ("dice", "iou", "precision", "recall"):
            defined = [per_class[name][metric] for name in names[1:] if per_class[name][metric] is not None]
            macro[metric] = float(np.mean(defined)) if defined else None
        return {
            "confusion_matrix": self.matrix.tolist(), "class_names": names,
            "per_class": per_class, "macro_fg": macro,
            "pixel_accuracy": float(true_positive.sum() / total), "valid_pixels": total,
            "aggregation": "Sum confusion counts over all valid pixels before calculating metrics.",
            "macro_policy": "Background is excluded. Zero-denominator class metrics are null and excluded from that metric's foreground mean.",
        }
