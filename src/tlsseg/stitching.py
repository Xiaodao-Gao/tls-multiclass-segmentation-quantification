"""In-memory fusion and area summaries on an explicitly defined output grid.

Inputs are four-channel probabilities, not logits or hard class labels. Tile
origins x/y are output-grid pixel coordinates; no WSI level or resize transform
is inferred. This module does not read slides or identify independent biological
structures from connected components.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

CLASS_NAMES = ("Background", "Agg", "FL1", "FL2")
_PHYSICAL_KEYS = ("mpp_x", "mpp_y", "downsample_x", "downsample_y")


def _integer(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _mask(value, shape, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != tuple(shape) or not np.isin(result, (0, 1)).all():
        raise ValueError(f"{name} must be a binary mask with shape {tuple(shape)}.")
    return result.astype(bool, copy=True)


def _shape(value, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise ValueError(f"{name} must be (height, width).")
    try:
        length = len(value)
    except TypeError:
        raise ValueError(f"{name} must be (height, width).") from None
    if length != 2:
        raise ValueError(f"{name} must be (height, width).")
    return (_integer(value[0], "height", positive=True), _integer(value[1], "width", positive=True))


def truncated_hann_weights(shape: tuple[int, int], minimum: float = 0.05) -> np.ndarray:
    """Outer product of two Hann windows, each clipped below by ``minimum``."""
    height, width = _shape(shape, "Weight shape")
    minimum = float(minimum)
    if not math.isfinite(minimum) or not 0 < minimum <= 1:
        raise ValueError("minimum must be in (0, 1].")
    y = np.ones(1) if height == 1 else np.maximum(np.hanning(height), minimum)
    x = np.ones(1) if width == 1 else np.maximum(np.hanning(width), minimum)
    return np.outer(y, x).astype(np.float32)


@dataclass(frozen=True)
class StitchResult:
    probabilities: np.ndarray
    prediction: np.ndarray
    coverage: np.ndarray
    summary: dict


class ProbabilityStitcher:
    """Accumulate probabilities in a fixed HxW grid, then classify each pixel once."""

    def __init__(self, shape: tuple[int, int], tissue_mask: np.ndarray | None = None):
        self.shape = _shape(shape, "Output-grid shape")
        self.tissue_mask_provided = tissue_mask is not None
        self.tissue_mask = np.ones(self.shape, dtype=bool) if tissue_mask is None else _mask(tissue_mask, self.shape, "tissue_mask")
        self._probability_sum = np.zeros((4, *self.shape), dtype=np.float32)
        self._weight_sum = np.zeros(self.shape, dtype=np.float32)
        self.tiles_added = 0

    def add_tile(
        self,
        probabilities: np.ndarray,
        x: int,
        y: int,
        weights: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        """Add CxHxW probabilities; partially out-of-grid tiles are clipped.

        ``valid_mask`` excludes padding/invalid tile pixels. Tissue exclusion is
        applied in grid coordinates. Neither exclusion counts as background.
        """
        x, y = _integer(x, "x"), _integer(y, "y")
        probability = np.asarray(probabilities, dtype=np.float32)
        if probability.ndim != 3 or probability.shape[0] != 4 or min(probability.shape[1:]) < 1:
            raise ValueError("Expected four-channel probabilities with shape (4, height, width).")
        if not np.isfinite(probability).all() or (probability < 0).any():
            raise ValueError("Probabilities must be finite and nonnegative.")
        height, width = probability.shape[1:]
        valid = np.ones((height, width), dtype=bool) if valid_mask is None else _mask(valid_mask, (height, width), "valid_mask")
        if weights is None:
            weight = truncated_hann_weights((height, width))
        else:
            weight = np.asarray(weights, dtype=np.float32)
            if weight.shape != (height, width) or not np.isfinite(weight).all() or (weight < 0).any():
                raise ValueError("weights must be finite nonnegative tile-sized values.")
        contributing = valid & (weight > 0)
        channel_sum = probability.sum(0)
        if not np.allclose(channel_sum[contributing], 1.0, atol=1e-4, rtol=0):
            raise ValueError("Input must contain normalized class probabilities, not logits or labels.")
        left, top = max(0, x), max(0, y)
        right, bottom = min(self.shape[1], x + width), min(self.shape[0], y + height)
        if left >= right or top >= bottom:
            raise ValueError("Tile does not intersect the output grid.")
        src = (slice(top - y, bottom - y), slice(left - x, right - x))
        dst = (slice(top, bottom), slice(left, right))
        effective = weight[src] * valid[src] * self.tissue_mask[dst]
        self._probability_sum[(slice(None), *dst)] += probability[(slice(None), *src)] * effective[None]
        self._weight_sum[dst] += effective
        self.tiles_added += 1

    def finalize(self, metadata: Mapping | None = None) -> StitchResult:
        """Fuse before argmax and quantify unique covered tissue-grid pixels.

        mm² is reported only if mpp_x/mpp_y (micrometres per base pixel) and
        downsample_x/downsample_y (base pixels per output-grid pixel, including
        any subsequent resizing) are all explicitly supplied and positive.
        Missing metadata never defaults to a physical scale.
        """
        coverage = (self._weight_sum > 0) & self.tissue_mask
        covered = int(coverage.sum())
        if covered == 0:
            raise ValueError("No covered tissue pixels are available for quantification.")
        probability = np.zeros_like(self._probability_sum)
        np.divide(self._probability_sum, self._weight_sum[None], out=probability, where=coverage[None])
        # Correct floating-point roundoff without changing the weighted mixture.
        normalizer = probability.sum(0)
        np.divide(probability, normalizer[None], out=probability, where=coverage[None])
        prediction = np.full(self.shape, -1, dtype=np.int16)
        prediction[coverage] = probability[:, coverage].argmax(0)
        counts = np.bincount(prediction[coverage], minlength=4)
        class_counts = {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)}
        foreground = int(counts[1:].sum())
        total_tissue = int(self.tissue_mask.sum())
        summary = {
            "grid_shape": list(self.shape), "tiles_added": self.tiles_added,
            "tissue_mask_provided": self.tissue_mask_provided,
            "class_pixel_counts": class_counts, "foreground_pixel_count": foreground,
            "covered_tissue_pixels": covered, "total_tissue_pixels": total_tissue,
            "tissue_coverage_fraction": covered / total_tissue,
            "class_area_fractions": {name: count / covered for name, count in class_counts.items()},
            "foreground_area_fraction": foreground / covered,
            "area_fraction_denominator": (
                "Covered eligible tissue-mask pixels on the unique output grid."
                if self.tissue_mask_provided else
                "Covered output-grid pixels; the caller declares the entire grid eligible because no tissue mask was supplied."
            ),
            "count_interpretation": "Unique output-grid pixels, not independent pathological structure counts.",
            "physical_area_available": False,
        }
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping of physical scale fields.")
        values = {} if metadata is None else dict(metadata)
        scale = {}
        for key in _PHYSICAL_KEYS:
            if key not in values or values[key] is None:
                continue
            value = values[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise ValueError(f"{key} must be a positive finite number.")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a positive finite number.")
            scale[key] = value
        missing = [key for key in _PHYSICAL_KEYS if key not in values or values[key] is None]
        if missing:
            summary["missing_physical_metadata"] = missing
        else:
            pixel_area = math.prod(scale.values()) * 1e-6
            if not math.isfinite(pixel_area) or pixel_area <= 0:
                raise ValueError("Physical pixel-area conversion is not finite and positive.")
            summary.update({
                "physical_area_available": True, "physical_metadata": scale,
                "output_pixel_area_mm2": pixel_area,
                "class_area_mm2": {name: count * pixel_area for name, count in class_counts.items()},
                "foreground_area_mm2": foreground * pixel_area,
                "covered_tissue_area_mm2": covered * pixel_area,
                "total_tissue_area_mm2": total_tissue * pixel_area,
            })
        return StitchResult(probability, prediction, coverage.copy(), summary)
