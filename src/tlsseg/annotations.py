"""Rasterize reviewed GeoJSON polygons in full-resolution slide coordinates.

Pixel labels are sampled at output-pixel centers. Points on polygon boundaries
are excluded. Polygon holes therefore retain any underlying reviewed background
label, or the unannotated value. Conflicting foreground labels and explicit
ignore regions are always encoded as 255.
"""
from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_LABELS = frozenset((0, 1, 2, 3, 255))


def _shapely():
    try:
        import shapely
        from shapely.geometry import shape
    except ImportError as error:
        raise ImportError(
            "GeoJSON preprocessing requires Shapely; install this package with the preprocessing extra."
        ) from error
    return shapely, shape


def _label(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or int(value) not in _LABELS:
        raise ValueError(f"{name} must be an integer label in 0, 1, 2, 3 or 255.")
    return int(value)


@dataclass(frozen=True)
class RegionAnnotation:
    """One valid Polygon/MultiPolygon and its reviewed class label."""

    geometry: Any
    class_id: int


def load_annotations(path: str | Path, class_map: Mapping[str, int]) -> list[RegionAnnotation]:
    """Read QuPath GeoJSON with explicit reviewed class-name mapping.

    Input can be a FeatureCollection, a Feature or a list of Features. A feature
    has either an integer ``properties.class_id`` or the QuPath field
    ``properties.classification.name``. If both are present they must agree.
    Unknown names, invalid polygons, nonpolygon geometries and nonfinite
    coordinates are rejected rather than silently repaired or relabelled.
    Coordinates must be in level-0 slide pixels with a top-left origin.
    """
    if not isinstance(class_map, Mapping):
        raise ValueError("class_map must map annotation class names to integer labels.")
    names = {}
    for name, value in class_map.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Annotation class names must be nonempty strings.")
        if name.strip() in names:
            raise ValueError("Annotation class names must be unique after whitespace normalization.")
        names[name.strip()] = _label(value, "Class-map value")
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        features = payload
    elif isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = payload.get("features")
    elif isinstance(payload, dict) and payload.get("type") == "Feature":
        features = [payload]
    else:
        raise ValueError("GeoJSON must be a FeatureCollection, a Feature or a list of Features.")
    if not isinstance(features, list):
        raise ValueError("GeoJSON features must be a list.")
    shapely, geometry_shape = _shapely()
    result = []
    for index, feature in enumerate(features):
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            raise ValueError(f"GeoJSON entry {index} is not a Feature.")
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError(f"Feature {index} requires a properties object.")
        explicit = _label(properties["class_id"], "Feature class_id") if "class_id" in properties else None
        classification = properties.get("classification")
        named = None
        if classification is not None:
            if not isinstance(classification, Mapping) or not isinstance(classification.get("name"), str):
                raise ValueError(f"Feature {index} classification must contain a class name.")
            name = classification["name"].strip()
            if name not in names:
                raise ValueError(f"Feature {index} has an unmapped annotation class: {name!r}.")
            named = names[name]
        if explicit is None and named is None:
            raise ValueError(f"Feature {index} has no class_id or mapped classification name.")
        if explicit is not None and named is not None and explicit != named:
            raise ValueError(f"Feature {index} class_id conflicts with its classification name.")
        raw = feature.get("geometry")
        if not isinstance(raw, Mapping) or raw.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Feature {index} requires Polygon or MultiPolygon geometry.")
        try:
            geometry = geometry_shape(raw)
        except (ValueError, TypeError, KeyError, shapely.errors.GEOSException) as error:
            raise ValueError(f"Feature {index} has invalid polygon coordinates.") from error
        coordinates = shapely.get_coordinates(geometry)
        if geometry.is_empty or not np.isfinite(coordinates).all() or not geometry.is_valid:
            raise ValueError(f"Feature {index} has empty, invalid or nonfinite polygon geometry.")
        result.append(RegionAnnotation(geometry, explicit if explicit is not None else named))
    return result


def _pair(value, name: str, *, positive: bool = False) -> tuple[float, float]:
    try:
        values = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must contain two finite numbers.") from None
    if len(values) != 2:
        raise ValueError(f"{name} must contain two finite numbers.")
    result = []
    for item in values:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real):
            raise ValueError(f"{name} must contain two finite numbers.")
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0):
            raise ValueError(f"{name} values must be finite{' and positive' if positive else ''}.")
        result.append(number)
    return result[0], result[1]


def rasterize_tile(
    annotations: Iterable[RegionAnnotation],
    origin_base: tuple[float, float] = (0, 0),
    downsample: tuple[float, float] = (1, 1),
    shape: tuple[int, int] = (224, 224),
    unannotated_label: int = 255,
) -> np.ndarray:
    """Rasterize at base-coordinate centers ``origin + (index+0.5)*scale``.

    Unannotated pixels default to ignore=255. Setting ``unannotated_label=0``
    explicitly declares all otherwise unannotated pixels reviewed background.
    Explicit background polygons are overwritten by foreground labels; two
    distinct foreground labels become ignore. Explicit ignore always wins.
    Same-class overlaps are a union. Output is an HxW uint8 mask.
    """
    shapely, _ = _shapely()
    x0, y0 = _pair(origin_base, "origin_base")
    dx, dy = _pair(downsample, "downsample", positive=True)
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(
        isinstance(size, (bool, np.bool_)) or not isinstance(size, numbers.Integral) or size <= 0
        for size in shape
    ):
        raise ValueError("shape must contain positive integer height and width.")
    height, width = int(shape[0]), int(shape[1])
    if _label(unannotated_label, "unannotated_label") not in (0, 255):
        raise ValueError("unannotated_label must be 0 (reviewed background) or 255 (unknown).")
    xs = x0 + (np.arange(width, dtype=np.float64) + 0.5) * dx
    ys = y0 + (np.arange(height, dtype=np.float64) + 0.5) * dy
    layers = {label: np.zeros((height, width), dtype=bool) for label in _LABELS}
    for annotation in annotations:
        if not isinstance(annotation, RegionAnnotation):
            raise ValueError("annotations must contain RegionAnnotation objects from load_annotations.")
        label = _label(annotation.class_id, "Annotation class_id")
        geometry = annotation.geometry
        if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty or not geometry.is_valid:
            raise ValueError("Every annotation must contain valid nonempty polygon geometry.")
        left, top, right, bottom = geometry.bounds
        if right < xs[0] or left > xs[-1] or bottom < ys[0] or top > ys[-1]:
            continue
        layers[label] |= shapely.contains_xy(geometry, xs[None, :], ys[:, None])
    result = np.full((height, width), unannotated_label, dtype=np.uint8)
    result[layers[0]] = 0
    for label in (1, 2, 3):
        result[layers[label]] = label
    conflicts = layers[1].astype(np.uint8) + layers[2] + layers[3] > 1
    result[conflicts | layers[255]] = 255
    return result
