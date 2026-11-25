"""Manifest and image interfaces for four-class, patient-disjoint TLS tiles.

Images must already be 224 by 224 pixels. This module never resizes images or
masks and does not infer a physical pixel scale from their pixel dimensions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

CLASS_NAMES = ("Background", "Agg", "FL1", "FL2")
IGNORE_INDEX = 255
IMAGE_SIZE = 224
_SPLIT_ALIASES = {
    "training": "train", "validation": "val", "valid": "val",
    "testing": "test", "prediction": "predict", "inference": "predict",
}
_SPLITS = ("train", "val", "test", "predict")


def _split(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Each record requires a split string.")
    value = value.strip().lower()
    value = _SPLIT_ALIASES.get(value, value)
    if value not in _SPLITS:
        raise ValueError(f"Unsupported split: {value!r}.")
    return value


def _path(value: Any, base: Path, field: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Each record requires a nonempty {field}.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _normalise_record(raw: Mapping[str, Any], index: int, base: Path) -> dict:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Manifest record {index} is not an object.")
    row = dict(raw)
    patient = row.get("patient_id", row.get("patient_group"))
    if patient is None or not str(patient).strip():
        raise ValueError(f"Manifest record {index} has no patient_id.")
    row["patient_id"] = str(patient).strip()
    row["split"] = _split(row.get("split", row.get("set")))
    image = row.get("image_path") or row.get("image") or row.get("rgb_path")
    row["image_path"] = _path(image, base, "image_path")
    mask = row.get("mask_path") or row.get("mask") or row.get("label_path")
    if mask is not None and str(mask).strip():
        row["mask_path"] = _path(mask, base, "mask_path")
    else:
        row.pop("mask_path", None)
    tile = row.get("tile_id", row.get("sample_id"))
    row["tile_id"] = str(index if tile is None or not str(tile).strip() else tile).strip()
    return row


def load_manifest(path: str | Path) -> list[dict]:
    """Read CSV/JSON records and resolve paths relative to the manifest.

    JSON can contain a record list, ``{"records": [...]}``, or split lists under
    ``splits``/the root. An unlabelled prediction record omits ``mask_path`` and
    can use ``split="predict"``. Split validation is a separate explicit call.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV manifest has no header.")
            records = [row for row in reader if any(str(v or "").strip() for v in row.values())]
    else:
        with path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and "records" in payload:
            records = payload["records"]
        elif isinstance(payload, dict):
            groups = payload.get("splits", payload)
            if not isinstance(groups, dict):
                raise ValueError("JSON splits must be an object of record lists.")
            records = []
            for name, rows in groups.items():
                if name not in _SPLITS and name not in _SPLIT_ALIASES:
                    continue
                split = _split(name)
                if not isinstance(rows, list):
                    raise ValueError(f"The {name!r} split must be a list.")
                for raw in rows:
                    if not isinstance(raw, Mapping):
                        raise ValueError("Every split entry must be an object.")
                    row = dict(raw)
                    if "split" in row and _split(row["split"]) != split:
                        raise ValueError("Record split conflicts with its enclosing split.")
                    row["split"] = split
                    records.append(row)
        else:
            raise ValueError("JSON manifest must contain records or split lists.")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest has no records.")
    return [_normalise_record(row, index, path.parent) for index, row in enumerate(records)]


def validate_patient_splits(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Reject duplicate tile images/IDs and a patient appearing in two splits.

    The returned summary contains counts and a relocatable split fingerprint,
    rather than exporting the patient identifiers themselves. Nonempty training
    and validation splits, when needed, are enforced by the training entry point.
    """
    if not rows:
        raise ValueError("Manifest has no records.")
    patient_split: dict[str, str] = {}
    patients = {name: set() for name in _SPLITS}
    counts: Counter = Counter()
    labelled: Counter = Counter()
    images, tiles, identities = set(), set(), []
    for index, raw in enumerate(rows):
        row = _normalise_record(raw, index, Path.cwd())
        patient, split = row["patient_id"], row["split"]
        if patient in patient_split and patient_split[patient] != split:
            raise ValueError("Patient leakage: the same patient occurs in multiple splits.")
        patient_split[patient] = split
        if row["image_path"] in images:
            raise ValueError("Duplicate tile image in manifest.")
        if row["tile_id"] in tiles:
            raise ValueError("Duplicate tile_id in manifest.")
        images.add(row["image_path"])
        tiles.add(row["tile_id"])
        patients[split].add(patient)
        counts[split] += 1
        labelled[split] += int(bool(row.get("mask_path")))
        identities.append((patient, split, row["tile_id"]))
    fingerprint = hashlib.sha256(
        json.dumps(sorted(identities), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "num_tiles": len(rows), "num_patients": len(patient_split),
        "tile_counts": {name: counts[name] for name in _SPLITS},
        "patient_counts": {name: len(patients[name]) for name in _SPLITS},
        "labeled_tile_counts": {name: labelled[name] for name in _SPLITS},
        "patient_disjoint": True, "split_fingerprint": fingerprint,
    }


class TileDataset(Dataset):
    """Strict RGB/index-mask tiles with aligned optional training augmentation."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], augment: bool = False):
        self.rows = [dict(row) for row in rows]
        self.records = self.rows
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = Path(row["image_path"]).expanduser().resolve()
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected a prebuilt {IMAGE_SIZE}x{IMAGE_SIZE} image; resizing is not implicit.")
        mask = None
        if row.get("mask_path"):
            with Image.open(row["mask_path"]) as source:
                mask = np.array(source)
            if mask.ndim != 2 or mask.dtype.kind not in "uib":
                raise ValueError("Mask must contain 2D integer class indices, not RGB colors.")
            if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError("Image and mask geometry must both be 224x224.")
            if not np.isin(mask, (0, 1, 2, 3, IGNORE_INDEX)).all():
                raise ValueError("Mask labels must be 0, 1, 2, 3 or 255 (ignore).")
        if self.augment:
            turns = random.randrange(4)
            image = image.rotate(90 * turns, expand=False)
            if mask is not None:
                mask = np.rot90(mask, turns)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if mask is not None:
                    mask = np.fliplr(mask)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                if mask is not None:
                    mask = np.flipud(mask)
            for operation in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = operation(image).enhance(random.uniform(0.8, 1.2))
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        result = {
            "image": torch.from_numpy(np.ascontiguousarray((pixels - mean) / std)),
            "tile_id": str(row.get("tile_id", index)), "path": str(path),
        }
        if mask is not None:
            result["mask"] = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64))
        return result
