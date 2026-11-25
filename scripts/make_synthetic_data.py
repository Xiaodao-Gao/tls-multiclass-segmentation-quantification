"""Generate artificial images/masks to check software; not a biological dataset."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def make_dataset(output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    if manifest.exists():
        raise ValueError("Synthetic manifest already exists; select a new directory.")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:224, :224]
    palette = np.array([[235, 222, 230], [115, 75, 150], [190, 100, 145], [135, 120, 210]])
    records = []
    for split, count in (("train", 8), ("val", 2), ("test", 2)):
        for i in range(count):
            tile_id = f"synthetic_{split}_{i}"
            mask = np.zeros((224, 224), dtype=np.uint8)
            for cls, (cx, cy) in enumerate(((55, 60), (165, 65), (115, 165)), start=1):
                cx += int(rng.integers(-8, 9)); cy += int(rng.integers(-8, 9))
                mask[(xx - cx) ** 2 + (yy - cy) ** 2 < (25 + cls * 3) ** 2] = cls
            rgb = np.clip(palette[mask] + rng.normal(0, 10, (224, 224, 3)), 0, 255).astype(np.uint8)
            mask[:3] = 255
            Image.fromarray(rgb).save(output / f"{tile_id}.png")
            Image.fromarray(mask).save(output / f"{tile_id}_mask.png")
            records.append(dict(patient_id=f"synthetic_{split}_patient_{i}", split=split,
                                image_path=f"{tile_id}.png", mask_path=f"{tile_id}_mask.png", tile_id=tile_id))
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    print(make_dataset(arguments.output, arguments.seed))
