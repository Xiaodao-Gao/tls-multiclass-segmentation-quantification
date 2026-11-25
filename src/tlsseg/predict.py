"""Tile probability export and explicit coordinate-based probability fusion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import TileDataset, load_manifest, validate_patient_splits
from .runtime import atomic_json, environment, load_checkpoint, select_device, sha256, source_hashes
from .stitching import ProbabilityStitcher


def predict(args):
    device = select_device(args.device)
    rows = load_manifest(args.manifest)
    validate_patient_splits(rows)
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError("No input tiles selected.")
    # Prediction does not read annotation masks even when the manifest includes them.
    rows = [{k: v for k, v in r.items() if k != "mask_path"} for r in rows]
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prediction output must be a new or empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()
    loader = DataLoader(TileDataset(rows), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)
    index = []
    with torch.inference_mode():
        for batch in loader:
            probabilities = model(batch["image"].to(device))["logits"].softmax(1).cpu().numpy()
            for offset, probability in enumerate(probabilities):
                stem = f"tile_{len(index):06d}"
                np.savez_compressed(output / f"{stem}.npz", probabilities=probability)
                Image.fromarray(probability.argmax(0).astype(np.uint8)).save(output / f"{stem}.png")
                index.append({"tile_id": batch["tile_id"][offset],
                              "probability_path": f"{stem}.npz", "mask_path": f"{stem}.png"})
    atomic_json(output / "index.json", {
        "tiles": index, "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(args.manifest), "checkpoint_epoch": checkpoint["epoch"],
        "architecture": model.architecture_config(),
        "probability_axes": "class,y,x", "class_ids": [0, 1, 2, 3],
        "inference": "t=0, no sampled noise; original native four-class head",
        "prediction_environment": environment(),
        "source_matches_training": source_hashes() == checkpoint["environment"]["source_sha256"],
    })
    print(json.dumps({"tiles_exported": len(index)}), flush=True)


def _array(path, key=None):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError("Mask/weight NPZ must have exactly one array.")
                key = archive.files[0]
            return archive[key].copy()
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image).copy()


def stitch(args):
    path = Path(args.layout)
    layout = json.loads(path.read_text(encoding="utf-8"))
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else path.parent / p
    tissue = _array(resolve(layout["tissue_mask"])) if layout.get("tissue_mask") else None
    if tissue is not None:
        if tissue.ndim != 2:
            raise ValueError("Tissue mask must be a scalar two-dimensional image/array.")
        tissue = tissue > 0
    stitcher = ProbabilityStitcher(tuple(layout["shape"]), tissue_mask=tissue)
    if not layout.get("tiles"):
        raise ValueError("Layout has no tile predictions.")
    input_hashes = []
    for tile in layout["tiles"]:
        probability_path = resolve(tile["probability_path"])
        probability = _array(probability_path, "probabilities")
        input_hashes.append(sha256(probability_path))
        weights = _array(resolve(tile["weights_path"])) if tile.get("weights_path") else None
        valid = _array(resolve(tile["valid_mask_path"])) if tile.get("valid_mask_path") else None
        if valid is not None:
            valid = valid > 0
        stitcher.add_tile(probability, tile["x"], tile["y"], weights=weights, valid_mask=valid)
    result = stitcher.finalize(layout.get("metadata"))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Stitch output must be a new or empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "fused.npz", probabilities=result.probabilities,
                        prediction=result.prediction, coverage=result.coverage)
    atomic_json(output / "quantification.json", dict(result.summary,
        layout_sha256=sha256(path), input_probability_sha256=input_hashes))
    print(json.dumps({"output": str(output), "layout_sha256": sha256(path)}), flush=True)
