"""Tile probability export and explicit coordinate-based probability fusion."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .coordinates import OutputGrid, finite_number, integer_number
from .data import TileDataset, load_manifest, validate_patient_splits
from .runtime import atomic_json, configure_precision, environment, load_checkpoint, select_device, sha256, source_hashes
from .stitching import ProbabilityStitcher

PIXEL_CLASS_NAMES = {
    "0": "background", "1": "lymphocyte_aggregate",
    "2": "primary_follicle_like", "3": "secondary_follicle_like",
}
_SPATIAL_FIELDS = (
    "slide_id", "base_x", "base_y", "level", "downsample_x", "downsample_y",
    "grid_x", "grid_y", "grid_height", "grid_width", "grid_origin_base_x",
    "grid_origin_base_y", "mpp_x", "mpp_y", "tissue_mask_path",
    "level_downsample", "source_read_size",
)


def _present(row, name):
    return name in row and row[name] is not None and str(row[name]).strip() != ""


def _valid_mask_metadata(row, base):
    if not _present(row, "valid_mask_path"):
        return None
    path = Path(row["valid_mask_path"]).expanduser()
    path = (path if path.is_absolute() else base / path).resolve()
    if not path.is_file():
        raise ValueError("Tile valid mask does not exist.")
    mask = _array(path)
    if mask.ndim != 2 or mask.shape != (224, 224) or not np.isin(mask, (0, 1, 255)).all():
        raise ValueError("Tile valid mask must be a 224x224 scalar binary mask.")
    return {"valid_mask_path": str(path)}


def prepare_spatial_metadata(rows, manifest_path):
    """Validate optional registered-tile metadata before prediction writes files.

    Unregistered manifests remain valid for tile-only prediction. Registered
    manifests require a slide ID, grid dimensions, explicit x/y scale and either
    a complete base-coordinate pair or a complete grid-coordinate pair for
    every selected row. The default basic origin (0,0) means the full-slide
    basic coordinate system; cropped grids must declare their origin.
    A tile-only valid mask does not imply registration or define coordinates.
    """
    base = Path(manifest_path).expanduser().resolve().parent
    has_spatial = [any(_present(row, key) for key in _SPATIAL_FIELDS) for row in rows]
    if not any(has_spatial):
        return [_valid_mask_metadata(row, base) for row in rows]
    if not all(has_spatial):
        raise ValueError("Prediction rows cannot mix registered and unregistered spatial metadata.")
    result, slide_contracts = [], {}
    for row in rows:
        required = ("slide_id", "grid_height", "grid_width", "downsample_x", "downsample_y")
        if any(not _present(row, key) for key in required):
            raise ValueError("Registered tiles require slide_id, grid dimensions and both downsample factors.")
        item = {"slide_id": str(row["slide_id"]).strip()}
        grid = OutputGrid(
            row["grid_height"], row["grid_width"], row["downsample_x"], row["downsample_y"],
            row.get("grid_origin_base_x") if _present(row, "grid_origin_base_x") else 0.0,
            row.get("grid_origin_base_y") if _present(row, "grid_origin_base_y") else 0.0,
        )
        item.update(grid_height=grid.height, grid_width=grid.width,
                    downsample_x=grid.downsample_x, downsample_y=grid.downsample_y,
                    grid_origin_base_x=grid.origin_base_x, grid_origin_base_y=grid.origin_base_y)
        base_pair = tuple(_present(row, key) for key in ("base_x", "base_y"))
        grid_pair = tuple(_present(row, key) for key in ("grid_x", "grid_y"))
        if any(base_pair) != all(base_pair) or any(grid_pair) != all(grid_pair):
            raise ValueError("Both x and y are required for each coordinate pair.")
        if not all(base_pair) and not all(grid_pair):
            raise ValueError("Registered tiles require base_x/base_y or grid_x/grid_y.")
        if all(base_pair):
            bx, by = finite_number(row["base_x"], "base_x"), finite_number(row["base_y"], "base_y")
            gx, gy = grid.base_to_grid(bx, by)
            if all(grid_pair) and (gx, gy) != (integer_number(row["grid_x"], "grid_x"), integer_number(row["grid_y"], "grid_y")):
                raise ValueError("Basic and output-grid tile coordinates disagree.")
        else:
            gx, gy = integer_number(row["grid_x"], "grid_x"), integer_number(row["grid_y"], "grid_y")
            bx, by = grid.grid_to_base(gx, gy)
        if gx >= grid.width or gy >= grid.height or gx + 224 <= 0 or gy + 224 <= 0:
            raise ValueError("Registered tile does not intersect the output grid.")
        item.update(base_x=bx, base_y=by, grid_x=gx, grid_y=gy)
        if _present(row, "level"):
            item["level"] = integer_number(row["level"], "level", nonnegative=True)
        if _present(row, "level_downsample"):
            item["level_downsample"] = finite_number(row["level_downsample"], "level_downsample", positive=True)
        if _present(row, "source_read_size"):
            item["source_read_size"] = integer_number(row["source_read_size"], "source_read_size", positive=True)
        if "level_downsample" in item and "source_read_size" in item:
            expected = item["level_downsample"] * item["source_read_size"] / 224
            if not all(math.isclose(item[key], expected, rel_tol=1e-9, abs_tol=1e-9)
                       for key in ("downsample_x", "downsample_y")):
                raise ValueError("Output-grid scale does not include the declared reading level and image resizing.")
        for key in ("mpp_x", "mpp_y"):
            if _present(row, key):
                item[key] = finite_number(row[key], key, positive=True)
        valid_metadata = _valid_mask_metadata(row, base)
        if valid_metadata is not None:
            item.update(valid_metadata)
        for key in ("tissue_mask_path",):
            if _present(row, key):
                path = Path(row[key]).expanduser()
                path = path if path.is_absolute() else base / path
                path = path.resolve()
                if not path.is_file():
                    raise ValueError(f"Registered mask does not exist: {key}.")
                item[key] = str(path)
        contract_keys = ("grid_height", "grid_width", "downsample_x", "downsample_y",
                         "grid_origin_base_x", "grid_origin_base_y", "mpp_x", "mpp_y", "tissue_mask_path")
        contract = {key: item.get(key) for key in contract_keys}
        contract["patient_id"] = str(row["patient_id"]).strip() if _present(row, "patient_id") else None
        prior = slide_contracts.setdefault(item["slide_id"], contract)
        if prior != contract:
            raise ValueError("Tiles from the same slide disagree on patient identity, grid, scale, tissue mask or physical calibration.")
        result.append(item)
    return result


def prediction_layouts(spatial_metadata, index):
    """Build reusable per-slide layouts in the prediction output directory."""
    if len(spatial_metadata) != len(index):
        raise ValueError("Spatial metadata and prediction counts differ.")
    slides = {}
    for spatial, prediction in zip(spatial_metadata, index):
        if spatial is None or "slide_id" not in spatial:
            continue
        slide_id = spatial["slide_id"]
        if slide_id not in slides:
            metadata = {key: spatial[key] for key in ("downsample_x", "downsample_y", "mpp_x", "mpp_y") if key in spatial}
            slides[slide_id] = {
                "slide_id": slide_id, "shape": [spatial["grid_height"], spatial["grid_width"]],
                "grid_origin_base": [spatial["grid_origin_base_x"], spatial["grid_origin_base_y"]],
                "metadata": metadata, "tiles": [],
                "coordinate_system": "Tile base coordinates are level-0 top-left positions; x/y are output-grid pixels.",
            }
            if "tissue_mask_path" in spatial:
                slides[slide_id]["tissue_mask"] = spatial["tissue_mask_path"]
        tile = {"probability_path": prediction["probability_path"], "x": spatial["grid_x"],
                "y": spatial["grid_y"], "base_x": spatial["base_x"], "base_y": spatial["base_y"]}
        for key in ("valid_mask_path", "level", "level_downsample", "source_read_size"):
            if key in spatial:
                tile[key] = spatial[key]
        slides[slide_id]["tiles"].append(tile)
    return list(slides.values())


def predict(args):
    configure_precision()
    device = select_device(args.device)
    rows = load_manifest(args.manifest)
    validate_patient_splits(rows)
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError("No input tiles selected.")
    spatial_metadata = prepare_spatial_metadata(rows, args.manifest)
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
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for batch in loader:
            probabilities = model(batch["image"].to(device))["logits"].softmax(1).cpu().numpy()
            for offset, probability in enumerate(probabilities):
                stem = f"tile_{len(index):06d}"
                np.savez_compressed(output / f"{stem}.npz", probabilities=probability)
                Image.fromarray(probability.argmax(0).astype(np.uint8)).save(output / f"{stem}.png")
                item = {"tile_id": batch["tile_id"][offset],
                        "probability_path": f"{stem}.npz", "mask_path": f"{stem}.png"}
                if spatial_metadata[len(index)] is not None:
                    item.update(spatial_metadata[len(index)])
                index.append(item)
    layouts = []
    for number, layout in enumerate(prediction_layouts(spatial_metadata, index)):
        filename = f"layout_{number:04d}.json"
        atomic_json(output / filename, layout)
        layouts.append({"slide_id": layout["slide_id"], "layout_path": filename})
    atomic_json(output / "index.json", {
        "tiles": index, "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(args.manifest), "checkpoint_epoch": checkpoint["epoch"],
        "architecture": model.architecture_config(),
        "probability_axes": "class,y,x", "class_ids": [0, 1, 2, 3],
        "class_names": PIXEL_CLASS_NAMES, "classification_scope": "pixel",
        "prediction_task": "four_class_semantic_segmentation", "stitch_layouts": layouts,
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


def layout_tile_position(layout, tile):
    """Resolve an explicit grid position or validate/map basic-level origins."""
    grid_pair = tuple(_present(tile, key) for key in ("x", "y"))
    base_pair = tuple(_present(tile, key) for key in ("base_x", "base_y"))
    if any(grid_pair) != all(grid_pair) or any(base_pair) != all(base_pair):
        raise ValueError("Layout tiles require both x/y or both base_x/base_y.")
    if not all(grid_pair) and not all(base_pair):
        raise ValueError("Layout tile has no coordinate pair.")
    position = None
    if all(grid_pair):
        position = integer_number(tile["x"], "x"), integer_number(tile["y"], "y")
    if all(base_pair):
        metadata = layout.get("metadata")
        if not isinstance(metadata, dict) or not all(_present(metadata, key) for key in ("downsample_x", "downsample_y")):
            raise ValueError("Basic-level tile coordinates require both output-grid downsample factors.")
        origin = layout.get("grid_origin_base", [0.0, 0.0])
        if not isinstance(origin, (list, tuple)) or len(origin) != 2:
            raise ValueError("grid_origin_base must be [base_x,base_y].")
        shape = layout.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("Layout shape must be [height,width].")
        grid = OutputGrid(shape[0], shape[1], metadata["downsample_x"], metadata["downsample_y"], *origin)
        mapped = grid.base_to_grid(tile["base_x"], tile["base_y"])
        if position is not None and position != mapped:
            raise ValueError("Basic and output-grid layout coordinates disagree.")
        position = mapped
    return position


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
    positions = [layout_tile_position(layout, tile) for tile in layout["tiles"]]
    input_hashes = []
    for tile, (x, y) in zip(layout["tiles"], positions):
        probability_path = resolve(tile["probability_path"])
        probability = _array(probability_path, "probabilities")
        input_hashes.append(sha256(probability_path))
        weights = _array(resolve(tile["weights_path"])) if tile.get("weights_path") else None
        valid = _array(resolve(tile["valid_mask_path"])) if tile.get("valid_mask_path") else None
        if valid is not None:
            valid = valid > 0
        stitcher.add_tile(probability, x, y, weights=weights, valid_mask=valid)
    result = stitcher.finalize(layout.get("metadata"))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Stitch output must be a new or empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "fused.npz", probabilities=result.probabilities,
                        prediction=result.prediction, coverage=result.coverage)
    atomic_json(output / "quantification.json", dict(result.summary,
        layout_sha256=sha256(path), input_probability_sha256=input_hashes,
        classification_scope="pixel", class_names=PIXEL_CLASS_NAMES))
    print(json.dumps({"output": str(output), "layout_sha256": sha256(path)}), flush=True)
