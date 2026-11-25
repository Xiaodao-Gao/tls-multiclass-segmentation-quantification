"""Prepare spatially recorded tiles and four-class labels from annotated slides."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .annotations import load_annotations, rasterize_tile
from .coordinates import OutputGrid
from .data import validate_patient_splits
from .runtime import atomic_json, sha256
from .slides import open_slide
from .splits import split_patients


def _positive(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a positive finite number.") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be a positive finite number.")
    return result


def _resolve(value, base):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def read_cohort(path):
    """CSV/JSON slides with patient_id, slide_id, slide_path and optional labels."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Cohort must contain a nonempty list of slide records.")
    result, slides, paths = [], set(), set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Every cohort entry must be an object.")
        row = dict(raw)
        for key in ("patient_id", "slide_id", "slide_path"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Cohort entries require {key}.")
            row[key] = row[key].strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["slide_id"]):
            raise ValueError("slide_id must use ASCII letters, numbers, underscore, dot or hyphen.")
        if row["slide_id"] in slides:
            raise ValueError("Duplicate slide_id in cohort.")
        slides.add(row["slide_id"])
        for key in ("slide_path", "annotation_path", "tissue_mask_path"):
            if row.get(key):
                row[key] = str(_resolve(row[key], path.parent))
        if row["slide_path"] in paths:
            raise ValueError("The same slide file cannot occur twice in the cohort.")
        paths.add(row["slide_path"])
        result.append(row)
    return result


def _assign_splits(rows, mode, seed):
    if mode == "predict":
        if any(r.get("split") not in (None, "", "predict") for r in rows):
            raise ValueError("Prediction preparation accepts no split or split=predict.")
        return {r["patient_id"]: "predict" for r in rows}
    supplied = [bool(r.get("split")) for r in rows]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide a split for every slide or omit it for every slide.")
    if not any(supplied):
        return split_patients([r["patient_id"] for r in rows], seed=seed)
    mapping = {}
    for row in rows:
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Labeled slide splits must be train, val or test.")
        patient = row["patient_id"]
        if patient in mapping and mapping[patient] != row["split"]:
            raise ValueError("A patient cannot occur in two cohort splits.")
        mapping[patient] = row["split"]
    return mapping


def _tissue_mask(path, shape):
    if str(path).lower().endswith(".npy"):
        value = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            value = np.asarray(image)
    if value.ndim != 2 or value.shape != shape:
        raise ValueError("Tissue mask must be a scalar image/NPY aligned to the declared output grid.")
    if not np.isin(value, (0, 1, 255)).all():
        raise ValueError("Tissue mask must use 0/1 or 0/255 values.")
    return value > 0


def _candidate_starts(grid, regions, downsample, stride, labeled):
    # Retain the global stride lattice even when labels occupy only a small ROI.
    if not labeled:
        return range(0, grid.width, stride), range(0, grid.height, stride)
    if not regions:
        raise ValueError("The annotation file has no polygons.")
    bounds = [region.geometry.bounds for region in regions]
    left = min(b[0] for b in bounds) / downsample
    top = min(b[1] for b in bounds) / downsample
    right = max(b[2] for b in bounds) / downsample
    bottom = max(b[3] for b in bounds) / downsample
    x0 = max(0, math.floor((left - 224) / stride) * stride)
    y0 = max(0, math.floor((top - 224) / stride) * stride)
    return range(x0, min(grid.width, math.ceil(right)), stride), range(y0, min(grid.height, math.ceil(bottom)), stride)


def prepare(args):
    rows = read_cohort(args.cohort)
    mapping = _assign_splits(rows, args.mode, args.seed)
    class_map = None
    if args.mode == "labeled":
        if not args.class_map:
            raise ValueError("Labeled preparation requires --class-map.")
        class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
        if not isinstance(class_map, dict):
            raise ValueError("Class map must be a JSON object mapping annotation names to label IDs.")
        if any(not r.get("annotation_path") for r in rows):
            raise ValueError("Every labeled slide needs annotation_path.")
    if args.stride > 224:
        raise ValueError("Stride must not exceed the 224-pixel tile size.")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output must be a new or empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("images", "masks", "valid", "metadata"):
        (output / directory).mkdir()
    records, sources = [], []
    for row in rows:
        slide_id = row["slide_id"]
        regions = load_annotations(row["annotation_path"], class_map) if args.mode == "labeled" else []
        with open_slide(row["slide_path"], backend=args.backend) as slide:
            if args.level >= len(slide.level_downsamples):
                raise ValueError("Requested reading level is unavailable in the slide.")
            level_downsample = slide.level_downsamples[args.level]
            downsample = level_downsample * args.read_size / 224.0
            if not math.isclose(downsample * args.stride, round(downsample * args.stride), abs_tol=1e-6, rel_tol=0):
                raise ValueError("The requested scale/stride does not align to integer basic-level tile origins. Use level 0 or a compatible read size.")
            width_base, height_base = slide.dimensions
            grid = OutputGrid.from_base_dimensions(width_base, height_base, downsample, downsample)
            if row.get("tissue_mask_path"):
                tissue = _tissue_mask(row["tissue_mask_path"], grid.shape)
                tissue_relative = f"metadata/{slide_id}_tissue.png"
                Image.fromarray(tissue.astype(np.uint8) * 255).save(output / tissue_relative)
            elif args.allow_full_grid:
                tissue, tissue_relative = None, ""
            else:
                raise ValueError("Supply tissue_mask_path for every slide. --allow-full-grid explicitly declares the entire grid eligible and is suitable only for a known tissue-only ROI or software demonstration.")
            def supplied(name):
                return name in row and row[name] is not None and str(row[name]).strip() != ""
            mpp_x = _positive(row["mpp_x"], "mpp_x") if supplied("mpp_x") else slide.mpp_x
            mpp_y = _positive(row["mpp_y"], "mpp_y") if supplied("mpp_y") else slide.mpp_y
            xs, ys = _candidate_starts(grid, regions, downsample, args.stride,
                                       args.mode == "labeled" and args.unannotated_label == 255)
            before = len(records)
            for gy in ys:
                for gx in xs:
                    valid = np.zeros((224, 224), dtype=bool)
                    h, w = min(224, grid.height - gy), min(224, grid.width - gx)
                    x_centers = (gx + np.arange(w) + 0.5) * downsample
                    y_centers = (gy + np.arange(h) + 0.5) * downsample
                    valid[:h, :w] = (y_centers[:, None] < height_base) & (x_centers[None] < width_base)
                    if tissue is not None:
                        valid[:h, :w] &= tissue[gy:gy+h, gx:gx+w]
                    if not valid.any():
                        continue
                    base_x, base_y = grid.grid_to_base(gx, gy)
                    if not math.isclose(base_x, round(base_x), abs_tol=1e-6, rel_tol=0) or not math.isclose(base_y, round(base_y), abs_tol=1e-6, rel_tol=0):
                        raise ValueError("Tile origins must align to integer basic-level pixels.")
                    base_x, base_y = round(base_x), round(base_y)
                    grid.base_to_grid(base_x, base_y)
                    mask = None
                    if args.mode == "labeled":
                        mask = rasterize_tile(regions, (base_x, base_y), (downsample, downsample), unannotated_label=args.unannotated_label)
                        mask[~valid] = 255
                        if not (mask != 255).any():
                            continue
                    stem = f"{slide_id}_x{gx}_y{gy}"
                    image = slide.read_region((base_x, base_y), args.level, (args.read_size, args.read_size))
                    if image.size != (224, 224):
                        image = image.resize((224, 224), Image.Resampling.BILINEAR)
                    image.save(output / f"images/{stem}.png")
                    Image.fromarray(valid.astype(np.uint8) * 255).save(output / f"valid/{stem}.png")
                    if mask is not None:
                        Image.fromarray(mask).save(output / f"masks/{stem}.png")
                    records.append({
                        "patient_id": row["patient_id"], "split": mapping[row["patient_id"]],
                        "tile_id": stem, "slide_id": slide_id,
                        "image_path": f"images/{stem}.png", "mask_path": f"masks/{stem}.png" if mask is not None else "",
                        "valid_mask_path": f"valid/{stem}.png", "tissue_mask_path": tissue_relative,
                        "base_x": base_x, "base_y": base_y, "grid_x": gx, "grid_y": gy,
                        "level": args.level, "level_downsample": level_downsample, "source_read_size": args.read_size,
                        "downsample_x": downsample, "downsample_y": downsample,
                        "grid_height": grid.height, "grid_width": grid.width,
                        "mpp_x": mpp_x if mpp_x is not None else "", "mpp_y": mpp_y if mpp_y is not None else "",
                    })
            if len(records) == before:
                raise ValueError(f"Slide {slide_id} generated no eligible tiles.")
            sources.append({"slide_id": slide_id, "slide_path": row["slide_path"],
                            "annotation_sha256": sha256(row["annotation_path"]) if args.mode == "labeled" else None,
                            "tissue_mask_sha256": sha256(row["tissue_mask_path"]) if row.get("tissue_mask_path") else None,
                            "dimensions_base": list(slide.dimensions), "output_downsample": downsample,
                            "reading_level": args.level, "reading_level_downsample": level_downsample,
                            "source_read_size": args.read_size, "tiles": len(records)-before,
                            "tissue_mask_provided": tissue is not None})
    if not records:
        raise ValueError("Preparation generated no tiles.")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    # Validate the exported manifest using its actual path resolution rules.
    from .data import load_manifest
    summary = validate_patient_splits(load_manifest(output / "manifest.csv"))
    atomic_json(output / "patient_splits.json", mapping)
    atomic_json(output / "preparation.json", {
        "cohort_sha256": sha256(args.cohort), "class_map_sha256": sha256(args.class_map) if args.class_map else None,
        "mode": args.mode, "seed": args.seed, "tile_size": 224, "stride_output_pixels": args.stride,
        "unannotated_label": args.unannotated_label, "dataset_summary": summary, "sources": sources,
        "classification_scope": "pixel", "class_ids": {"background": 0, "lymphocyte_aggregate": 1,
            "primary_follicle_like": 2, "secondary_follicle_like": 3, "ignore": 255},
        "split_source": ("prediction_only" if args.mode == "predict" else
                         "cohort" if any(r.get("split") for r in rows) else "seeded_patient_level_allocation"),
    })
    print(json.dumps({"tiles": len(records), "patients": len(mapping), "manifest": str(output / "manifest.csv")}), flush=True)
