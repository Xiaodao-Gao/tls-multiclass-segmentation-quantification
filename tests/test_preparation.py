import json

import numpy as np
import pytest
from PIL import Image

from tlsseg.cli import main
from tlsseg.data import TileDataset, load_manifest, validate_patient_splits
from tlsseg.predict import prepare_spatial_metadata, prediction_layouts


def polygon(label, coordinates):
    return {"type": "Feature", "properties": {"class_id": label},
            "geometry": {"type": "Polygon", "coordinates": [coordinates]}}


def make_cohort(folder, count=10, tissue=True):
    folder.mkdir()
    rows = []
    annotation = {"type": "FeatureCollection", "features": [
        polygon(0, [[0, 0], [672, 0], [672, 448], [0, 448], [0, 0]]),
        polygon(1, [[40, 40], [180, 40], [180, 180], [40, 180], [40, 40]]),
        polygon(2, [[180, 100], [300, 100], [300, 260], [180, 260], [180, 100]]),
        polygon(3, [[400, 200], [600, 200], [600, 400], [400, 400], [400, 200]]),
    ]}
    for i in range(count):
        stem = f"slide_{i}"
        Image.new("RGB", (672, 448), (220, 180, 210)).save(folder / f"{stem}.png")
        (folder / f"{stem}.geojson").write_text(json.dumps(annotation))
        row = {"patient_id": f"patient_{i}", "slide_id": stem, "slide_path": f"{stem}.png",
               "annotation_path": f"{stem}.geojson", "mpp_x": 0.25, "mpp_y": 0.5}
        if tissue:
            mask = np.ones((224, 336), dtype=np.uint8) * 255
            mask[:5] = 0
            Image.fromarray(mask).save(folder / f"{stem}_tissue.png")
            row["tissue_mask_path"] = f"{stem}_tissue.png"
        rows.append(row)
    cohort = folder / "cohort.json"
    cohort.write_text(json.dumps(rows))
    class_map = folder / "classes.json"
    class_map.write_text("{}")
    return cohort, class_map


def test_prepare_patient_split_rasterization_coordinates_and_layout(tmp_path):
    cohort, class_map = make_cohort(tmp_path / "source")
    output = tmp_path / "prepared"
    main(["prepare", "--cohort", str(cohort), "--class-map", str(class_map),
          "--output", str(output), "--backend", "pillow", "--read-size", "448"])
    rows = load_manifest(output / "manifest.csv")
    summary = validate_patient_splits(rows)
    assert summary["patient_counts"] == {"train": 7, "val": 2, "test": 1, "predict": 0}
    first = rows[0]
    assert float(first["downsample_x"]) == float(first["downsample_y"]) == 2
    assert int(first["base_x"]) == int(first["grid_x"]) * 2
    assert int(first["base_y"]) == int(first["grid_y"]) * 2
    with Image.open(first["mask_path"]) as image:
        mask = np.asarray(image)
    assert mask.shape == (224, 224)
    assert (mask[:5] == 255).all()
    assert 1 in np.unique(mask) and 2 in np.unique(mask)
    item = TileDataset([first])[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["mask"][:5].eq(255).all()
    registered = prepare_spatial_metadata(rows, output / "manifest.csv")
    index = [{"probability_path": f"tile_{i}.npz"} for i in range(len(rows))]
    layouts = prediction_layouts(registered, index)
    assert len(layouts) == 10
    assert layouts[0]["metadata"] == {"downsample_x": 2, "downsample_y": 2, "mpp_x": 0.25, "mpp_y": 0.5}
    # Use known-answer probabilities to check unique tissue area from the
    # automatically generated base-coordinate layout, independent of a model.
    layout = layouts[0]
    for tile in layout["tiles"]:
        probability = np.zeros((4, 224, 224), dtype=np.float32)
        probability[1] = 1
        np.savez_compressed(output / tile["probability_path"], probabilities=probability)
    (output / "layout.json").write_text(json.dumps(layout))
    main(["stitch", "--layout", str(output / "layout.json"), "--output", str(tmp_path / "fused")])
    q = json.loads((tmp_path / "fused/quantification.json").read_text())
    expected = 219 * 336
    assert q["covered_tissue_pixels"] == expected
    assert q["foreground_pixel_count"] == expected
    assert q["foreground_area_mm2"] == pytest.approx(expected * 2 * 2 * 0.25 * 0.5 * 1e-6)
    assert q["classification_scope"] == "pixel"


def test_prepare_requires_tissue_or_explicit_full_grid_declaration(tmp_path):
    cohort, class_map = make_cohort(tmp_path / "source", count=3, tissue=False)
    args = ["prepare", "--cohort", str(cohort), "--class-map", str(class_map),
            "--backend", "pillow", "--read-size", "448"]
    with pytest.raises(ValueError, match="tissue_mask_path"):
        main(args + ["--output", str(tmp_path / "strict")])
    main(args + ["--output", str(tmp_path / "full"), "--allow-full-grid"])
    prep = json.loads((tmp_path / "full/preparation.json").read_text())
    assert all(s["tissue_mask_provided"] is False for s in prep["sources"])


def test_prediction_preparation_needs_no_annotation_or_label(tmp_path):
    cohort, _ = make_cohort(tmp_path / "source", count=1)
    rows = json.loads(cohort.read_text())
    del rows[0]["annotation_path"]
    cohort.write_text(json.dumps(rows))
    output = tmp_path / "prepared"
    main(["prepare", "--cohort", str(cohort), "--mode", "predict", "--output", str(output),
          "--backend", "pillow", "--read-size", "448"])
    result = load_manifest(output / "manifest.csv")
    assert all(row["split"] == "predict" and "mask_path" not in row for row in result)
    assert validate_patient_splits(result)["patient_counts"]["predict"] == 1


def test_valid_mask_excludes_prebuilt_training_pixels(tmp_path):
    Image.new("RGB", (224, 224), "white").save(tmp_path / "image.png")
    Image.fromarray(np.ones((224, 224), dtype=np.uint8)).save(tmp_path / "mask.png")
    valid = np.ones((224, 224), dtype=np.uint8)
    valid[0] = 0
    Image.fromarray(valid).save(tmp_path / "valid.png")
    item = TileDataset([{"image_path": str(tmp_path / "image.png"), "mask_path": str(tmp_path / "mask.png"),
                         "valid_mask_path": str(tmp_path / "valid.png"), "tile_id": "one"}])[0]
    assert item["mask"][0].eq(255).all() and item["mask"][1:].eq(1).all()


@pytest.mark.parametrize("invalid_mpp", [0, False, True])
def test_explicit_invalid_mpp_cannot_fall_back_or_become_scale(tmp_path, invalid_mpp):
    cohort, _ = make_cohort(tmp_path / "source", count=1)
    rows = json.loads(cohort.read_text())
    rows[0]["mpp_x"] = invalid_mpp
    cohort.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="mpp_x"):
        main(["prepare", "--cohort", str(cohort), "--mode", "predict", "--output", str(tmp_path / "prepared"),
              "--backend", "pillow", "--read-size", "448"])
