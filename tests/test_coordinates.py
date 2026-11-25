"""Known-answer basic-slide geometry and coordinate-preserving prediction checks."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tlsseg.coordinates import OutputGrid, integer_number
from tlsseg.predict import (
    PIXEL_CLASS_NAMES, layout_tile_position, predict, prediction_layouts,
    prepare_spatial_metadata, stitch,
)
from tlsseg.runtime import source_hashes


def test_base_dimensions_alignment_and_anisotropic_scale():
    grid = OutputGrid.from_base_dimensions(1001, 803, 4, 2)
    assert grid.shape == (402, 251)
    assert grid.base_to_grid(448, 224) == (112, 112)
    assert grid.grid_to_base(112, 112) == (448.0, 224.0)
    crop = OutputGrid(224, 336, 4, 2, 100, 200)
    assert crop.base_to_grid(548, 424) == (112, 112)
    assert crop.grid_to_base(-1, -1) == (96, 198)
    with pytest.raises(ValueError, match="not aligned"):
        grid.base_to_grid(449, 224)


@pytest.mark.parametrize("arguments", [
    (0, 224, 4, 4), (224, 224, 0, 4), (224, 224, float("nan"), 4),
    (224.5, 224, 4, 4), (224, True, 4, 4),
])
def test_invalid_grid_rejected(arguments):
    with pytest.raises(ValueError):
        OutputGrid(*arguments)


@pytest.mark.parametrize("value", ["112.01", "NaN", "9007199254740993", True])
def test_csv_grid_integer_is_not_silently_rounded(value):
    with pytest.raises(ValueError):
        integer_number(value, "grid_x")


def _spatial_row(**kwargs):
    return dict(slide_id="synthetic_slide", base_x="448", base_y="0", level="0",
                grid_x="112", grid_y="0", grid_height="224", grid_width="336",
                downsample_x="4", downsample_y="4", mpp_x="0.25", mpp_y="0.25", **kwargs)


def test_metadata_parse_and_layout_preserve_base_grid_and_physical_contract(tmp_path):
    valid = tmp_path / "valid.npy"
    tissue = tmp_path / "tissue.npy"
    np.save(valid, np.ones((224, 224), dtype=bool))
    np.save(tissue, np.ones((224, 336), dtype=bool))
    rows = [_spatial_row(valid_mask_path="valid.npy", tissue_mask_path="tissue.npy")]
    metadata = prepare_spatial_metadata(rows, tmp_path / "manifest.csv")
    spatial = metadata[0]
    assert spatial["grid_x"] == 112 and spatial["base_x"] == 448
    assert spatial["level"] == 0
    assert spatial["valid_mask_path"] == str(valid)
    layout = prediction_layouts(metadata, [{"probability_path": "tile_000000.npz"}])[0]
    assert layout["shape"] == [224, 336]
    assert layout["metadata"] == dict(downsample_x=4.0, downsample_y=4.0, mpp_x=0.25, mpp_y=0.25)
    assert layout["tissue_mask"] == str(tissue)
    assert layout["tiles"][0]["base_x"] == 448
    assert layout_tile_position(layout, layout["tiles"][0]) == (112, 0)


def test_partial_conflicting_and_misaligned_registration_rejected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    assert prepare_spatial_metadata([{"tile_id": "a"}], manifest) == [None]
    with pytest.raises(ValueError, match="cannot mix"):
        prepare_spatial_metadata([_spatial_row(), {}], manifest)
    row = _spatial_row()
    row.pop("grid_width")
    with pytest.raises(ValueError, match="Registered tiles require"):
        prepare_spatial_metadata([row], manifest)
    row = _spatial_row()
    row["base_x"] = 449
    with pytest.raises(ValueError, match="not aligned"):
        prepare_spatial_metadata([row], manifest)
    row = _spatial_row()
    row["grid_x"] = 111
    with pytest.raises(ValueError, match="disagree"):
        prepare_spatial_metadata([row], manifest)
    first, second = _spatial_row(), _spatial_row()
    second["mpp_x"] = "0.5"
    with pytest.raises(ValueError, match="same slide disagree"):
        prepare_spatial_metadata([first, second], manifest)
    first, second = _spatial_row(patient_id="synthetic_a"), _spatial_row(patient_id="synthetic_b")
    with pytest.raises(ValueError, match="patient identity"):
        prepare_spatial_metadata([first, second], manifest)


def test_stitch_layout_maps_basic_coordinates_without_inventing_scale():
    layout = {"shape": [224, 336], "metadata": {"downsample_x": 4, "downsample_y": 2}}
    assert layout_tile_position(layout, {"base_x": 448, "base_y": 224}) == (112, 112)
    assert layout_tile_position({"shape": [224, 336]}, {"x": 112, "y": 0}) == (112, 0)
    with pytest.raises(ValueError, match="downsample"):
        layout_tile_position({"shape": [224, 336]}, {"base_x": 448, "base_y": 224})
    with pytest.raises(ValueError, match="disagree"):
        layout_tile_position(layout, {"base_x": 448, "base_y": 224, "x": 0, "y": 0})
    with pytest.raises(ValueError, match="both"):
        layout_tile_position(layout, {"base_x": 448})


def test_reading_level_and_resize_factor_preserved_and_checked(tmp_path):
    row = _spatial_row(level_downsample="2", source_read_size="448")
    metadata = prepare_spatial_metadata([row], tmp_path / "manifest.csv")
    assert metadata[0]["level_downsample"] == 2
    assert metadata[0]["source_read_size"] == 448
    tile = prediction_layouts(metadata, [{"probability_path": "tile.npz"}])[0]["tiles"][0]
    assert tile["level_downsample"] == 2 and tile["source_read_size"] == 448
    row["source_read_size"] = "224"
    with pytest.raises(ValueError, match="reading level and image resizing"):
        prepare_spatial_metadata([row], tmp_path / "manifest.csv")


def test_unregistered_valid_mask_does_not_invent_registration(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("tlsseg.predict")

    class ConstantLogits(torch.nn.Module):
        def forward(self, images):
            return {"logits": images.new_zeros((len(images), 4, 224, 224))}

        def architecture_config(self):
            return {"architecture_id": "synthetic_roi_stub"}

    checkpoint = tmp_path / "synthetic_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic ROI fixture")
    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (
        ConstantLogits(), {"epoch": 0, "environment": {"source_sha256": source_hashes()}}))
    image = tmp_path / "image.png"
    valid = tmp_path / "valid.png"
    Image.fromarray(np.full((224, 224, 3), 180, dtype=np.uint8)).save(image)
    Image.fromarray(np.full((224, 224), 255, dtype=np.uint8)).save(valid)
    manifest = tmp_path / "manifest.csv"
    row = dict(patient_id="synthetic_patient", split="predict", tile_id="synthetic_roi",
               image_path=image.name, valid_mask_path=valid.name)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    metadata = prepare_spatial_metadata([row], manifest)
    assert metadata == [{"valid_mask_path": str(valid)}]
    assert prediction_layouts(metadata, [{"probability_path": "tile.npz"}]) == []
    output = tmp_path / "predictions"
    predict(SimpleNamespace(device="cpu", manifest=str(manifest), split="predict", output=str(output),
                            checkpoint=str(checkpoint), batch_size=1, num_workers=0))
    index = json.loads((output / "index.json").read_text())
    assert index["tiles"][0]["valid_mask_path"] == str(valid)
    assert "slide_id" not in index["tiles"][0] and "grid_x" not in index["tiles"][0]
    assert index["stitch_layouts"] == []
    assert not list(output.glob("layout_*.json"))


@pytest.mark.parametrize("mask", [
    np.ones((223, 224), dtype=np.uint8),
    np.full((224, 224), 2, dtype=np.uint8),
    np.ones((224, 224, 3), dtype=np.uint8),
])
def test_unregistered_valid_mask_geometry_and_binary_values_checked(tmp_path, mask):
    path = tmp_path / "valid.npy"
    np.save(path, mask)
    with pytest.raises(ValueError, match="224x224 scalar binary"):
        prepare_spatial_metadata([{"valid_mask_path": path.name}], tmp_path / "manifest.csv")


def test_registered_predict_exports_pixel_classes_and_automatic_layout(tmp_path, monkeypatch):
    """Exercise metadata I/O with a synthetic stub; this does not test accuracy."""
    import importlib
    module = importlib.import_module("tlsseg.predict")

    class ConstantLogits(torch.nn.Module):
        def forward(self, images):
            return {"logits": images.new_zeros((len(images), 4, 224, 224))}

        def architecture_config(self):
            return {"architecture_id": "synthetic_metadata_stub"}

    checkpoint = tmp_path / "synthetic_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic metadata test fixture")
    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (
        ConstantLogits(), {"epoch": 0, "environment": {"source_sha256": source_hashes()}}))
    tissue = tmp_path / "tissue.npy"
    valid = tmp_path / "valid.npy"
    np.save(tissue, np.ones((224, 336), dtype=bool))
    np.save(valid, np.ones((224, 224), dtype=bool))
    records = []
    for i in range(2):
        image = tmp_path / f"image_{i}.png"
        Image.fromarray(np.full((224, 224, 3), 180, dtype=np.uint8)).save(image)
        row = _spatial_row(valid_mask_path="valid.npy", tissue_mask_path="tissue.npy")
        row.update(patient_id="synthetic_patient", split="predict", tile_id=f"synthetic_{i}",
                   image_path=image.name, grid_x=str(i * 112), base_x=str(i * 448))
        records.append(row)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    output = tmp_path / "predictions"
    predict(SimpleNamespace(device="cpu", manifest=str(manifest), split="predict", output=str(output),
                            checkpoint=str(checkpoint), batch_size=2, num_workers=0))
    index = json.loads((output / "index.json").read_text())
    assert index["classification_scope"] == "pixel"
    assert index["class_names"] == PIXEL_CLASS_NAMES
    assert index["tiles"][1]["base_x"] == 448
    assert index["tiles"][1]["grid_x"] == 112
    assert index["stitch_layouts"] == [{"slide_id": "synthetic_slide", "layout_path": "layout_0000.json"}]
    fused_output = tmp_path / "stitched"
    stitch(SimpleNamespace(layout=str(output / "layout_0000.json"), output=str(fused_output)))
    summary = json.loads((fused_output / "quantification.json").read_text())
    assert summary["covered_tissue_pixels"] == 224 * 336
    assert summary["physical_area_available"] is True
    assert summary["classification_scope"] == "pixel"
    assert summary["class_names"] == PIXEL_CLASS_NAMES
