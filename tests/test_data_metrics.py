import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image

from tlsseg.data import TileDataset, load_manifest, validate_patient_splits
from tlsseg.losses import segmentation_loss
from tlsseg.metrics import ConfusionMatrix


def write_tile(tmp_path, *, mask=None, size=(224, 224)):
    image = np.zeros((*size, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "image.png")
    if mask is not None:
        Image.fromarray(mask).save(tmp_path / "mask.png")
    row = {"patient_id": "synthetic-01", "split": "train", "tile_id": "tile-01", "image_path": "image.png"}
    if mask is not None:
        row["mask_path"] = "mask.png"
    return row


@pytest.mark.parametrize("form", ["list", "records", "splits", "csv"])
def test_manifest_relative_paths_and_patient_summary(tmp_path, form):
    row = write_tile(tmp_path, mask=np.zeros((224, 224), np.uint8))
    path = tmp_path / ("manifest.csv" if form == "csv" else "manifest.json")
    if form == "csv":
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    else:
        payload = [row] if form == "list" else {"records": [row]} if form == "records" else {"splits": {"train": [row]}}
        path.write_text(json.dumps(payload))
    rows = load_manifest(path)
    assert rows[0]["image_path"] == str(tmp_path / "image.png")
    assert rows[0]["mask_path"] == str(tmp_path / "mask.png")
    summary = validate_patient_splits(rows)
    assert summary["patient_counts"]["train"] == 1
    assert summary["labeled_tile_counts"]["train"] == 1
    assert "patients" not in summary
    json.dumps(summary, allow_nan=False)


def test_patient_leakage_and_duplicate_images_are_rejected():
    first = {"patient_id": "synthetic-01", "split": "train", "image_path": "a.png", "tile_id": "a"}
    second = {"patient_id": "synthetic-01", "split": "val", "image_path": "b.png", "tile_id": "b"}
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_patient_splits([first, second])
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_patient_splits([first, dict(second, patient_id=" synthetic-01 ")])
    with pytest.raises(ValueError, match="Duplicate tile image"):
        validate_patient_splits([first, dict(first, tile_id="different")])


def test_unlabelled_prediction_and_imagenet_normalization(tmp_path):
    row = write_tile(tmp_path)
    row["split"] = "predict"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([row]))
    item = TileDataset(load_manifest(path))[0]
    assert set(item) == {"image", "tile_id", "path"}
    assert item["image"].shape == (3, 224, 224)
    assert item["image"].dtype == torch.float32
    expected = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(item["image"][:, 0, 0], expected)


@pytest.mark.parametrize("bad_mask", [np.zeros((224, 224, 3), np.uint8), np.full((224, 224), 4, np.uint8), np.zeros((223, 224), np.uint8)])
def test_invalid_masks_are_rejected(tmp_path, bad_mask):
    row = write_tile(tmp_path, mask=bad_mask)
    row.update(image_path=str(tmp_path / "image.png"), mask_path=str(tmp_path / "mask.png"))
    with pytest.raises(ValueError):
        TileDataset([row])[0]


def test_wrong_image_size_is_not_resized(tmp_path):
    row = write_tile(tmp_path, size=(223, 224))
    row["image_path"] = str(tmp_path / "image.png")
    with pytest.raises(ValueError, match="resizing is not implicit"):
        TileDataset([row])[0]


def test_geometry_augmentation_keeps_image_and_mask_aligned(tmp_path, monkeypatch):
    image = np.zeros((224, 224, 3), np.uint8)
    mask = np.zeros((224, 224), np.uint8)
    image[10:30, 40:70] = 255
    mask[10:30, 40:70] = 2
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    monkeypatch.setattr("tlsseg.data.random.randrange", lambda _: 1)
    monkeypatch.setattr("tlsseg.data.random.random", lambda: 1.0)
    monkeypatch.setattr("tlsseg.data.random.uniform", lambda *_: 1.0)
    item = TileDataset([{"image_path": str(tmp_path / "image.png"), "mask_path": str(tmp_path / "mask.png"), "tile_id": "synthetic"}], augment=True)[0]
    assert torch.equal(item["image"][0] > 0, item["mask"] == 2)
    assert item["mask"].dtype == torch.long


def test_loss_ignores_masked_logits_and_backpropagates():
    torch.manual_seed(7)
    logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    mask = torch.randint(0, 4, (1, 8, 8))
    mask[:, :2, :2] = 255
    changed = logits.detach().clone()
    changed[:, :, :2, :2] += torch.randn_like(changed[:, :, :2, :2]) * 100
    total, parts = segmentation_loss(logits, mask, torch.tensor(0.7), 0.01)
    other, _ = segmentation_loss(changed, mask, torch.tensor(0.7), 0.01)
    assert torch.allclose(total, other, atol=1e-6)
    assert torch.allclose(total, parts["ce"] + parts["dice"] + 0.01 * parts["noise_mse"])
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, :, :2, :2]) == 0


def test_all_ignore_loss_is_differentiable_zero():
    logits = torch.randn(1, 4, 2, 2, requires_grad=True)
    total, _ = segmentation_loss(logits, torch.full((1, 2, 2), 255), torch.tensor(1.0))
    assert total.item() == 0
    total.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_global_confusion_metrics_and_per_metric_absence_policy():
    target = torch.tensor([[0, 1, 1], [2, 255, 0]])
    prediction = torch.tensor([[0, 1, 0], [3, 2, 0]])
    cm = ConfusionMatrix()
    cm.update(prediction, target)
    result = cm.compute()
    assert result["confusion_matrix"] == [[2, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    assert result["valid_pixels"] == 5
    assert result["macro_fg"]["dice"] == pytest.approx(2 / 9)
    assert result["macro_fg"]["precision"] == pytest.approx(0.5)
    assert result["macro_fg"]["recall"] == pytest.approx(0.25)
    assert result["per_class"]["FL1"]["precision"] is None
    assert result["per_class"]["FL2"]["recall"] is None
    logits = torch.nn.functional.one_hot(prediction, 4).permute(2, 0, 1).float()[None]
    other = ConfusionMatrix()
    other.update(logits, target[None])
    assert other.compute() == result
    json.dumps(result, allow_nan=False)


def test_absent_class_is_null_and_empty_epoch_is_error():
    cm = ConfusionMatrix()
    cm.update(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long))
    result = cm.compute()
    assert result["per_class"]["Agg"]["dice"] is None
    assert result["macro_fg"]["dice"] is None
    cm.reset()
    cm.update(torch.zeros(2, 2, dtype=torch.long), torch.full((2, 2), 255))
    with pytest.raises(ValueError, match="No valid pixels"):
        cm.compute()
