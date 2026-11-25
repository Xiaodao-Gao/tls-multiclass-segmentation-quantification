import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tlsseg.cli import main
from tlsseg.engine import read_config


ROOT = Path(__file__).resolve().parents[1]


def test_config_rejects_unknown_and_invalid(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("native_lrr: 1\n")
    with pytest.raises(ValueError, match="Unknown"):
        read_config(config)
    config.write_text("batch_size: 0\n")
    with pytest.raises(ValueError, match="positive"):
        read_config(config)


def test_synthetic_train_reload_evaluate_predict(tmp_path):
    spec = importlib.util.spec_from_file_location("synthetic", ROOT / "scripts/make_synthetic_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.make_dataset(tmp_path / "data")
    run = tmp_path / "run"
    main(["train", "--config", str(ROOT / "configs/patent.yaml"), "--manifest", str(manifest),
          "--output", str(run), "--epochs", "1", "--device", "cpu",
          "--max-train-batches", "2", "--max-val-batches", "1"])
    status = json.loads((run / "status.json").read_text())
    assert status["completed"] and status["partial_smoke_run"]
    assert status["test_evaluated"] is False
    checkpoint = run / "best.pt"
    assert checkpoint.is_file()
    report = tmp_path / "evaluation.json"
    main(["evaluate", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
          "--split", "test", "--device", "cpu", "--output", str(report)])
    evaluation = json.loads(report.read_text())
    assert evaluation["objective"]["noise_mse"] is None
    assert evaluation["checkpoint_partial_smoke_run"] is True
    prediction = tmp_path / "prediction"
    main(["predict", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
          "--split", "test", "--device", "cpu", "--output", str(prediction)])
    with np.load(prediction / "tile_000000.npz") as archive:
        probability = archive["probabilities"]
    assert probability.shape == (4, 224, 224)
    np.testing.assert_allclose(probability.sum(0), 1, atol=1e-6)
    index = json.loads((prediction / "index.json").read_text())
    assert len(index["tiles"]) == 2
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"shape": [224, 336], "tiles": [
        {"probability_path": str(prediction / "tile_000000.npz"), "x": 0, "y": 0},
        {"probability_path": str(prediction / "tile_000001.npz"), "x": 112, "y": 0},
    ]}))
    stitched = tmp_path / "stitched"
    main(["stitch", "--layout", str(layout), "--output", str(stitched)])
    quantification = json.loads((stitched / "quantification.json").read_text())
    assert quantification["covered_tissue_pixels"] == 224 * 336
    assert quantification["physical_area_available"] is False
    # A new CSV cannot relabel the known training patient as a held-out patient.
    relabeled = tmp_path / "relabeled.json"
    relabeled.write_text(json.dumps([{
        "patient_id": "synthetic_train_patient_0", "split": "test",
        "image_path": str(manifest.parent / "synthetic_train_0.png"),
        "mask_path": str(manifest.parent / "synthetic_train_0_mask.png"),
    }]))
    evaluate_args = ["evaluate", "--checkpoint", str(checkpoint), "--manifest", str(relabeled),
                     "--split", "test", "--device", "cpu", "--output", str(report)]
    with pytest.raises(ValueError, match="manifest differs"):
        main(evaluate_args)
    with pytest.raises(ValueError, match="patient IDs"):
        main(evaluate_args + ["--external-cohort"])
    # Partial history remains labeled after an unrestricted resume.
    main(["train", "--config", str(ROOT / "configs/patent.yaml"), "--manifest", str(manifest),
          "--output", str(run), "--resume", str(run / "last.pt"), "--epochs", "2", "--device", "cpu"])
    resumed = torch.load(run / "last.pt", weights_only=True, map_location="cpu")
    assert resumed["epoch"] == 2 and resumed["partial_smoke_run"] is True
    assert resumed["history"][0]["batch_limits"] == {"train": 2, "val": 1}
    assert resumed["history"][1]["batch_limits"] == {"train": None, "val": None}
