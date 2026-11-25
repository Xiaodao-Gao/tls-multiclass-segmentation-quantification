# Multi-scale fusion and single-step feature refinement for TLS segmentation

[Repository](https://github.com/Xiaodao-Gao/tls-single-step-segmentation) · [中文说明](README.zh-CN.md) · [Method](docs/METHOD.md) · [Data contract](docs/DATA_FORMAT.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Provenance](docs/PROVENANCE.md)

This repository contains a **patent-aligned research implementation** of four-class tertiary lymphoid structure (TLS) segmentation. It selects four **encoder** feature maps from EfficientNet-B3, fuses them, refines the fused feature with one noise-prediction reconstruction, and injects the result into the final native U-Net decoder feature through a learned channel gate. The original U-Net output head produces all segmentation logits.

This is a new integration of the supplied components. Historical results from decoder-feature fusion or encoder-skip injection do not establish its performance. This release provides software tests and a synthetic end-to-end example; it contains no patient dataset, pretrained checkpoint or claimed paper benchmark. The project is prepared for an author-controlled private submission repository. Publication metadata and licensing must reflect the eventual manuscript and verified component rights.

## Install

Use Python 3.10–3.12 (tested locally with Python 3.12 on macOS ARM64/CPU):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
tlsseg --help
```

For exact versions of the complete locally tested environment, install `requirements-tested.txt` before the editable package. Its package versions describe this release's environment, not the historical training environment. For a CUDA machine, install the matching PyTorch 2.5.1 / torchvision 0.20.1 build for the machine's driver first. No command downloads model weights implicitly. The SMP interface is pinned to 0.5.0; see [official model creation documentation](https://smp.readthedocs.io/en/v0.5.0/quickstart.html).

## Quick software check

Run from the repository root:

```bash
python -m pytest -q
python scripts/make_synthetic_data.py --output outputs/synthetic
tlsseg train --config configs/patent.yaml --manifest outputs/synthetic/manifest.csv --output outputs/smoke --epochs 1 --device cpu --max-train-batches 2 --max-val-batches 1
tlsseg evaluate --checkpoint outputs/smoke/best.pt --manifest outputs/synthetic/manifest.csv --split test --output outputs/smoke/test.json --device cpu
tlsseg predict --checkpoint outputs/smoke/best.pt --manifest outputs/synthetic/manifest.csv --split test --output outputs/predictions --device cpu
tlsseg stitch --layout examples/stitch_layout.json --output outputs/stitched
```

These artificial images check the software path. Their Dice scores do not measure biological segmentation accuracy. The example stitch layout places two synthetic tiles on a declared grid purely to demonstrate the interface; real slide coordinates and scale must come from actual tile extraction metadata. Partial training is explicitly marked in checkpoints and reports.

## Train on an annotated dataset

Prepare 224×224 RGB tiles and scalar label masks, and create a patient-disjoint CSV or JSON manifest. Classes are `0=background`, `1=lymphocyte aggregate`, `2=primary follicle-like`, `3=secondary follicle-like`, with `255=ignore`. The manifest paths are relative to the manifest file or absolute local paths. See [the exact input contract](docs/DATA_FORMAT.md).

```bash
tlsseg train --config configs/patent.yaml --manifest /path/to/manifest.csv --output runs/patent_seed42 --device cuda
tlsseg evaluate --checkpoint runs/patent_seed42/best.pt --manifest /path/to/manifest.csv --split test --output runs/patent_seed42/final_test.json --device cuda
tlsseg predict --checkpoint runs/patent_seed42/best.pt --manifest /path/to/manifest.csv --split test --output outputs/test_predictions --device cuda
```

Training reads the train and validation tiles only and selects `best.pt` by validation **global foreground macro Dice**. Evaluation of the held-out test split is an explicit, separate command; do not repeatedly select settings using its results. The loss is cross-entropy + foreground soft-Dice loss + 0.01×noise MSE. Default optimization uses AdamW, native U-Net LR 1e-5, new branch LR 1e-4, weight decay 1e-4, batch size 4, at most 50 epochs and early-stopping patience 15. These are starting settings inherited from the supplied training recipe, not validated hyperparameters for this new architecture.

Optional `--native-checkpoint /path/to/native.pt` strictly loads a local **plain SMP U-Net** `state_dict` into the native network before training. Random initialization is the default. Full checkpoints from the historical alternative architectures cannot be substituted. To resume, retain the output directory and run with `--resume runs/patent_seed42/last.pt`; only `epochs` may change in the configuration.

## Repository layout

```text
src/tlsseg/models/   Encoder-four-scale model, fusion components, feature denoiser
src/tlsseg/          Data, objectives, metrics, training, prediction, stitching
configs/            Patent architecture and explicit fusion-only ablation
examples/           Manifest and coordinate layout schemas
scripts/            Synthetic data generator and submission archive builder
tests/              Architecture, gradients, data/metrics, area and CLI checks
docs/               Method, data, reproducibility, provenance and verification
.github/workflows/  CPU software-test workflow
```

Probability fusion accepts already predicted tiles and explicit grid coordinates. It does not implement a WSI reader, annotation rasterizer or scanner calibration. The stitched summary counts each covered tissue pixel once. Physical area is reported only with complete x/y scale metadata; connected components are not interpreted as independent biological structures. Large slides require an external chunked/out-of-core backend; the provided NumPy stitcher holds the declared grid in memory.

Please consult [verification and limitations](docs/VERIFICATION.md) before using this code as evidence for a manuscript. The package's [license notice](LICENSE) records the current rights status. There is no paper DOI or claimed pretrained-model release in this repository.
