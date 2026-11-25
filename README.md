# TLS Multi-class Segmentation and Quantification

[Repository](https://github.com/Xiaodao-Gao/tls-multiclass-segmentation-quantification) · [Method](docs/METHOD.md) · [Slide preparation](docs/PREPARATION.md) · [Data contract](docs/DATA_FORMAT.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Provenance](docs/PROVENANCE.md)

This repository implements the final patent's **four-class pixel classification, segmentation and area quantification** for tertiary lymphoid structures (TLS). Every pixel is assigned to background, lymphocyte aggregate, primary follicle-like or secondary follicle-like tissue by one native four-class output head. The model selects four **encoder** feature maps from EfficientNet-B3, fuses them, refines the fused feature with one noise-prediction reconstruction, and injects the result into the final native U-Net decoder feature through a learned channel gate. Pixel classification and segmentation use this same output path.

This is a new integration of the supplied components. Historical results from decoder-feature fusion or encoder-skip injection do not establish its performance. This release provides software tests and a synthetic end-to-end example; it contains no patient dataset, pretrained checkpoint or claimed paper benchmark. This public repository provides source and reproducibility materials for research submission. Publication metadata and licensing must reflect the eventual manuscript and verified component rights.

## Install

Use Python 3.10–3.12 (tested locally with Python 3.12 on macOS ARM64/CPU):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
tlsseg --help
```

For GeoJSON preparation install `python -m pip install -e '.[dev,preprocessing]'`. For region-based WSI reading with OpenSlide install `python -m pip install -e '.[dev,wsi]'`. Ordinary PNG/JPEG examples can use the guarded Pillow backend. The slide reader records basic-level coordinates, reading level and actual output downsample, and never infers scanner MPP from image size or print DPI.

For exact versions of the complete locally tested environment, install `requirements-tested.txt` before the editable package. Its package versions describe this release's environment, not the historical training environment. For a CUDA machine, install the matching PyTorch 2.5.1 / torchvision 0.20.1 build for the machine's driver first. No command downloads model weights implicitly. The SMP interface is pinned to 0.5.0; see [official model creation documentation](https://smp.readthedocs.io/en/v0.5.0/quickstart.html).

## Quick software check

Run from the repository root:

```bash
python -m pytest -q
python scripts/make_synthetic_data.py --output outputs/synthetic
tlsseg train --config configs/synthetic_smoke.yaml --manifest outputs/synthetic/manifest.csv --output outputs/smoke --epochs 1 --device cpu --max-train-batches 2 --max-val-batches 1
tlsseg evaluate --checkpoint outputs/smoke/best.pt --manifest outputs/synthetic/manifest.csv --split test --output outputs/smoke/test.json --device cpu
tlsseg predict --checkpoint outputs/smoke/best.pt --manifest outputs/synthetic/manifest.csv --split test --output outputs/predictions --device cpu
tlsseg stitch --layout examples/stitch_layout.json --output outputs/stitched
```

These artificial images check the software path. `synthetic_smoke.yaml` explicitly selects random initialization and is separate from the patent training recipe. Its Dice scores do not measure biological segmentation accuracy. The example stitch layout places two synthetic tiles on a declared grid purely to demonstrate the interface; real slide coordinates and scale must come from actual tile extraction metadata. Partial training is explicitly marked in checkpoints and reports.

## Train on an annotated dataset

Prepare slides and annotations with the following command, or supply an existing compatible tile manifest:

```bash
tlsseg prepare --cohort /path/to/cohort.csv --class-map examples/class_map.json --output data/prepared --level 0 --read-size 224 --stride 112
```

The cohort contains patient/slide identifiers, slide paths, reviewed GeoJSON annotations and binary tissue masks aligned to the chosen output grid. The command rasterizes polygon exteriors and holes, marks unknown/conflicting regions as ignore=255, creates patient-disjoint 70:20:10 subsets with seed 42 when original assignments are absent, and writes RGB tiles, labels, valid masks and coordinate records. Use the study's original split assignments to reproduce that specific cohort. See [slide preparation](docs/PREPARATION.md) for the input schema and coordinate/scale contract.

Prepare 224×224 RGB tiles and scalar label masks, and create a patient-disjoint CSV or JSON manifest. Classes are `0=background`, `1=lymphocyte aggregate`, `2=primary follicle-like`, `3=secondary follicle-like`, with `255=ignore`. The manifest paths are relative to the manifest file or absolute local paths. See [the exact input contract](docs/DATA_FORMAT.md).

```bash
tlsseg train --config configs/patent.yaml --manifest /path/to/manifest.csv --native-checkpoint /path/to/native_epoch45.pt --output runs/patent_seed42 --device cuda
tlsseg evaluate --checkpoint runs/patent_seed42/best.pt --manifest /path/to/manifest.csv --split test --output runs/patent_seed42/final_test.json --device cuda
tlsseg predict --checkpoint runs/patent_seed42/best.pt --manifest /path/to/manifest.csv --split test --output outputs/test_predictions --device cuda
```

When the manifest contains prepared slide geometry, prediction preserves it and writes one `layout_0000.json` (and subsequent numbered files) per slide. Fuse a slide using `tlsseg stitch --layout outputs/test_predictions/layout_0000.json --output outputs/fused_slide_0000`. Basic-level tile positions are checked against the declared grid before fusion. Overlap probabilities are fused before class selection, and each eligible grid pixel contributes once to the area summary.

Training reads the train and validation tiles only and selects `best.pt` by validation **global foreground macro Dice**. Evaluation of the held-out test split is an explicit, separate command; do not repeatedly select settings using its results. The loss is cross-entropy + foreground soft-Dice loss + 0.01×noise MSE. `patent.yaml` uses AdamW, native U-Net LR 5e-6, new branch LR 3e-4, weight decay 1e-4, batch size 4, at most 50 epochs and early-stopping patience 15. Training, evaluation and prediction use FP32 with autocast/AMP and both matmul/cuDNN TF32 disabled; the precision settings are recorded in run metadata.

`patent.yaml` requires `initialization: native_checkpoint` and an explicit `--native-checkpoint` pointing to a local **plain SMP U-Net** `state_dict`. The patent's experimental setup uses a native U-Net checkpoint selected at epoch 45; that actual checkpoint is not included here. The trainer strictly checks tensor compatibility and records the file SHA256, but does not certify its training epoch, cohort or provenance. The data owner must supply and verify the intended source checkpoint. Full checkpoints from the historical alternative architectures cannot be substituted. To resume, retain the output directory and use `--resume runs/patent_seed42/last.pt` without supplying the native checkpoint again; only `epochs` may change in the configuration.

The 50-index noise schedule and DiT patch, embedding and head settings are defaults retained from the supplied source components. The patent does not specify a numeric total diffusion-step count, so 50 is not presented as a patent-mandated constant. No real-cohort retraining has been completed for this integration, and historical experimental scores are not reproduced by the synthetic check.

## Repository layout

```text
src/tlsseg/models/   Encoder-four-scale model, fusion components, feature denoiser
src/tlsseg/          Slide/annotation preparation, patient splits, data, training, prediction and quantification
configs/            Patent recipe, random-initialized software smoke and fusion-only ablation
examples/           Manifest and coordinate layout schemas
scripts/            Synthetic data generator and submission archive builder
tests/              Architecture, gradients, data/metrics, area and CLI checks
docs/               Method, data, reproducibility, provenance and verification
.github/workflows/  CPU software-test workflow
```

The implementation covers region-based slide reading, reviewed polygon rasterization, patient-level split generation, four-class model training/prediction, coordinate conversion, probability fusion and area quantification. Reviewed pathology annotations, the eligible-tissue mask, source checkpoint and valid scanner calibration remain study inputs. Physical area is reported only with complete x/y scale metadata; connected components are not interpreted as independent biological structures. The grid stitcher and tissue-mask handling are in memory, so choose an output scale or a defined ROI that fits available memory; out-of-core whole-slide aggregation is not included. Vendor-specific WSI and real-cohort accuracy require study-specific validation.

Please consult [verification and limitations](docs/VERIFICATION.md) before using this code as evidence for a manuscript. The package's [license notice](LICENSE) records the current rights status. There is no paper DOI or claimed pretrained-model release in this repository.
