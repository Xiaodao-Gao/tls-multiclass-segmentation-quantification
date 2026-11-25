# Reproducing a study with this code

## Software state

Record the repository commit and release ZIP checksum. `pyproject.toml` pins the direct runtime dependencies; `requirements-tested.txt` records the complete locally installed environment excluding this editable package. The original historical training logs only establish PyTorch 2.3.1+cu118/CUDA 11.8 and an SMP 0.5 API expectation; the new environment is not presented as that historical environment.

Run `python -m pytest -q` before the study. The CI workflow runs the same software tests on a CPU runner. Local checks are documented in `VERIFICATION.md`; a configured workflow is not a claim that remote GitHub CI has already run.

## Data and training record

Define patient-disjoint training/validation/test sets before training. `prepare` creates a seeded patient-level 70:20:10 allocation if no original assignments are supplied, or preserves the cohort's explicit frozen splits. It also reads slide regions, rasterizes reviewed polygon annotations and records tile geometry, reading scale and valid masks. Reproducing the historical study requires its original patient assignments and source data, not only the same seed or cohort size. Reviewed annotations, tissue masks and valid scanner calibration remain study inputs. Run training with the appropriate config and a manifest belonging to the intended cohort. Every run records the manifest SHA256, split counts, architecture contract, dependency versions, source SHA256 values, configuration, precision settings and device in `run.json`. Patient IDs and input paths belong in the data owner's local manifest; they are not needed in a code release. A manifest checksum records its bytes but does not hash the image/mask files themselves. Archive the data version and preprocessing code separately under the study's data governance.

`configs/patent.yaml` requires `initialization: native_checkpoint`: begin training with `--native-checkpoint /path/to/native_epoch45.pt`. The supplied file must contain the compatible plain SMP U-Net state dictionary intended for the study. The patent's source checkpoint was selected at epoch 45; its actual weights are absent from this public repository. Strict loading verifies tensor compatibility and the recorded SHA256 identifies the file, but neither authenticates its epoch or cohort provenance. The data owner must verify those facts. The recipe uses native-network LR 5e-6, added-branch LR 3e-4 and weight decay 1e-4. For artificial software checks only, `configs/synthetic_smoke.yaml` explicitly selects `initialization: random`.

The configured total noise-step count of 50 and the DiT patch, embedding and head settings come from the supplied source-component defaults. The patent does not state a numeric total diffusion-step count; do not present 50 as a patent requirement. The released integration has not been retrained on a real cohort and does not reproduce historical performance figures.

`best.pt` is selected solely by validation global foreground macro Dice. `last.pt` also records the optimizer, epoch, history and Python/NumPy/PyTorch RNG states. Training does not evaluate test images. The explicit evaluate command performs no calibration or parameter fitting. To claim a held-out result, freeze model/config choices before inspecting the test metrics, and report the number of patients/tiles and foreground class support.

Use multiple preregistered seeds for performance estimates. Re-run the full architecture and any ablation on the same data/splits/initialization protocol. `fusion_without_dit.yaml` changes the feature reconstruction branch only. The code does not include a completed ablation study, statistical uncertainty estimate, cross-validation result or claimed paper table.

## Resume

Resume `last.pt` in its original output directory, retaining the previously selected `best.pt`:

```bash
tlsseg train --config configs/patent.yaml --manifest /path/to/manifest.csv --output runs/patent_seed42 --resume runs/patent_seed42/last.pt --epochs 100 --device cuda
```

Resume loads the native and added-branch weights from `last.pt`; do not pass `--native-checkpoint` again. Source files and the manifest checksum must match the checkpoint; only the epoch limit may change in the YAML configuration. Batch limits are recorded per epoch; a resumed run inherits any prior partial-smoke designation even if limits are removed. A new data preprocessing version, source edit or hyperparameter setting should use a new run directory. The data loader uses a reproducible seed per epoch, worker seeding and no persistent workers. Training, evaluation and prediction use FP32 with autocast/AMP and matmul/cuDNN TF32 explicitly disabled. These precision settings are recorded. Deterministic PyTorch algorithms are requested with `warn_only=True`: CUDA interpolation/pooling backward in this architecture can be nondeterministic and emits a warning rather than disabling CUDA training. Bitwise equality across different platforms, GPU models or dependency builds is not promised. The automatic device choice uses CUDA when available and CPU otherwise; MPS is opt-in.

By default, evaluation requires the same manifest SHA256 used for training. For a genuinely new cohort, pass `--external-cohort`; the selected patients' ID hashes must not occur anywhere in the original manifest. This protects against relabeling the known training patients as test patients in a new CSV. It cannot detect an individual assigned a different ID or copied images; data-owner identity audits remain necessary. The report distinguishes original-manifest evaluation from external-cohort evaluation.

## Checkpoint compatibility

The repository checkpoint schema stores the model configuration and explicit architecture contract; loading uses `weights_only=True` and strict tensor-key checking. No model weights are downloaded implicitly. The patent training config requires a local plain SMP U-Net initialization checkpoint, while the synthetic-smoke config explicitly allows random initialization; the resulting repository checkpoints record the selected policy and any native-source file SHA256. Historical full models containing decoder-mixed adapters or encoder-skip gates do not match this architecture and must not be renamed to `best.pt` for reuse.

## Probability and area record

Archive the prediction checkpoint checksum and `preparation.json` with the actual reading/grid metadata. Prepared manifests record basic-level positions, reading level, source read size, output downsample and masks. `predict` retains and checks those records and produces per-slide fusion layouts automatically. Compare unique covered tissue pixels to eligible tissue pixels before claiming whole-slide coverage. Include both x/y physical scales and all resize factors. Verify the source of the tissue mask and describe the denominator of each reported area fraction. Pixel-level or area accuracy is not established by the software's known-answer arithmetic tests.

## Release contents

The submission archive includes source, tests, configs and documentation, with an internal SHA256 file. It omits local virtual environments, data, private weights, caches, output predictions, patient manifests and the patent Word file. Paper-specific title/DOI/repository URL and access instructions can be added once they exist; this package does not fabricate them. The current license notice preserves the unverified upstream rights status rather than granting a new open-source license.
