# Reproducing a study with this code

## Software state

Record the repository commit and release ZIP checksum. `pyproject.toml` pins the direct runtime dependencies; `requirements-tested.txt` records the complete locally installed environment excluding this editable package. The original historical training logs only establish PyTorch 2.3.1+cu118/CUDA 11.8 and an SMP 0.5 API expectation; the new environment is not presented as that historical environment.

Run `python -m pytest -q` before the study. The CI workflow runs the same software tests on a CPU runner. Local checks are documented in `VERIFICATION.md`; a configured workflow is not a claim that remote GitHub CI has already run.

## Data and training record

Define patient-disjoint training/validation/test sets before training. Run training with the supplied config and a manifest belonging to the intended cohort. Every run records the manifest SHA256, split counts, architecture contract, dependency versions, source SHA256 values, configuration and device in `run.json`. Patient IDs and input paths belong in the data owner's local manifest; they are not needed in a code release. A manifest checksum records its bytes but does not hash the image/mask files themselves. Archive the data version and preprocessing code separately under the study's data governance.

`best.pt` is selected solely by validation global foreground macro Dice. `last.pt` also records the optimizer, epoch, history and Python/NumPy/PyTorch RNG states. Training does not evaluate test images. The explicit evaluate command performs no calibration or parameter fitting. To claim a held-out result, freeze model/config choices before inspecting the test metrics, and report the number of patients/tiles and foreground class support.

Use multiple preregistered seeds for performance estimates. Re-run the full architecture and any ablation on the same data/splits/initialization protocol. `fusion_without_dit.yaml` changes the feature reconstruction branch only. The code does not include a completed ablation study, statistical uncertainty estimate, cross-validation result or claimed paper table.

## Resume

Resume `last.pt` in its original output directory, retaining the previously selected `best.pt`:

```bash
tlsseg train --config configs/patent.yaml --manifest /path/to/manifest.csv --output runs/patent_seed42 --resume runs/patent_seed42/last.pt --epochs 100 --device cuda
```

Source files and the manifest checksum must match the checkpoint; only the epoch limit may change in the YAML configuration. Batch limits are recorded per epoch; a resumed run inherits any prior partial-smoke designation even if limits are removed. A new data preprocessing version, source edit or hyperparameter setting should use a new run directory. The data loader uses a reproducible seed per epoch, worker seeding and no persistent workers. Deterministic PyTorch algorithms are requested with `warn_only=True`: CUDA interpolation/pooling backward in this architecture can be nondeterministic and emits a warning rather than disabling CUDA training. Bitwise equality across different platforms, GPU models or dependency builds is not promised. The automatic device choice uses CUDA when available and CPU otherwise; MPS is opt-in.

By default, evaluation requires the same manifest SHA256 used for training. For a genuinely new cohort, pass `--external-cohort`; the selected patients' ID hashes must not occur anywhere in the original manifest. This protects against relabeling the known training patients as test patients in a new CSV. It cannot detect an individual assigned a different ID or copied images; data-owner identity audits remain necessary. The report distinguishes original-manifest evaluation from external-cohort evaluation.

## Checkpoint compatibility

The repository checkpoint schema stores the model configuration and explicit architecture contract; loading uses `weights_only=True` and strict tensor-key checking. The default model starts with random weights and performs no implicit download. A local plain SMP U-Net state dictionary can initialize the native network using `--native-checkpoint`; its SHA256 is recorded. Historical full models containing decoder-mixed adapters or encoder-skip gates do not match this architecture and must not be renamed to `best.pt` for reuse.

## Probability and area record

Archive the prediction checkpoint checksum and the actual extraction/grid metadata. Compare unique covered tissue pixels to eligible tissue pixels before claiming whole-slide coverage. Include both x/y physical scales and all resize factors. Verify the source of the tissue mask and describe the denominator of each reported area fraction. Pixel-level or area accuracy is not established by the software's known-answer arithmetic tests.

## Release contents

The submission archive includes source, tests, configs and documentation, with an internal SHA256 file. It omits local virtual environments, data, private weights, caches, output predictions, patient manifests and the patent Word file. Paper-specific title/DOI/repository URL and access instructions can be added once they exist; this package does not fabricate them. The current license notice preserves the unverified upstream rights status rather than granting a new open-source license.
