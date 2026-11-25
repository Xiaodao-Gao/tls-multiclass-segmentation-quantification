# Changes

## 0.2.0 — 2026-11-25

- Name the project TLS Multi-class Segmentation and Quantification and make the four pixel classes explicit in prediction and quantification records.
- Implement reviewed QuPath GeoJSON rasterization, polygon holes/conflict/ignore handling, slide-region reading and patient-level 70:20:10 split generation with seed 42.
- Retain basic-level coordinates, reading level, actual resize factors and valid/tissue masks from preparation through prediction; automatically export and validate per-slide fusion layouts.
- Preserve the single encoder-four-scale fusion, feature-reconstruction, final-gate and native-output-head model path and its mathematics.

## 0.1.1 — 2026-11-25

- Audit the model against the accepted final patent text and Figure 3: encoder-only four-scale fusion, one feature reconstruction, final-decoder channel gate and the native output head remain the sole main segmentation path.
- Align the training recipe to native/new-component learning rates 5e-6/3e-4 and require an explicit native U-Net initialization checkpoint; keep random initialization in a separately named synthetic software-check config.
- Enforce FP32 execution with automatic mixed precision and both TF32 paths disabled for training, evaluation and tile prediction.
- Use English-only repository documentation, citation metadata and runtime messages.

## 0.1.0 — 2026-11-25

- Integrate four EfficientNet-B3 encoder features with the supplied multi-scale fusion and feature denoiser, then use one final-decoder channel-gated residual and the native four-class output head.
- Provide standalone training, frozen evaluation, tile probability prediction and declared-grid probability fusion/area quantification.
- Add patient-disjoint manifests, reproducibility metadata, strict checkpoint schemas, example configs, synthetic checks, English/Chinese documentation and a source-only submission archive builder.

This version does not carry over performance claims or full branch weights from either historical alternative architecture.
