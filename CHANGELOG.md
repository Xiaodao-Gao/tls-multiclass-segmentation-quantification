# Changes

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
