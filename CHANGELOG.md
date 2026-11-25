# Changes

## 0.1.0 — 2026-11-25

- Integrate four EfficientNet-B3 encoder features with the supplied multi-scale fusion and feature denoiser, then use one final-decoder channel-gated residual and the native four-class output head.
- Provide standalone training, frozen evaluation, tile probability prediction and declared-grid probability fusion/area quantification.
- Add patient-disjoint manifests, reproducibility metadata, strict checkpoint schemas, example configs, synthetic checks, English/Chinese documentation and a source-only submission archive builder.

This version does not carry over performance claims or full branch weights from either historical alternative architecture.
