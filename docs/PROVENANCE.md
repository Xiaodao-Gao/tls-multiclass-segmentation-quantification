# Source provenance and integration changes

The author supplied a frozen research snapshot named `decoder_refine_search_20261008/code`. Its seven Python files matched the attached `code_sha256.json` when inspected for this preparation. They were not edited in place. The complete historical snapshot, data and weights are not copied into this repository.

Important input source checksums:

```text
frozen_base/tls_ssfp_dit.py  4818c73a391d6772c62fbb7d59b99d902442ba2c7e02aed8cf64c591619a5f4e
frozen_fresh_model.py       40bcd05f487488b9ec34616cf1a05e7b4fc476ce1b9cb89b3a9153c9bd5a2f3f
model.py                    2201a4c2a26e5d24628796e7b4adc7c7ab849eb20641ed603bfd7305db517c19
frozen_base/train_segmentation.py  76a44388c4d44888dbb74a423e1b3356e3cf5e7201238e2392887a4738403c11
```

`models/components.py` retains the supplied fusion-attention, GroupNorm, noise schedule and DiT component definitions; its source checksum is also embedded in that module. The source snapshot describes the branch as inherited from an earlier "009" implementation. A complete original upstream repository/license was not present. This repository preserves that provenance and does not assert that these pre-existing components are newly invented here.

`models/segmenter.py` newly integrates four **encoder** tensors (32/48/136/384 channels) with the supplied fusion computation and the **single final decoder injection** (32→16 projection, upsample and learned channel gate). No supplied whole-model file had exactly this combination:

- The old `model.py` used decoder channels 64/128/256 plus deepest encoder 384, followed by final decoder injection.
- The old `frozen_fresh_model.py` used the four encoder tensors but injected into four encoder skips.
- The old base model used a different backbone and a direct pixel classification head.

Therefore this release is a patent-governed reimplementation assembled from the supplied components. It is not an exact reproduction of any earlier complete trained model. Its branch checkpoints and previous experimental metrics are not interchangeable with those historical variants. Only a compatible pure native U-Net initialization can be strictly shared.

The standalone CLI, checkpoint schema, source/environment logging, generic manifest handling and software tests were prepared for this release. Data augmentation, losses and global confusion metrics follow the supplied training code's declared conventions. Coordinate-based probability fusion and area quantification are new utilities, not recovered historical WSI code. They are checked using artificial known-answer inputs; WSI extraction, annotation conversion and scanner calibration are external prerequisites.

Third-party runtime libraries remain subject to their own licenses and citations. Primary upstream projects include [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision), [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch), [timm](https://github.com/huggingface/pytorch-image-models), [einops](https://github.com/arogozhnikov/einops), [NumPy](https://github.com/numpy/numpy), [Pillow](https://github.com/python-pillow/Pillow) and [PyYAML](https://github.com/yaml/pyyaml). Cite the relevant original methods and verify component permissions in the final manuscript/release. No third-party library is vendored in the ZIP.
