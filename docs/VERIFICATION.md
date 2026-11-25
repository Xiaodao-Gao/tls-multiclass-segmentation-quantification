# Verification record

The final software-check counts and environment are recorded in `verification.json`. Tests use artificial tensors and generated images only; no patient images, study manifest or trained research checkpoint are included or used for this release check.

Checks cover:

- Actual EfficientNet-B3 encoder stage channels/spatial sizes and the single final-decoder injection.
- Exact native U-Net equality at zero gate, nonzero-gate effect, finite gradients through the branch after two optimizer steps, deterministic evaluation and strict weight roundtrip.
- Known-answer sampled-noise construction/target and clean-feature, single-call t=0 inference without sampling.
- Relative manifests, patient leakage and duplicate input rejection, scalar label IDs, geometry, aligned augmentation and ignore-255 handling.
- Global confusion metrics, undefined denominators and foreground macro definitions.
- Probability fusion before argmax, overlapping unique-pixel counts, tissue/coverage exclusion, anisotropic physical scale and missing/invalid metadata.
- Synthetic train → checkpoint → evaluate → predict → stitch, explicit test evaluation, rejection of relabeled known training patients and preservation of the partial-run label after resume.
- Reviewed polygon holes, foreground-class conflict/ignore precedence, pixel-center basic-coordinate rasterization and deterministic patient-level split allocation.
- Region reading, level-zero location versus reading-level size, RGBA/white padding, missing/invalid MPP, and rejection of an ordinary-image fallback for pyramidal WSI.
- Slide preparation → registered manifest → automatic prediction layouts → probability fusion, valid/tissue coverage and known-answer physical area with all resize factors retained.

The direct dependencies were checked with `pip check`. An installable wheel was built with `pip wheel . --no-deps`. The submission ZIP contains tracked source/config/documentation only and an internal `MANIFEST.sha256`; the archive is separately checksummed and inspected after creation.

These checks establish software contracts, not a paper's accuracy, clinical utility or whole-slide biological quantification validity. Slide tests use synthetic ordinary images and mocked WSI reader APIs; no proprietary vendor-slide accuracy is claimed. GPU/CUDA and MPS execution have not been tested in this local environment. No remote CI result is asserted before GitHub executes the included workflow. The manuscript-aligned architecture must still be trained and evaluated on the study cohort before reporting performance.
