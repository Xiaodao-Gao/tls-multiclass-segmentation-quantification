# TLS Multi-class Segmentation and Quantification

## Prepare annotated slides for the model

`tlsseg prepare` reads slides, rasterizes reviewed polygon labels, assigns patients to splits and writes 224x224 RGB tiles with spatial metadata. Classification remains pixel-level: background (0), lymphocyte aggregate (1), primary follicle-like (2) and secondary follicle-like (3). Ignore=255. This command does not add a patient-level classifier or change the encoder-fusion, single-step reconstruction and final-decoder output path.

Install the appropriate preparation dependencies from the repository root:

```bash
python -m pip install -e '.[preprocessing]'
# For vendor whole-slide images, including the OpenSlide binary library:
python -m pip install -e '.[wsi]'
tlsseg prepare --help
```

The `preprocessing` extra provides Shapely for polygon handling. The `wsi` extra also provides OpenSlide. `--backend auto` uses OpenSlide for recognized slides and permits Pillow for ordinary PNG, JPEG and single-frame untiled TIFF images. A recognized WSI read error is propagated rather than silently switching readers. `--backend openslide` selects WSI reading explicitly; `--backend pillow` selects the ordinary-image reader, which supports level 0 only and is intended for small images and software examples.

## Cohort input

Supply a UTF-8 CSV or a JSON list of slide records. Every record requires `patient_id`, `slide_id` and `slide_path`. Slide IDs must be unique and use ASCII letters, digits, underscore, dot or hyphen. One slide file may appear only once. Paths are relative to the cohort file or absolute local paths.

Labeled mode additionally requires `annotation_path` for every slide and the CLI option `--class-map`. The default workflow also requires `tissue_mask_path` for each slide. Optional `mpp_x` and `mpp_y` specify measured micrometers per basic-level pixel; provided values override the corresponding OpenSlide properties. Optional `split` preserves a previously frozen assignment.

An example cohort with explicit assignments is:

```csv
patient_id,slide_id,slide_path,annotation_path,tissue_mask_path,mpp_x,mpp_y,split
example_A,slide_A,slides/slide_A.svs,annotations/slide_A.geojson,tissue/slide_A_grid.png,0.25,0.25,train
example_B,slide_B,slides/slide_B.svs,annotations/slide_B.geojson,tissue/slide_B_grid.png,0.25,0.25,val
example_C,slide_C,slides/slide_C.svs,annotations/slide_C.geojson,tissue/slide_C_grid.png,0.25,0.25,test
```

These are schema examples, not patient records or scanner defaults. Replace the scale values with the actual slide calibration. If `split` is supplied, it must be provided for every slide and must be `train`, `val` or `test`; all slides from one patient must have the same value. To construct a new split, omit the field for every record or leave every value empty. Patients are deduplicated, sorted and shuffled with seed 42 by default, then assigned to target proportions 70:20:10. Largest-remainder allocation gives 295/84/42 patients for a 421-patient cohort; small cohorts are adjusted to keep every subset nonempty and require at least three unique patients. Tile counts need not follow the patient proportions. Reproducing an existing study requires its original assignment, not just the same random seed.

## GeoJSON labels

Use QuPath GeoJSON with coordinates in full-resolution, basic-level slide pixels and a top-left origin. Supported containers are FeatureCollection, one Feature or a list of Features; geometries must be valid Polygon or MultiPolygon objects. Polygon holes are preserved. The reader accepts an integer `properties.class_id`, or a QuPath `properties.classification.name` that appears in the class map. If both are present they must agree. Unknown class names and invalid or nonpolygon geometries are rejected.

Create `/path/to/study/class_map.json` with names matching the annotation export, for example:

```json
{
  "Background": 0,
  "Aggregate": 1,
  "Primary": 2,
  "Secondary": 3,
  "Ignore": 255
}
```

Rasterization samples each output pixel at its basic-coordinate center. Points exactly on an outer or hole boundary are excluded. Explicit background polygons supply reviewed background; foreground polygons overwrite background. Overlap between distinct foreground classes becomes 255, same-class overlaps form a union, and explicit ignore regions override all classes. A hole retains an underlying label, or the unannotated value if no reviewed polygon covers it.

The default `--unannotated-label 255` treats unannotated pixels as unknown. Use `--unannotated-label 0` only when the annotation protocol explicitly establishes all otherwise unannotated pixels as reviewed background. This option is a data declaration, not automatic negative-label detection. Pixels outside the slide or the supplied eligible tissue mask are excluded, and labeled tiles containing no valid class pixels are skipped.

## Reading scale and tissue eligibility

The selected pyramid level is `--level`, and `--read-size` is the square region size in pixels of that level. RGB regions are resized bilinearly to 224x224. Labels are rasterized directly on the corresponding output grid; class masks are not bilinearly resized. The effective isotropic scale is:

```text
output_downsample = level_downsample * read_size / 224
output_grid_shape = [ceil(base_height / output_downsample),
                     ceil(base_width / output_downsample)]
```

Default stride is 112 output-grid pixels. Slide reading always receives a basic-level top-left location, while the read size is measured at the selected level. Scale times stride and each tile origin must align to integer basic-level pixels within the explicit numerical tolerance. Incompatible settings raise an error; fractional origins are not silently rounded to another grid. Level 0 with a compatible read size is available when a pyramid level does not align. Reading level, source read size, effective scale and both coordinate systems are saved in the manifest.

`tissue_mask_path` must already be a two-dimensional binary PNG/image or NPY aligned to the **final output grid**, using 0/1 or 0/255. It is not a basic-level mask unless the output scale is 1. No automatic tissue detector or tissue-mask resampling is applied. For example, level 0 with `--read-size 896` has output downsample 4, so its tissue mask must have the grid dimensions calculated above at that scale.

If no tissue mask is supplied, preparation stops unless `--allow-full-grid` explicitly declares the entire grid eligible. That mode is appropriate for a known tissue-only ROI or a software demonstration; it does not establish that every slide pixel is tissue. MPP comes from supplied calibration or OpenSlide properties. Missing calibration remains missing: image dimensions and print DPI are not used to invent MPP, and physical area is reported only when both MPP values and both grid scales are known.

## Labeled preparation and training

With a cohort, reviewed annotation mapping and correctly aligned tissue masks:

```bash
tlsseg prepare --cohort /path/to/study/cohort.csv --class-map /path/to/study/class_map.json --mode labeled --backend openslide --level 0 --read-size 896 --stride 112 --seed 42 --output outputs/prepared_labeled
tlsseg train --config configs/patent.yaml --manifest outputs/prepared_labeled/manifest.csv --native-checkpoint /path/to/weights/native_epoch45.pt --output runs/patent_seed42 --device cuda
```

Choose a new or empty preparation output directory. With the default unknown label 255, labeled candidates retain the global stride lattice around annotation bounds and skip all-ignore tiles; this is a reviewed-region training dataset, not a promise of complete slide coverage. Declaring unannotated pixels as reviewed background with label 0 expands candidates to the full eligible grid. Use prediction preparation for full eligible-grid slide quantification. Output contains `images/`, scalar labels in `masks/`, per-tile eligibility masks in `valid/`, copied grid tissue masks in `metadata/`, `manifest.csv`, `patient_splits.json` and `preparation.json`. The manifest links patients, slides, labels, basic/grid coordinates, reading level, scale and calibration. The preparation record logs input checksums and settings. Preserve these records with the study data outside the source repository.

## Prepare and quantify an unlabeled slide

For prediction mode, use a separate CSV/JSON with `patient_id`, `slide_id`, `slide_path` and `tissue_mask_path`; annotations and a class map are unnecessary. Omit `split` or set it to `predict` for every record. A previously labeled train/val/test cohort cannot be passed unchanged as a prediction-preparation cohort.

```bash
tlsseg prepare --cohort /path/to/study/prediction_cohort.csv --mode predict --backend openslide --level 0 --read-size 896 --stride 112 --output outputs/prepared_prediction
tlsseg predict --checkpoint /path/to/weights/best.pt --manifest outputs/prepared_prediction/manifest.csv --split predict --output outputs/slide_predictions --device cuda
tlsseg stitch --layout outputs/slide_predictions/layout_0000.json --output outputs/slide_quantification
```

Prediction exports four-channel probabilities and pixel-class masks, retains the registered metadata and automatically writes one `layout_XXXX.json` per slide. `index.json` maps slide IDs to layouts; run `stitch` for each needed layout. The layouts use basic-to-output-grid conversion and carry valid/tissue masks and scale metadata. Retain the prepared masks at their referenced paths, or rebase those paths if moving the data.

Stitching fuses overlapping class probabilities with the truncated Hann weights before argmax. It counts each covered eligible grid pixel once and reports class areas/fractions with the declared denominator. Padding and excluded tissue pixels are not counted as background. No connected-component count is interpreted as the number of independent biological structures. The current NumPy stitch accumulator holds the declared grid in memory; choose a grid that fits available RAM.

These preparation algorithms connect the patent's data and coordinate steps to the unchanged model path. Real scanner calibration, pathologist-reviewed annotations, the intended source checkpoint and real-cohort performance validation remain study inputs. Synthetic software tests do not establish clinical accuracy or reproduce historical experimental scores.

Primary interface references: [QuPath annotation export](https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html), [OpenSlide Python](https://openslide.org/api/python/) and [Shapely pixel-center containment](https://shapely.readthedocs.io/en/2.1.2/reference/shapely.contains_xy.html).
