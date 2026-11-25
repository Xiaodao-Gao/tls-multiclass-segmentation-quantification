# Data and coordinate contract

Use `tlsseg prepare` to create this tile manifest from annotated slides, or supply an existing compatible manifest. The [preparation guide](PREPARATION.md) specifies the cohort, reviewed class map, tissue mask, reading level and scale inputs.

## Tile manifest

Supply a CSV (UTF-8) or JSON list of records with:

- `patient_id`: a nonempty de-identified grouping ID. Keep all slides/tiles from a patient in one split.
- `split`: `train`, `val` or `test`; `predict` is available for unlabeled-only prediction manifests.
- `image_path`: an RGB image path, relative to the manifest or absolute.
- `mask_path`: a scalar class-index mask path for training/evaluation; may be omitted for prediction.
- `tile_id`: a unique optional tile identifier; row-derived IDs are assigned if absent.

`examples/manifest.csv` illustrates the schema; its files are placeholders. Generate a complete artificial dataset with `scripts/make_synthetic_data.py`.

```json
[
  {"patient_id":"example_A","split":"train","image_path":"images/a.png","mask_path":"masks/a.png","tile_id":"a"},
  {"patient_id":"example_B","split":"val","image_path":"images/b.png","mask_path":"masks/b.png","tile_id":"b"}
]
```

Duplicate image paths and overlapping patient IDs between splits are rejected. Path checks resolve relative paths and symlinks; these checks cannot detect images copied under another path or inconsistent patient identifiers. The dataset owner must audit those cases. Masks must have the same geometry as the 224×224 image and contain only `0,1,2,3,255`. RGB color masks are rejected; export scalar labels explicitly. There is no silent image or mask resizing. Training and evaluation require masks for every selected row; prediction does not read masks.

Class IDs are background (0), lymphocyte aggregate (1), primary follicle-like (2) and secondary follicle-like (3). Ignore=255. Before supplying tiles, define the slide reading level and physical scale consistently. Training augmentation applies synchronized 90-degree rotation and horizontal/vertical flips to images/masks, and image-only brightness/contrast/color factors in [0.8,1.2]. RGB floats are divided by 255 and normalized with mean `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)`.

## Predictions

`tlsseg predict` exports one NPZ per tile containing `probabilities` with axes `[class,y,x]`, shape `[4,224,224]`, and probabilities summing to one along class. A scalar-index PNG contains argmax class IDs. `index.json` maps exported numeric filenames to tile IDs; it does not infer slide coordinates from names. Do not use color-rendered images as probability inputs.

## Stitch layout

The JSON layout declares `shape: [height,width]` and a list of tiles with `probability_path`, integer `x`, integer `y`. These are **output grid coordinates**, not arbitrary WSI basic-level coordinates. Each tile pixel must represent one output grid pixel; the stitcher does not silently resize probabilities. Tiles extending beyond the declared grid are clipped. Optional `weights_path` and `valid_mask_path` are scalar arrays aligned to the tile. A `tissue_mask` is a two-dimensional PNG/NPY/NPZ aligned to the whole declared output grid; positive values denote eligible tissue.

Optional `metadata` must provide all four positive finite values to calculate square millimeters:

```json
{"mpp_x":0.25,"mpp_y":0.25,"downsample_x":4.0,"downsample_y":4.0}
```

The numbers above are an example, not scanner defaults. The MPP values describe the basic slide level, while downsample factors map the final output grid to that level. If a tile read at downsample 4 is later resized to half its width/height, the effective output factors are 8 rather than 4. Different x/y factors are supported. Missing metadata yields unique pixel counts only; invalid or incomplete metadata must not be used to invent physical scale.

`fused.npz` stores fused probabilities, prediction indices and coverage; pixels outside eligible covered tissue have prediction `-1`. `quantification.json` reports eligible/covered pixels and class areas/fractions. Review coverage before reporting slide-wide results. `prepare` can read slide regions and rasterize reviewed pathologist polygons. Its manifest preserves basic-level coordinates, grid positions, reading scale and valid/tissue masks; `predict` validates these values and exports per-slide layouts automatically. Fractional unaligned tile origins require explicit probability resampling and are rejected. The entire stitching accumulator must fit in memory.

## Data availability for a manuscript

The repository includes artificial examples only. Supply the study-specific cohort access route, annotation protocol, patient inclusion/exclusion criteria, patient-level split rule and approved data-use terms in the manuscript. Keep identifiable metadata and real manifests outside the repository. No public patient dataset URL or access approval is asserted by this release.
