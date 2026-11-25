import json
from collections import Counter

import numpy as np
import pytest

from tlsseg.annotations import load_annotations, rasterize_tile
from tlsseg.splits import split_patients


def feature(label, shell, holes=(), *, name=None):
    properties = {"classification": {"name": name}} if name else {"class_id": label}
    return {"type": "Feature", "properties": properties,
            "geometry": {"type": "Polygon", "coordinates": [shell, *holes]}}


def rectangle(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]


def load(tmp_path, features, mapping=None):
    path = tmp_path / "annotations.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return load_annotations(path, {} if mapping is None else mapping)


def test_polygon_holes_retain_reviewed_background(tmp_path):
    annotations = load(tmp_path, [
        feature(0, rectangle(0, 0, 6, 6)),
        feature(2, rectangle(1, 1, 5, 5), [rectangle(2, 2, 4, 4)]),
    ])
    mask = rasterize_tile(annotations, shape=(6, 6))
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask == 2) == 12
    assert np.count_nonzero(mask == 0) == 24
    assert np.all(mask[2:4, 2:4] == 0)


def test_foreground_conflicts_ignore_and_same_class_union_are_order_invariant(tmp_path):
    foreground = feature(1, rectangle(1, 1, 4, 4))
    annotations = load(tmp_path, [
        feature(0, rectangle(0, 0, 7, 7)), foreground, foreground,
        feature(2, rectangle(3, 1, 6, 4)), feature(255, rectangle(0, 0, 1, 1)),
    ])
    mask = rasterize_tile(annotations, shape=(7, 7))
    np.testing.assert_array_equal(mask, rasterize_tile(reversed(annotations), shape=(7, 7)))
    assert np.all(mask[1:4, 3] == 255)
    assert mask[0, 0] == 255
    assert mask[1, 1] == 1 and mask[1, 4] == 2 and mask[6, 6] == 0


def test_base_origin_anisotropic_scale_and_qupath_name_mapping(tmp_path):
    annotations = load(tmp_path, [feature(None, rectangle(10, 20, 14, 23), name="Primary")], {"Primary": 2})
    mask = rasterize_tile(annotations, origin_base=(10, 20), downsample=(2, 3), shape=(2, 3))
    np.testing.assert_array_equal(mask, [[2, 2, 255], [255, 255, 255]])


def test_multipolygon_and_center_boundary_rule(tmp_path):
    polygons = [rectangle(0, 0, 1, 1), rectangle(2, 2, 3, 3)]
    raw = feature(3, polygons[0])
    raw["geometry"] = {"type": "MultiPolygon", "coordinates": [[p] for p in polygons]}
    annotations = load(tmp_path, [raw])
    mask = rasterize_tile(annotations, shape=(3, 3))
    assert mask[0, 0] == mask[2, 2] == 3 and np.count_nonzero(mask == 3) == 2
    boundary = load(tmp_path, [feature(1, rectangle(0.5, 0, 2, 2))])
    assert rasterize_tile(boundary, shape=(1, 1))[0, 0] == 255


def test_unannotated_is_unknown_unless_explicitly_declared_background():
    assert rasterize_tile([], shape=(2, 2)).tolist() == [[255, 255], [255, 255]]
    assert rasterize_tile([], shape=(2, 2), unannotated_label=0).tolist() == [[0, 0], [0, 0]]
    with pytest.raises(ValueError, match="unannotated_label"):
        rasterize_tile([], shape=(2, 2), unannotated_label=1)


@pytest.mark.parametrize("mutation,match", [
    (lambda f: f["properties"].update(class_id=7), "label"),
    (lambda f: f["properties"].update(class_id=True), "label"),
    (lambda f: f.update(geometry={"type": "Point", "coordinates": [1, 1]}), "Polygon"),
    (lambda f: f["geometry"].update(coordinates=[[[0, 0], [2, 2], [2, 0], [0, 2], [0, 0]]]), "invalid"),
])
def test_invalid_annotations_are_rejected(tmp_path, mutation, match):
    raw = feature(1, rectangle(0, 0, 2, 2))
    mutation(raw)
    with pytest.raises(ValueError, match=match):
        load(tmp_path, [raw])


def test_unmapped_or_conflicting_class_name_is_rejected(tmp_path):
    raw = feature(None, rectangle(0, 0, 2, 2), name="Unreviewed")
    with pytest.raises(ValueError, match="unmapped"):
        load(tmp_path, [raw], {"Other": 1})
    raw["properties"]["class_id"] = 2
    with pytest.raises(ValueError, match="conflicts"):
        load(tmp_path, [raw], {"Unreviewed": 1})


@pytest.mark.parametrize("kwargs", [
    {"downsample": (0, 1)}, {"downsample": (1, float("nan"))},
    {"origin_base": (float("inf"), 0)}, {"shape": (0, 2)}, {"shape": (2.5, 2)},
])
def test_invalid_raster_grid_is_rejected(kwargs):
    with pytest.raises(ValueError):
        rasterize_tile([], **kwargs)


def test_patient_split_matches_counts_is_disjoint_and_order_invariant():
    patients = [f"patient_{i:03d}" for i in range(421)]
    assignment = split_patients(patients)
    assert Counter(assignment.values()) == {"train": 295, "val": 84, "test": 42}
    assert assignment == split_patients(list(reversed(patients)) + patients[:10])
    groups = {split: {p for p, value in assignment.items() if value == split} for split in ("train", "val", "test")}
    assert not groups["train"] & groups["val"] and not groups["train"] & groups["test"] and not groups["val"] & groups["test"]
    assert assignment != split_patients(patients, seed=43)


def test_small_cohort_split_keeps_each_subset_nonempty():
    assert Counter(split_patients(["a", "b", "c"]).values()) == {"train": 1, "val": 1, "test": 1}
    with pytest.raises(ValueError, match="three unique"):
        split_patients(["a", "a", "b"])


@pytest.mark.parametrize("fractions", [(0.7, 0.3, 0), (0.7, 0.2, 0.2), (0.5, 0.5), (0.7, 0.2, float("nan"))])
def test_invalid_patient_split_fractions_are_rejected(fractions):
    with pytest.raises(ValueError, match="fractions"):
        split_patients(["a", "b", "c"], fractions=fractions)
