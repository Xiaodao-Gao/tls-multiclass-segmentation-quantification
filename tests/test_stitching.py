import json

import numpy as np
import pytest

from tlsseg.stitching import ProbabilityStitcher, truncated_hann_weights


def constant_probability(label, shape, confidence=1.0):
    result = np.zeros((4, *shape), dtype=np.float32)
    result[label] = confidence
    if label != 0:
        result[0] = 1 - confidence
    return result


def test_fuses_probabilities_before_argmax_and_counts_unique_pixels():
    stitcher = ProbabilityStitcher((1, 1))
    # Two weak class-1 votes must not overrule one strongly background tile.
    for confidence in (0.51, 0.51, 0.01):
        stitcher.add_tile(constant_probability(1, (1, 1), confidence), 0, 0)
    result = stitcher.finalize()
    assert result.prediction[0, 0] == 0
    assert result.probabilities[1, 0, 0] == pytest.approx(1.03 / 3)
    assert result.summary["class_pixel_counts"]["Background"] == 1
    assert result.summary["covered_tissue_pixels"] == 1
    assert result.summary["tiles_added"] == 3
    assert result.summary["tissue_mask_provided"] is False
    assert "caller declares the entire grid eligible" in result.summary["area_fraction_denominator"]


def test_explicit_weights_affect_probabilities_before_classification():
    stitcher = ProbabilityStitcher((1, 1))
    stitcher.add_tile(constant_probability(1, (1, 1), 0.8), 0, 0, weights=np.array([[3.0]]))
    stitcher.add_tile(constant_probability(2, (1, 1), 0.9), 0, 0, weights=np.ones((1, 1)))
    result = stitcher.finalize()
    assert result.prediction.item() == 1
    assert result.probabilities[:, 0, 0].tolist() == pytest.approx([0.175, 0.6, 0.225, 0])


def covered_tissue_result(metadata=None):
    tissue = np.zeros((3, 4), dtype=bool)
    tissue[:2, :3] = True
    valid = np.ones((2, 3), dtype=bool)
    valid[0, 0] = False
    stitcher = ProbabilityStitcher((3, 4), tissue_mask=tissue)
    stitcher.add_tile(constant_probability(1, (2, 3)), 0, 0, valid_mask=valid)
    return stitcher.finalize(metadata)


def test_tissue_and_coverage_define_area_denominator():
    result = covered_tissue_result()
    assert result.summary["covered_tissue_pixels"] == 5
    assert result.summary["total_tissue_pixels"] == 6
    assert result.summary["tissue_coverage_fraction"] == pytest.approx(5 / 6)
    assert result.summary["foreground_area_fraction"] == 1.0
    assert result.summary["class_pixel_counts"]["Agg"] == 5
    assert result.summary["tissue_mask_provided"] is True
    assert result.summary["area_fraction_denominator"] == "Covered eligible tissue-mask pixels on the unique output grid."
    assert np.all(result.prediction[~result.coverage] == -1)
    assert np.all(result.probabilities[:, ~result.coverage] == 0)
    assert not result.summary["physical_area_available"]
    assert "class_area_mm2" not in result.summary
    assert "structure_count" not in result.summary
    json.dumps(result.summary, allow_nan=False)


def test_partial_scale_never_defaults_to_physical_area():
    result = covered_tissue_result({"mpp_x": 0.5, "mpp_y": 0.8, "downsample_x": 2})
    assert not result.summary["physical_area_available"]
    assert result.summary["missing_physical_metadata"] == ["downsample_y"]
    assert "output_pixel_area_mm2" not in result.summary


def test_anisotropic_physical_scale_uses_both_axis_factors():
    result = covered_tissue_result({"mpp_x": 0.5, "mpp_y": 0.8, "downsample_x": 2, "downsample_y": 4})
    assert result.summary["physical_area_available"]
    assert result.summary["output_pixel_area_mm2"] == pytest.approx(3.2e-6)
    assert result.summary["class_area_mm2"]["Agg"] == pytest.approx(16e-6)
    assert result.summary["covered_tissue_area_mm2"] == pytest.approx(16e-6)
    assert result.summary["total_tissue_area_mm2"] == pytest.approx(19.2e-6)


def test_partial_edge_tile_is_clipped_in_the_correct_grid_coordinates():
    stitcher = ProbabilityStitcher((2, 3))
    stitcher.add_tile(constant_probability(2, (2, 2)), x=2, y=1)
    result = stitcher.finalize()
    assert result.summary["covered_tissue_pixels"] == 1
    assert result.prediction[1, 2] == 2
    assert np.count_nonzero(result.prediction >= 0) == 1


def test_positive_truncated_hann_edges_and_single_pixel_window():
    weights = truncated_hann_weights((224, 224))
    assert weights[0, 0] == pytest.approx(0.05**2)
    assert np.all(weights > 0)
    assert weights[112, 112] > weights[0, 0]
    assert truncated_hann_weights((1, 1)).item() == 1


@pytest.mark.parametrize("bad", [np.ones((3, 2, 2)), np.ones((4, 2, 2)), np.full((4, 2, 2), np.nan), np.full((4, 2, 2), -0.1)])
def test_invalid_probability_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        ProbabilityStitcher((2, 2)).add_tile(bad, 0, 0)


def test_no_coverage_and_invalid_metadata_are_errors():
    with pytest.raises(ValueError, match="No covered tissue"):
        ProbabilityStitcher((2, 2)).finalize()
    with pytest.raises(ValueError, match="mpp_y"):
        covered_tissue_result({"mpp_x": 0.5, "mpp_y": 0, "downsample_x": 1, "downsample_y": 1})
    with pytest.raises(ValueError, match="mpp_y"):
        covered_tissue_result({"mpp_y": -0.5})
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        covered_tissue_result([])


@pytest.mark.parametrize("shape", [None, (2,), (2, 0), (2.5, 3), 2])
def test_invalid_grid_shapes_have_explicit_errors(shape):
    with pytest.raises(ValueError):
        ProbabilityStitcher(shape)
    with pytest.raises(ValueError):
        truncated_hann_weights(shape)


def test_noninteger_and_outside_origins_are_rejected():
    stitcher = ProbabilityStitcher((2, 2))
    probability = constant_probability(1, (1, 1))
    with pytest.raises(ValueError, match="integer"):
        stitcher.add_tile(probability, 0.5, 0)
    with pytest.raises(ValueError, match="does not intersect"):
        stitcher.add_tile(probability, 3, 3)
