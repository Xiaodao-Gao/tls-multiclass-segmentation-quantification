import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tlsseg.slides import open_slide


def test_pillow_basic_coordinates_white_padding_and_no_inferred_mpp(tmp_path):
    path = tmp_path / "ordinary.png"
    Image.new("RGB", (4, 3), (20, 40, 60)).save(path, dpi=(100, 100))
    with open_slide(path, backend="pillow") as slide:
        assert slide.dimensions == (4, 3)
        assert slide.level_downsamples == (1.0,)
        assert slide.mpp_x is None and slide.mpp_y is None
        region = np.asarray(slide.read_region((-1, -1), 0, (4, 4)))
        assert region.shape == (4, 4, 3)
        np.testing.assert_array_equal(region[0], 255)
        np.testing.assert_array_equal(region[:, 0], 255)
        np.testing.assert_array_equal(region[1:, 1:], np.broadcast_to([20, 40, 60], (3, 3, 3)))
        with pytest.raises(ValueError, match="level"):
            slide.read_region((0, 0), 1, (2, 2))
    with pytest.raises(ValueError, match="closed"):
        slide.read_region((0, 0), 0, (1, 1))
    slide.close()


def test_pillow_alpha_composites_over_white(tmp_path):
    path = tmp_path / "alpha.png"
    image = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    image.putpixel((1, 0), (255, 0, 0, 255))
    image.save(path)
    with open_slide(path, "pillow") as slide:
        region = slide.read_region((0, 0), 0, (3, 1))
        assert region.mode == "RGB"
        assert list(region.getdata()) == [(255, 255, 255), (255, 0, 0), (255, 255, 255)]


def test_mock_openslide_preserves_base_location_level_size_and_metadata(tmp_path, monkeypatch):
    path = tmp_path / "slide.svs"
    path.write_bytes(b"synthetic-format-test")
    calls = []

    class NativeSlide:
        dimensions = (1000, 800)
        level_downsamples = (1.0, 4.0)
        properties = {"openslide.mpp-x": "0.26", "openslide.mpp-y": "0.27", "vendor": "synthetic"}

        def __init__(self, filename):
            calls.append(("open", filename))

        @staticmethod
        def detect_format(filename):
            return "synthetic"

        def read_region(self, location, level, size):
            calls.append(("read", location, level, size))
            return Image.new("RGBA", size, (10, 20, 30, 255))

        def close(self):
            calls.append(("close",))

    monkeypatch.setitem(sys.modules, "openslide", SimpleNamespace(OpenSlide=NativeSlide))
    with open_slide(path) as slide:
        assert slide.backend == "openslide"
        assert slide.dimensions == (1000, 800)
        assert slide.level_downsamples == (1.0, 4.0)
        assert slide.mpp_x == 0.26 and slide.mpp_y == 0.27
        assert slide.properties["vendor"] == "synthetic"
        image = slide.read_region((123, 47), 1, (10, 8))
        assert image.mode == "RGB" and image.size == (10, 8)
    assert ("read", (123, 47), 1, (10, 8)) in calls
    assert calls[-1] == ("close",)


def test_recognized_wsi_io_errors_do_not_fall_back_to_pillow(tmp_path, monkeypatch):
    path = tmp_path / "recognized.tif"
    Image.new("RGB", (2, 2)).save(path)

    class BrokenSlide:
        @staticmethod
        def detect_format(filename):
            return "synthetic-wsi"

        def __init__(self, filename):
            raise RuntimeError("WSI I/O failure")

    monkeypatch.setitem(sys.modules, "openslide", SimpleNamespace(OpenSlide=BrokenSlide))
    with pytest.raises(RuntimeError, match="WSI I/O"):
        open_slide(path)


def test_auto_uses_ordinary_fallback_when_not_recognized(tmp_path, monkeypatch):
    path = tmp_path / "ordinary.png"
    Image.new("RGB", (2, 2)).save(path)
    native = SimpleNamespace(detect_format=lambda filename: None)
    monkeypatch.setitem(sys.modules, "openslide", SimpleNamespace(OpenSlide=native))
    with open_slide(path) as slide:
        assert slide.backend == "pillow"


def test_wsi_metadata_io_failure_closes_reader_and_is_not_hidden(tmp_path, monkeypatch):
    path = tmp_path / "slide.svs"
    path.write_bytes(b"synthetic")
    closed = []

    class BrokenMetadata:
        def __init__(self, filename):
            pass

        @property
        def properties(self):
            raise RuntimeError("Metadata I/O failure")

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "openslide", SimpleNamespace(OpenSlide=BrokenMetadata))
    with pytest.raises(RuntimeError, match="Metadata I/O"):
        open_slide(path, "openslide")
    assert closed == [True]


def test_optional_openslide_is_required_for_wsi_but_not_ordinary_png(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openslide", None)
    png = tmp_path / "ordinary.png"
    Image.new("RGB", (2, 2)).save(png)
    with open_slide(png) as slide:
        assert slide.backend == "pillow"
    wsi = tmp_path / "slide.svs"
    wsi.write_bytes(b"synthetic")
    with pytest.raises(RuntimeError, match="WSI extra"):
        open_slide(wsi)


def test_pyramidal_or_multiframe_tiff_rejected_by_pillow(tmp_path):
    path = tmp_path / "pyramid.tiff"
    Image.new("RGB", (8, 8)).save(path, save_all=True, append_images=[Image.new("RGB", (4, 4))])
    with pytest.raises(ValueError, match="Multiframe/pyramidal"):
        open_slide(path, "pillow")


@pytest.mark.parametrize("value", ["nan", "0", "-0.25", "not-a-number", True])
def test_invalid_physical_metadata_is_rejected_without_inventing_scale(tmp_path, monkeypatch, value):
    path = tmp_path / "slide.svs"
    path.write_bytes(b"synthetic")
    closed = []

    class InvalidMPP:
        dimensions = (10, 10)
        level_downsamples = (1.0,)
        properties = {"openslide.mpp-x": value}

        def __init__(self, filename):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "openslide", SimpleNamespace(OpenSlide=InvalidMPP))
    with pytest.raises(ValueError, match="positive finite"):
        open_slide(path, "openslide")
    assert closed == [True]


@pytest.mark.parametrize("location,level,size", [
    ((0.5, 0), 0, (1, 1)), ((True, 0), 0, (1, 1)),
    ((0, 0), -1, (1, 1)), ((0, 0), True, (1, 1)),
    ((0, 0), 0, (0, 1)), ((0, 0), 0, (1.5, 1)),
])
def test_invalid_region_geometry_rejected(tmp_path, location, level, size):
    path = tmp_path / "ordinary.png"
    Image.new("RGB", (2, 2)).save(path)
    with open_slide(path, "pillow") as slide:
        with pytest.raises(ValueError):
            slide.read_region(location, level, size)
