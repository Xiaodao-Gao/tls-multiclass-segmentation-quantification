"""Region-based slide reading with explicit basic-level coordinates.

OpenSlide reads real whole-slide images without decoding the whole image into
RAM. Pillow is a single-level fallback for ordinary PNG/JPEG/untiled TIFF test
images; it is not a replacement WSI pyramid reader. No physical scale is
inferred from an image's pixel dimensions or conventional print-DPI metadata.
"""
from __future__ import annotations

import importlib
import math
import numbers
from pathlib import Path

from PIL import Image

_ORDINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_PILLOW_FORMATS = {"PNG", "JPEG", "TIFF"}


def _integer(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _pair(value, name: str, *, positive: bool = False) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two integers.")
    return tuple(_integer(v, name, positive=positive) for v in value)


def _mpp(properties: dict, name: str) -> float | None:
    value = properties.get(name)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Slide property {name} must be a positive finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Slide property {name} must be a positive finite number.") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Slide property {name} must be a positive finite number.")
    return result


def _rgb_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def _openslide_module(required: bool):
    try:
        return importlib.import_module("openslide")
    except (ImportError, OSError) as exc:
        if required:
            raise RuntimeError(
                "OpenSlide is required for this slide. Install the WSI extra "
                "with python -m pip install -e '.[wsi]'."
            ) from exc
        return None


class SlideReader:
    """Context-managed adapter; locations use level-zero basic-slide pixels.

    Region sizes are pixels of the requested pyramid level. ``dimensions`` are
    level-zero (width, height), and each entry of ``level_downsamples`` maps a
    level pixel to a basic-level pixel. Returned images are always RGB, with
    transparent or out-of-bounds regions composited over white.
    """

    def __init__(self, reader, backend: str, properties: dict | None = None):
        self._reader = reader
        self.backend = backend
        self._closed = False
        try:
            if backend == "pillow":
                self.dimensions = _pair(reader.size, "Slide dimensions", positive=True)
                self.level_downsamples = (1.0,)
            else:
                self.dimensions = _pair(reader.dimensions, "Slide dimensions", positive=True)
                self.level_downsamples = tuple(float(v) for v in reader.level_downsamples)
                if not self.level_downsamples or any(
                    not math.isfinite(v) or v <= 0 for v in self.level_downsamples
                ) or not math.isclose(self.level_downsamples[0], 1.0):
                    raise ValueError("Slide pyramid downsample factors must be positive, finite and start at 1.")
            self.properties = dict(properties or {})
            self.mpp_x = _mpp(self.properties, "openslide.mpp-x")
            self.mpp_y = _mpp(self.properties, "openslide.mpp-y")
        except Exception:
            self.close()
            raise

    def read_region(self, location, level: int, size) -> Image.Image:
        if self._closed:
            raise ValueError("Slide reader is closed.")
        x, y = _pair(location, "Basic-level location")
        width, height = _pair(size, "Region size", positive=True)
        level = _integer(level, "Pyramid level")
        if level < 0 or level >= len(self.level_downsamples):
            raise ValueError("Pyramid level is outside this slide's available levels.")
        if self.backend == "openslide":
            # OpenSlide's location is always level zero, while size is at level.
            result = self._reader.read_region((x, y), level, (width, height))
            if result.size != (width, height):
                raise RuntimeError("Slide reader returned an unexpected region size.")
            return _rgb_on_white(result)
        # Pillow's ordinary-image fallback has exactly one level. Intersect the
        # crop explicitly because RGB Image.crop pads outside pixels with black.
        output = Image.new("RGB", (width, height), "white")
        left, top = max(0, x), max(0, y)
        right = min(self.dimensions[0], x + width)
        bottom = min(self.dimensions[1], y + height)
        if left < right and top < bottom:
            crop = self._reader.crop((left, top, right, bottom))
            output.paste(_rgb_on_white(crop), (left - x, top - y))
        return output

    def close(self):
        if not self._closed:
            self._closed = True
            self._reader.close()

    def __enter__(self):
        if self._closed:
            raise ValueError("Slide reader is closed.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def _pillow_reader(path: Path) -> SlideReader:
    source = Image.open(path)
    try:
        if source.format not in _PILLOW_FORMATS:
            raise ValueError("Pillow fallback supports only ordinary PNG, JPEG and untiled TIFF images.")
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Multiframe/pyramidal images require the OpenSlide backend.")
        if source.format == "TIFF" and any(tag in source.tag_v2 for tag in (322, 323)):
            raise ValueError("Tiled TIFF images require the OpenSlide backend.")
        limit = Image.MAX_IMAGE_PIXELS
        if limit is not None and source.width * source.height > limit:
            raise ValueError("Image exceeds Pillow's ordinary-image pixel limit; use OpenSlide for WSI reading.")
        return SlideReader(source, "pillow")
    except Exception:
        source.close()
        raise


def open_slide(path, backend: str = "auto") -> SlideReader:
    """Open a slide without silently substituting a failed WSI read.

    ``auto`` asks OpenSlide to identify the file when the optional dependency is
    installed. A recognized WSI is opened with OpenSlide and all I/O errors are
    propagated. Only an unrecognized ordinary-image suffix can fall back to
    Pillow. ``openslide`` and ``pillow`` explicitly select one backend.
    """
    if backend not in {"auto", "openslide", "pillow"}:
        raise ValueError("backend must be auto, openslide or pillow.")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if backend == "pillow":
        return _pillow_reader(path)
    module = _openslide_module(required=backend == "openslide")
    if module is not None:
        recognized = backend == "openslide" or module.OpenSlide.detect_format(str(path)) is not None
        if recognized:
            source = module.OpenSlide(str(path))
            try:
                properties = dict(source.properties)
            except Exception:
                source.close()
                raise
            return SlideReader(source, "openslide", properties)
    if backend == "openslide" or path.suffix.lower() not in _ORDINARY_SUFFIXES:
        if module is None:
            _openslide_module(required=True)
        raise ValueError("File is not recognized by OpenSlide; no ordinary-image fallback is available.")
    return _pillow_reader(path)
