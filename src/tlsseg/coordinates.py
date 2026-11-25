"""Explicit basic-slide to output-grid geometry without silent coordinate rounding."""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


def finite_number(value, name: str, *, positive: bool = False) -> float:
    """Parse JSON/CSV numeric fields while rejecting bools, NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, str)):
        raise ValueError(f"{name} must be a finite number.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number.") from None
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} number.")
    return number


def integer_number(value, name: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    finite_number(value, name)
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{name} must be an integer.") from None
    if number != number.to_integral_value() or abs(number) > 2**53:
        raise ValueError(f"{name} must be an exactly representable integer.")
    result = int(number)
    if (positive and result <= 0) or (nonnegative and result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


@dataclass(frozen=True)
class OutputGrid:
    """A grid whose pixels have explicitly declared basic-level extents.

    Pixel (x,y)'s top-left basic coordinate is origin + (x*dx,y*dy).
    Pixel-center sampling uses origin + ((x+0.5)*dx,(y+0.5)*dy).
    ``base_to_grid`` accepts aligned tile origins only; probability resampling
    or rounding a fractional origin is not performed by this interface.
    """

    height: int
    width: int
    downsample_x: float
    downsample_y: float
    origin_base_x: float = 0.0
    origin_base_y: float = 0.0

    def __post_init__(self):
        for field in ("height", "width"):
            object.__setattr__(self, field, integer_number(getattr(self, field), field, positive=True))
        for field in ("downsample_x", "downsample_y"):
            object.__setattr__(self, field, finite_number(getattr(self, field), field, positive=True))
        for field in ("origin_base_x", "origin_base_y"):
            object.__setattr__(self, field, finite_number(getattr(self, field), field))

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @classmethod
    def from_base_dimensions(cls, width_base, height_base, downsample_x, downsample_y,
                             origin_base_x=0.0, origin_base_y=0.0):
        width = integer_number(width_base, "width_base", positive=True)
        height = integer_number(height_base, "height_base", positive=True)
        dx = finite_number(downsample_x, "downsample_x", positive=True)
        dy = finite_number(downsample_y, "downsample_y", positive=True)
        ox = finite_number(origin_base_x, "origin_base_x")
        oy = finite_number(origin_base_y, "origin_base_y")
        if ox >= width or oy >= height:
            raise ValueError("The output-grid origin must precede the basic slide's far edge.")
        return cls(math.ceil((height - oy) / dy), math.ceil((width - ox) / dx), dx, dy, ox, oy)

    def base_to_grid(self, base_x, base_y, tolerance=1e-6) -> tuple[int, int]:
        tolerance = finite_number(tolerance, "tolerance")
        if tolerance < 0 or tolerance >= 0.5:
            raise ValueError("Alignment tolerance must be in [0,0.5) grid pixels.")
        coordinates = (
            (finite_number(base_x, "base_x") - self.origin_base_x) / self.downsample_x,
            (finite_number(base_y, "base_y") - self.origin_base_y) / self.downsample_y,
        )
        result = []
        for name, coordinate in zip(("base_x", "base_y"), coordinates):
            if abs(coordinate) > 2**53:
                raise ValueError(f"{name} exceeds exactly representable output-grid coordinates.")
            aligned = round(coordinate)
            if abs(coordinate - aligned) > tolerance:
                raise ValueError(f"{name} is not aligned to the declared output grid; resampling is required.")
            result.append(int(aligned))
        return tuple(result)

    def grid_to_base(self, x, y) -> tuple[float, float]:
        x, y = integer_number(x, "grid_x"), integer_number(y, "grid_y")
        return self.origin_base_x + x * self.downsample_x, self.origin_base_y + y * self.downsample_y
