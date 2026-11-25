"""Deterministic, patient-disjoint train/validation/test assignment."""
from __future__ import annotations

import math
import numbers
import random
from collections.abc import Iterable, Sequence


def split_patients(
    patient_ids: Iterable[str],
    seed: int = 42,
    fractions: Sequence[float] = (0.7, 0.2, 0.1),
) -> dict[str, str]:
    """Assign unique patients using largest-remainder counts and seeded shuffle.

    Normalize IDs to stripped strings, deduplicate them, sort before shuffling,
    then assign train/val/test in that order. Counts use largest remainders
    (split order breaks ties). If a small cohort gives an empty split, transfer
    one patient from the most over-allocated split with at least two patients.
    At least three patients and strictly positive fractions summing to one are
    required. For 421 patients the default counts are 295/84/42.

    This constructs a new split; reproducing a previously frozen cohort still
    requires that study's original patient assignment rather than just its seed.
    """
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if isinstance(patient_ids, (str, bytes)):
        raise ValueError("patient_ids must be an iterable of patient identifiers.")
    patients = set()
    for value in patient_ids:
        if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)) or not str(value).strip():
            raise ValueError("Patient identifiers must be nonempty strings or integers.")
        patients.add(str(value).strip())
    ordered = sorted(patients)
    if len(ordered) < 3:
        raise ValueError("At least three unique patients are required for nonempty train/val/test splits.")
    try:
        ratios = tuple(fractions)
    except TypeError:
        raise ValueError("fractions must contain train/val/test proportions.") from None
    if len(ratios) != 3 or any(
        isinstance(value, bool) or not isinstance(value, numbers.Real) or
        not math.isfinite(value) or value <= 0 for value in ratios
    ) or not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("fractions must contain three positive finite values summing to one.")
    expected = [len(ordered) * float(value) for value in ratios]
    counts = [math.floor(value) for value in expected]
    missing = len(ordered) - sum(counts)
    remainder_order = sorted(range(3), key=lambda i: (-(expected[i] - counts[i]), i))
    for index in remainder_order[:missing]:
        counts[index] += 1
    for index in range(3):
        if counts[index] == 0:
            donor = max((i for i in range(3) if counts[i] > 1), key=lambda i: (counts[i] - expected[i], counts[i], -i))
            counts[donor] -= 1
            counts[index] += 1
    random.Random(int(seed)).shuffle(ordered)
    assignments = {}
    offset = 0
    for split, count in zip(("train", "val", "test"), counts):
        assignments.update((patient, split) for patient in ordered[offset:offset + count])
        offset += count
    return assignments
