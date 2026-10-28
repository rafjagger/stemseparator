# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Averaging several models' stems into one set (MSST ensemble.py avg_wave, all weights 1)."""
import numpy as np

from stemseparator.naming import STEM_NAMES


def _stacked(result: dict[str, np.ndarray]) -> np.ndarray:
    missing = [name for name in STEM_NAMES if name not in result]
    if missing:
        raise ValueError(f"model result lacks stem(s) {', '.join(missing)}")
    shapes = {result[name].shape for name in STEM_NAMES}
    if len(shapes) != 1:
        raise ValueError(f"stems of one model differ in shape: {sorted(shapes)}")
    return np.stack([result[name] for name in STEM_NAMES]).astype(np.float32, copy=False)


class StemAccumulator:
    """A running float32 sum, so a long track holds one set of stems, not one per model."""

    def __init__(self):
        self._sum: np.ndarray | None = None
        self._count = 0

    def add(self, result: dict[str, np.ndarray]) -> None:
        stacked = _stacked(result)
        if self._sum is None:
            self._sum = stacked
        elif stacked.shape != self._sum.shape:
            # MSST does no alignment either: models that disagree on length are a bug, not a trim.
            raise ValueError(f"model results differ in shape: {self._sum.shape} vs {stacked.shape}")
        else:
            self._sum += stacked
        self._count += 1

    def mean(self) -> np.ndarray:
        """(4, channels, frames) float32 in STEM_NAMES order; divides the sum in place."""
        if self._sum is None:
            raise ValueError("no model results to average")
        self._sum /= np.float32(self._count)
        return self._sum


def average_stems(results: list[dict[str, np.ndarray]]) -> np.ndarray:
    accumulator = StemAccumulator()
    for result in results:
        accumulator.add(result)
    return accumulator.mean()
