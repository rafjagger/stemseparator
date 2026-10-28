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
    return np.stack([result[name] for name in STEM_NAMES]).astype(np.float64)


def average_stems(results: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Plain mean over models; (4, channels, frames) float32 in STEM_NAMES order."""
    if not results:
        raise ValueError("no model results to average")
    stacks = [_stacked(result) for result in results]
    shapes = {stack.shape for stack in stacks}
    # MSST does no alignment either: models that disagree on length are a bug, not a trim.
    if len(shapes) != 1:
        raise ValueError(f"model results differ in shape: {sorted(shapes)}")
    return np.mean(stacks, axis=0).astype(np.float32)
