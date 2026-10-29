# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turning a model set's runs into one set of four stems; pure numpy, the models are injected."""
from typing import Callable

import numpy as np

from stemseparator.naming import STEM_NAMES
from stemseparator.remote.ensemble import StemAccumulator
from stemseparator.remote.registry import AveragedSet, ModelSet, SpecialistSet

RunModel = Callable[[str], dict[str, np.ndarray]]


def _averaged(model_set: AveragedSet, run: RunModel) -> np.ndarray:
    accumulator = StemAccumulator()
    for model in model_set.models:
        accumulator.add(run(model))
    return accumulator.mean()


def _picked_stem(result: dict[str, np.ndarray], stem: str, model: str,
                 mix: np.ndarray) -> np.ndarray:
    if stem not in result:
        raise ValueError(f"model {model} returned no {stem} stem")
    if result[stem].shape != mix.shape:
        raise ValueError(f"model {model}: {stem} has shape {result[stem].shape}, "
                         f"the mix {mix.shape}")
    return result[stem]


def _specialist(model_set: SpecialistSet, mix: np.ndarray, run: RunModel) -> np.ndarray:
    stems = np.empty((len(STEM_NAMES),) + mix.shape, dtype=np.float32)
    picked_sum = np.zeros(mix.shape, dtype=np.float64)
    for model in model_set.models:
        result = run(model)
        for stem, source in model_set.picks:
            if source != model:
                continue
            picked = _picked_stem(result, stem, model, mix)
            stems[STEM_NAMES.index(stem)] = picked
            picked_sum += picked
        del result  # a 4-stem result is gigabytes on a long track; keep only the picked stems
    # float64 for the subtraction, so the only error left is the final float32 rounding.
    stems[STEM_NAMES.index(model_set.residual)] = mix - picked_sum
    return stems


def separate_with_set(model_set: ModelSet, mix: np.ndarray, run: RunModel) -> np.ndarray:
    """mix: float32 (channels, frames). Returns float32 (4, channels, frames) in STEM_NAMES order."""
    if isinstance(model_set, SpecialistSet):
        return _specialist(model_set, mix, run)
    return _averaged(model_set, run)
