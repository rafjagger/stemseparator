# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turning a model set's runs into one set of four stems; pure numpy, the models are injected."""
from typing import Callable

import numpy as np

from stemseparator.naming import STEM_NAMES
from stemseparator.remote.ensemble import StemAccumulator
from stemseparator.remote.registry import AveragedSet, ModelSet, SpatialSet, SpecialistSet

# (model name, audio (channels, frames)) -> stem name -> (channels, frames)
RunModel = Callable[[str, np.ndarray], dict[str, np.ndarray]]


def _averaged(model_set: AveragedSet, mix: np.ndarray, run: RunModel) -> np.ndarray:
    accumulator = StemAccumulator()
    for model in model_set.models:
        accumulator.add(run(model, mix))
    return accumulator.mean()


def _picked_stem(result: dict[str, np.ndarray], stem: str, model: str,
                 shape: tuple[int, ...]) -> np.ndarray:
    if stem not in result:
        raise ValueError(f"model {model} returned no {stem} stem")
    if result[stem].shape != shape:
        raise ValueError(f"model {model}: {stem} has shape {result[stem].shape}, "
                         f"the mix {shape}")
    return result[stem]


def _specialist(model_set: SpecialistSet, mix: np.ndarray, run: RunModel) -> np.ndarray:
    stems = np.empty((len(STEM_NAMES),) + mix.shape, dtype=np.float32)
    picked_sum = np.zeros(mix.shape, dtype=np.float64)
    for model in model_set.models:
        result = run(model, mix)
        for stem, source in model_set.picks:
            if source != model:
                continue
            picked = _picked_stem(result, stem, model, mix.shape)
            stems[STEM_NAMES.index(stem)] = picked
            picked_sum += picked
        del result  # a 4-stem result is gigabytes on a long track; keep only the picked stems
    # float64 for the subtraction, so the only error left is the final float32 rounding.
    stems[STEM_NAMES.index(model_set.residual)] = mix - picked_sum
    return stems


def _picked(model: str, audio: np.ndarray, run: RunModel,
            stems: tuple[str, ...]) -> list[np.ndarray]:
    """Only the named stems of one run; the rest of a long track's result is dropped at once."""
    result = run(model, audio)
    return [_picked_stem(result, stem, model, audio.shape) for stem in stems]


def _spatial(model_set: SpatialSet, mix: np.ndarray, run: RunModel) -> np.ndarray:
    drums, bass = _picked(model_set.rhythm, mix, run, ("drums", "bass"))
    [kick] = _picked(model_set.drumsep, drums, run, (model_set.kick_stem,))
    [vocals] = _picked(model_set.vocals, mix, run, ("vocals",))
    stems = np.empty((len(model_set.stem_names),) + mix.shape, dtype=np.float32)
    low, tops, music, voice = (model_set.stem_names.index(name)
                               for name in ("low", "tops", "music", "vocals"))
    # float64 for every sum, so the only error left is the final float32 rounding.
    stems[low] = bass.astype(np.float64) + kick
    stems[tops] = drums.astype(np.float64) - kick
    stems[voice] = vocals
    rest = drums.astype(np.float64)
    rest += bass
    rest += vocals
    stems[music] = mix - rest
    return stems


def _lower_case_stems(run: RunModel) -> RunModel:
    """Model configs name their stems as they please ("Vocals" in becruily's deux)."""
    def run_lower(model: str, audio: np.ndarray) -> dict[str, np.ndarray]:
        return {stem.lower(): signal for stem, signal in run(model, audio).items()}
    return run_lower


def separate_with_set(model_set: ModelSet, mix: np.ndarray, run: RunModel) -> np.ndarray:
    """mix: float32 (channels, frames). Returns float32 (4, channels, frames) in the set's
    stem_names order."""
    run = _lower_case_stems(run)
    if isinstance(model_set, SpatialSet):
        return _spatial(model_set, mix, run)
    if isinstance(model_set, SpecialistSet):
        return _specialist(model_set, mix, run)
    return _averaged(model_set, mix, run)
