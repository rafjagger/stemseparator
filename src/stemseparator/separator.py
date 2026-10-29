# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The interface between the pipeline and whatever separates a track."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stemseparator.naming import STEM_NAMES


@dataclass(frozen=True)
class Separation:
    stems: np.ndarray  # float32 (4, frames, channels), in stem_names order, at sample_rate
    gpu_seconds: float
    gpu: str  # e.g. "NVIDIA A10"; "" if unknown
    sample_rate: int
    residual_db: float  # mix the remote separated vs sum of stems, before any clamp
    stem_names: tuple[str, ...] = STEM_NAMES  # the model set's names, in bus order


class SeparatorError(Exception):
    def __init__(self, message: str, transient: bool):
        super().__init__(message)
        self.transient = transient


class SeparatorUnavailable(Exception):
    """Preflight failed; the message is shown verbatim."""


class Separator(Protocol):
    def preflight(self) -> None: ...

    def separate(
        self, audio: np.ndarray, sample_rate: int, model_set: str
    ) -> Separation: ...
