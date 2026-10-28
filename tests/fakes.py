# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import threading

import numpy as np

from stemseparator import sumcheck
from stemseparator.separator import Separation, SeparatorError


def test_signal(seconds: float = 1.0, rate: int = 44100, channels: int = 2) -> np.ndarray:
    """440 Hz sine at 0.5 plus white noise at 0.05, float32 (frames, channels)."""
    rng = np.random.default_rng(0)
    frames = int(seconds * rate)
    sine = 0.5 * np.sin(2 * np.pi * 440 * np.arange(frames) / rate)
    noise = 0.05 * rng.standard_normal((frames, channels))
    return (sine[:, None] + noise).astype(np.float32)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    target_frames = round(audio.shape[0] * target_rate / source_rate)
    positions = np.arange(target_frames) * source_rate / target_rate
    source_positions = np.arange(audio.shape[0])
    channels = [np.interp(positions, source_positions, audio[:, c]) for c in range(audio.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


class FakeSeparator:
    """Weighted copies of the input, so the stems sum to the mix exactly."""

    def __init__(self, weights=(0.4, 0.3, 0.2, 0.1), fail_times=0, error=None,
                 unavailable=None, gpu_seconds=10.0):
        self.weights = weights
        self.fail_times = fail_times
        self.error = error if error is not None else SeparatorError("boom", transient=True)
        self.unavailable = unavailable
        self.gpu_seconds = gpu_seconds
        self.calls: list[tuple[int, tuple[str, ...]]] = []
        self._lock = threading.Lock()

    def preflight(self) -> None:
        if self.unavailable is not None:
            raise self.unavailable

    def _register_call(self, frames: int, models: tuple[str, ...]) -> bool:
        with self._lock:
            self.calls.append((frames, models))
            return len(self.calls) <= self.fail_times

    def separate(self, audio: np.ndarray, sample_rate: int, models: tuple[str, ...]) -> Separation:
        if self._register_call(audio.shape[0], models):
            raise self.error
        mix = audio if sample_rate == 44100 else _resample_linear(audio, sample_rate, 44100)
        stems = np.stack([w * mix for w in self.weights]).astype(np.float32)
        return Separation(stems, self.gpu_seconds, "FAKE", 44100, sumcheck.residual_db(mix, stems))
