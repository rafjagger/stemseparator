# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np


def test_signal(seconds: float = 1.0, rate: int = 44100, channels: int = 2) -> np.ndarray:
    """440 Hz sine at 0.5 plus white noise at 0.05, float32 (frames, channels)."""
    rng = np.random.default_rng(0)
    frames = int(seconds * rate)
    sine = 0.5 * np.sin(2 * np.pi * 440 * np.arange(frames) / rate)
    noise = 0.05 * rng.standard_normal((frames, channels))
    return (sine[:, None] + noise).astype(np.float32)
