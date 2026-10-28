# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""How far the four stems are from adding up to the mix they came from."""
import numpy as np

# Frames per block when squaring: a float64 copy of a whole 20-minute track would be large.
_CHUNK_FRAMES = 1 << 18


def _rms(signal: np.ndarray) -> float:
    total = 0.0
    for start in range(0, signal.shape[0], _CHUNK_FRAMES):
        block = signal[start:start + _CHUNK_FRAMES].astype(np.float64, copy=False)
        total += float(np.vdot(block, block))
    return float(np.sqrt(total / signal.size)) if signal.size else 0.0


def residual_db(mix: np.ndarray, stems: np.ndarray) -> float:
    """20*log10(rms(mix - sum(stems)) / rms(mix)); -inf when exact or the mix is silent."""
    mix_rms = _rms(mix)
    if mix_rms == 0.0:
        return -np.inf
    # The only full-size float64 array: the stem sum, turned into the residual in place.
    residual = stems.sum(axis=0, dtype=np.float64)
    np.subtract(mix, residual, out=residual)
    residual_rms = _rms(residual)
    if residual_rms == 0.0:
        return -np.inf
    return float(20.0 * np.log10(residual_rms / mix_rms))
