# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""How far the four stems are from adding up to the mix they came from."""
import numpy as np


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def residual_db(mix: np.ndarray, stems: np.ndarray) -> float:
    """20*log10(rms(mix - sum(stems)) / rms(mix)); -inf when exact or the mix is silent."""
    mix_rms = _rms(mix)
    if mix_rms == 0.0:
        return -np.inf
    residual_rms = _rms(mix.astype(np.float64) - stems.astype(np.float64).sum(axis=0))
    if residual_rms == 0.0:
        return -np.inf
    return float(20.0 * np.log10(residual_rms / mix_rms))
