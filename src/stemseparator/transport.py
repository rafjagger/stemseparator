# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wire format between the local pipeline and the remote function.

Also imported inside the remote image: numpy, soundfile, stdlib and stemseparator only.
"""
import io

import numpy as np
import soundfile as sf

from stemseparator.separator import Separation, SeparatorError

_STEM_COUNT = 4
_FULL_SCALE = 1.0


def encode_upload(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    # FLAC cannot carry values beyond full scale; a hot signal must stay unclipped.
    if float(np.max(np.abs(audio), initial=0.0)) > _FULL_SCALE:
        sf.write(buffer, audio, sample_rate, format="WAV", subtype="FLOAT")
    else:
        sf.write(buffer, audio, sample_rate, format="FLAC", subtype="PCM_24")
    return buffer.getvalue()


def decode_upload(data: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio, sample_rate


def encode_result(
    stems: np.ndarray, gpu_seconds: float, gpu: str, sample_rate: int, residual_db: float
) -> dict:
    little_endian = np.ascontiguousarray(stems, dtype="<f4")
    return {
        "stems": little_endian.tobytes(),
        "shape": list(little_endian.shape),
        "gpu_seconds": float(gpu_seconds),
        "gpu": gpu,
        "sample_rate": int(sample_rate),
        "residual_db": float(residual_db),
    }


def _malformed(reason: str) -> SeparatorError:
    return SeparatorError(f"Malformed result from the separator: {reason}", transient=False)


def _validated_shape(result: dict) -> tuple[int, ...]:
    shape = result.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise _malformed("shape is not (stems, frames, channels)")
    if not all(isinstance(n, int) and n > 0 for n in shape):
        raise _malformed("shape has non-positive entries")
    if shape[0] != _STEM_COUNT:
        raise _malformed(f"expected {_STEM_COUNT} stems, got {shape[0]}")
    return tuple(shape)


def decode_result(result: dict) -> Separation:
    shape = _validated_shape(result)
    data = result.get("stems")
    if not isinstance(data, (bytes, bytearray)):
        raise _malformed("stems are not bytes")
    if len(data) != int(np.prod(shape)) * 4:
        raise _malformed("stem byte length does not match shape")
    sample_rate = result.get("sample_rate")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise _malformed("sample_rate missing or not an integer")
    try:
        gpu_seconds = float(result["gpu_seconds"])
        residual_db = float(result["residual_db"])
        gpu = str(result["gpu"])
    except (KeyError, TypeError, ValueError):
        raise _malformed("gpu_seconds, gpu or residual_db missing or invalid") from None
    stems = np.frombuffer(data, dtype="<f4").reshape(shape).astype(np.float32)
    return Separation(stems, gpu_seconds, gpu, sample_rate, residual_db)
