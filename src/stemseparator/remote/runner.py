# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weights download and one MSST model run, inside the Modal image only.

Needs torch, scipy and the MSST checkout at MSST_DIR; nothing outside modal_app.py imports it.
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import uuid
from math import gcd
from pathlib import Path

import numpy as np

from stemseparator.remote.registry import MSST_DIR, WEIGHTS_DIR, Checkpoint, ModelSpec

_CHUNK_BYTES = 1 << 20


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Polyphase resampling along the last axis (frames); float32 out."""
    if from_rate == to_rate:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    divisor = gcd(from_rate, to_rate)
    resampled = resample_poly(audio, to_rate // divisor, from_rate // divisor, axis=-1)
    return resampled.astype(np.float32)


def _weights_path(url: str) -> Path:
    return Path(WEIGHTS_DIR) / url.rsplit("/", 1)[-1]


def _download(url: str, target: Path, expected_sha256: str | None) -> None:
    # A unique temporary name: parallel containers may fetch the same file on a cold start.
    partial = target.with_name(f"{target.name}.part-{uuid.uuid4().hex}")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            while chunk := response.read(_CHUNK_BYTES):
                digest.update(chunk)
                handle.write(chunk)
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise RuntimeError(f"sha256 mismatch for {url}: got {actual}, "
                               f"expected {expected_sha256}")
        if expected_sha256 is None:
            print(f"sha256 {actual} {url} (not pinned yet)", flush=True)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _ensure_file(url: str, expected_sha256: str | None) -> Path:
    target = _weights_path(url)
    if not target.exists():
        Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
        _download(url, target, expected_sha256)
    return target


def config_path(spec: ModelSpec) -> Path:
    if "://" in spec.config:
        return _weights_path(spec.config)
    return Path(MSST_DIR) / spec.config


def ensure_weights(spec: ModelSpec) -> list[Path]:
    """Checkpoint paths in spec order; downloads what is missing. The caller commits the Volume."""
    if "://" in spec.config:
        # Configs are tiny and change with the checkpoint, so they are not hash-pinned.
        _ensure_file(spec.config, None)
    return [_ensure_file(checkpoint.url, checkpoint.sha256) for checkpoint in spec.checkpoints]


def _msst():
    if MSST_DIR not in sys.path:
        sys.path.insert(0, MSST_DIR)
    from utils.model_utils import demix, load_start_checkpoint
    from utils.settings import get_model_from_config
    return get_model_from_config, demix, load_start_checkpoint


def _checkpoint_args(path: Path, model_type: str) -> argparse.Namespace:
    return argparse.Namespace(start_check_point=str(path), model_type=model_type,
                              load_only_compatible_weights=False, lora_checkpoint_loralib=None)


def _run_checkpoint(spec: ModelSpec, checkpoint: Checkpoint, path: Path,
                    mix: np.ndarray, device) -> dict[str, np.ndarray]:
    import torch
    get_model_from_config, demix, load_start_checkpoint = _msst()
    model, config = get_model_from_config(spec.model_type, str(config_path(spec)))
    # MSST's own call: these checkpoints are pickles, which is why the registry pins sha256.
    state = torch.load(path, weights_only=False, map_location="cpu")
    load_start_checkpoint(_checkpoint_args(path, spec.model_type), model, state,
                          type_="inference")
    model = model.to(device).eval()
    with torch.inference_mode():
        separated = demix(config, model, mix, device, model_type=spec.model_type)
    del model, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if checkpoint.stem is None:
        return dict(separated)
    return {checkpoint.stem: separated[checkpoint.stem]}


def run_model(spec: ModelSpec, mix_44k: np.ndarray, device) -> dict[str, np.ndarray]:
    """mix_44k: float32 (channels, frames) at MODEL_SAMPLE_RATE; returns stem -> (channels, frames)."""
    paths = ensure_weights(spec)
    stems: dict[str, np.ndarray] = {}
    for checkpoint, path in zip(spec.checkpoints, paths):
        stems.update(_run_checkpoint(spec, checkpoint, path, mix_44k, device))
    return stems
