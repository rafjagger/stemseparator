# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Modal app: one GPU function that separates a track, one CPU function that proves the deploy.

Deploy with `modal deploy modal_app.py`. Only the registry is imported at module level: every
container imports this file, and the ping image carries neither numpy nor torch.
"""
import modal

from stemseparator.remote.registry import (
    MODEL_SAMPLE_RATE, MODELS, MSST_COMMIT, MSST_DIR, MSST_REPO, WEIGHTS_DIR)

app = modal.App("stemseparator")
weights = modal.Volume.from_name("stemseparator-weights", create_if_missing=True)

# torch 2.11.0: the newest torch that matches the last torchaudio release (2.11.0), which
# demucs 4.0.0 pulls in; CUDA 13.0 wheels for Python 3.11, within Modal's CUDA 13.3 driver.
# librosa: MSST's models/bs_roformer package imports it. build-essential: demucs needs diffq,
# which has no Python 3.11 wheel and compiles a Cython extension.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.11.0", "torchaudio==2.11.0", "numpy", "soundfile", "scipy",
                    "ml_collections", "omegaconf==2.2.3", "pyyaml", "einops==0.8.1",
                    "beartype==0.14.1", "rotary_embedding_torch==0.3.5", "tqdm", "librosa",
                    "demucs==4.0.0")
    .run_commands(f"git clone {MSST_REPO} {MSST_DIR}",
                  f"git -C {MSST_DIR} checkout {MSST_COMMIT}")
    .add_local_python_source("stemseparator")
)

ping_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("stemseparator")


def _model_specs(models: list[str]) -> list:
    unknown = [name for name in models if name not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")
    if not models:
        raise ValueError("no models requested")
    return [MODELS[name] for name in models]


@app.function(image=image, gpu=["A10", "L4"], volumes={WEIGHTS_DIR: weights}, timeout=1800,
              max_containers=4, scaledown_window=60)
def separate(upload: bytes, models: list[str]) -> dict:
    import time

    import numpy as np
    import torch

    from stemseparator import sumcheck, transport
    from stemseparator.remote import ensemble, runner

    started = time.monotonic()
    specs = _model_specs(models)
    audio, sample_rate = transport.decode_upload(upload)
    mix = runner.resample(audio.T, sample_rate, MODEL_SAMPLE_RATE)
    for spec in specs:
        runner.ensure_weights(spec)
    weights.commit()
    device = torch.device("cuda")
    results = [runner.run_model(spec, mix, device) for spec in specs]
    stems = np.ascontiguousarray(np.transpose(ensemble.average_stems(results), (0, 2, 1)))
    residual = sumcheck.residual_db(mix.T, stems)
    return transport.encode_result(stems, time.monotonic() - started,
                                   torch.cuda.get_device_name(), MODEL_SAMPLE_RATE, residual)


@app.function(image=ping_image)
def ping() -> str:
    return "ok"
