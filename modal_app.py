# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Modal app: one GPU function that separates a track, one CPU function that proves the deploy.

Deploy with `modal deploy modal_app.py`. Only the registry is imported at module level: every
container imports this file, and the ping image carries neither numpy nor torch.
"""
import modal

from stemseparator.remote.registry import (
    MODEL_SAMPLE_RATE, MODEL_SETS, MODELS, MSST_COMMIT, MSST_DIR, MSST_REPO, WEIGHTS_DIR)

app = modal.App("stemseparator")
weights = modal.Volume.from_name("stemseparator-weights", create_if_missing=True)

# torch 2.11.0: the newest torch that matches the last torchaudio release (2.11.0), which
# demucs 4.0.0 pulls in; CUDA 13.0 wheels for Python 3.11, within Modal's CUDA 13.3 driver.
# librosa: MSST's models/bs_roformer package imports it. PoPE-pytorch (+ einx, torch-einops-utils,
# approved 2026-10-29): BS PolarFormer's use_pope; torch-einops-utils 0.1.37, not the 3-day-old 0.2.0. build-essential: demucs needs diffq,
# which has no Python 3.11 wheel and compiles a Cython extension.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.11.0", "torchaudio==2.11.0", "numpy", "soundfile", "scipy",
                    "ml_collections", "omegaconf==2.2.3", "pyyaml", "einops==0.8.1",
                    "beartype==0.14.1", "rotary_embedding_torch==0.3.5", "tqdm", "librosa",
                    "demucs==4.0.0", "PoPE-pytorch==0.2.1", "einx[torch]==0.4.3",
                    "torch-einops-utils==0.1.37")
    .run_commands(f"git clone {MSST_REPO} {MSST_DIR}",
                  f"git -C {MSST_DIR} checkout {MSST_COMMIT}")
    .add_local_python_source("stemseparator")
)

ping_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("stemseparator")


def _model_set(name: str):
    if name not in MODEL_SETS:
        raise ValueError(f"unknown model set '{name}' (known: {', '.join(MODEL_SETS)})")
    return MODEL_SETS[name]


def _commit_new_weights(downloaded: bool) -> None:
    if not downloaded:
        return
    try:
        weights.commit()
    except Exception as error:
        # The files are already usable in this container; a failed commit (two cold starts
        # committing at once) only costs a download next time, not this track.
        print(f"weights volume commit failed, continuing: {error!r}", flush=True)


# memory=16384 fits every set at the 20-min cap: the largest holds the mix, one 6-stem result
# (BS-RoFormer SW, ~2.5 GB), a float64 sum of the picked stems and the output set.
@app.function(image=image, gpu=["A10", "L4"], volumes={WEIGHTS_DIR: weights}, timeout=1800,
              max_containers=4, scaledown_window=60, memory=16384)
def separate(upload: bytes, model_set: str) -> dict:
    import time

    import numpy as np
    import torch

    from stemseparator import sumcheck, transport
    from stemseparator.remote import assemble, runner

    started = time.monotonic()
    chosen = _model_set(model_set)
    audio, sample_rate = transport.decode_upload(upload)
    mix = runner.resample(audio.T, sample_rate, MODEL_SAMPLE_RATE)
    del audio
    downloaded = [runner.download_missing(MODELS[name]) for name in chosen.models]
    _commit_new_weights(any(downloaded))
    device = torch.device("cuda")
    separated = assemble.separate_with_set(
        chosen, mix, lambda name, audio: runner.run_model(MODELS[name], audio, device))
    stems = np.ascontiguousarray(np.transpose(separated, (0, 2, 1)))
    del separated
    residual = sumcheck.residual_db(mix.T, stems)
    return transport.encode_result(stems, time.monotonic() - started,
                                   torch.cuda.get_device_name(), MODEL_SAMPLE_RATE, residual,
                                   chosen.stem_names)


@app.function(image=ping_image)
def ping() -> str:
    return "ok"
