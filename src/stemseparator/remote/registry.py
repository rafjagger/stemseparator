# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The separation models the remote function knows, as pure data (no torch, no modal)."""
from dataclasses import dataclass

MSST_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
MSST_COMMIT = "6fdad13686bb7e84d029ec66f9dbe76e65c93ebf"
MSST_DIR = "/msst"
WEIGHTS_DIR = "/weights"
MODEL_SAMPLE_RATE = 44100

_MSST_RELEASES = f"{MSST_REPO}/releases/download"
_DEMUCS_RELEASES = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer"


@dataclass(frozen=True)
class Checkpoint:
    url: str
    sha256: str | None  # None until S0 measured it; the runner then logs the hash it saw
    stem: str | None = None  # set: keep only this stem of the model's output (htdemucs_ft)


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    config: str  # an https URL, or a path relative to MSST_DIR
    checkpoints: tuple[Checkpoint, ...]


MODELS: dict[str, ModelSpec] = {
    "scnet_xl_ihf": ModelSpec(
        model_type="scnet_masked",
        config=f"{_MSST_RELEASES}/v1.0.17/config_musdb18_scnet_xl_ihf.yaml",
        checkpoints=(Checkpoint(
            url=f"{_MSST_RELEASES}/v1.0.17/model_scnet_masked_ep_111_sdr_9.8286.ckpt",
            sha256=None),),
    ),
    "bs_roformer_ep17": ModelSpec(
        model_type="bs_roformer",
        config=f"{_MSST_RELEASES}/v1.0.12/config_bs_roformer_384_8_2_485100.yaml",
        checkpoints=(Checkpoint(
            url=f"{_MSST_RELEASES}/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
            sha256=None),),
    ),
    "htdemucs_ft": ModelSpec(
        model_type="htdemucs",
        config="configs/config_musdb18_htdemucs.yaml",
        checkpoints=(
            Checkpoint(url=f"{_DEMUCS_RELEASES}/f7e0c4bc-ba3fe64a.th", sha256=None, stem="drums"),
            Checkpoint(url=f"{_DEMUCS_RELEASES}/d12395a8-e57c48e6.th", sha256=None, stem="bass"),
            Checkpoint(url=f"{_DEMUCS_RELEASES}/92cfc3b6-ef3bcb9c.th", sha256=None, stem="other"),
            Checkpoint(url=f"{_DEMUCS_RELEASES}/04573f0d-f3cf25b2.th", sha256=None, stem="vocals"),
        ),
    ),
}

DEFAULT_MODELS: tuple[str, ...] = ("scnet_xl_ihf", "bs_roformer_ep17")
