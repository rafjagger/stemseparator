# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The separation models and model sets the remote function knows, as pure data (no torch, no modal).

Published numbers are SDR in dB as MSST's docs/pretrained_models.md (at MSST_COMMIT) or the
release notes state them; "Multisong" is MVSep's 100-track test set, "MUSDB" is MUSDB18-HQ test.
"""
from dataclasses import dataclass

from stemseparator.naming import SPACE4_NAMES, STEM_NAMES

MSST_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
MSST_COMMIT = "6fdad13686bb7e84d029ec66f9dbe76e65c93ebf"
MSST_DIR = "/msst"
WEIGHTS_DIR = "/weights"
MODEL_SAMPLE_RATE = 44100

_MSST_RELEASES = f"{MSST_REPO}/releases/download"
_MSST_RAW = f"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{MSST_COMMIT}"
_DEMUCS_RELEASES = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer"
# Revision-pinned, so the file behind the URL cannot change under the sha256.
_KJ_REVISION = "ac9b0614ab3cd7f77219e18ba494dfd93956c348"
_KJ_REPO = "https://huggingface.co/KimberleyJSN/melbandroformer"

_MSST_WEIGHTS_LICENCE = ("no licence stated for the weights (MSST code: MIT); trained on "
                         "MUSDB18-HQ, whose licence is research/non-commercial")


@dataclass(frozen=True)
class Checkpoint:
    url: str
    # None until S0 measured it (the runner then logs the hash it saw). A value is the digest the
    # host publishes (GitHub release asset digest, Hugging Face LFS oid); the runner checks it.
    sha256: str | None
    stem: str | None = None  # set: keep only this stem of the model's output (htdemucs_ft)


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    config: str  # an https URL, or a path relative to MSST_DIR
    checkpoints: tuple[Checkpoint, ...]
    licence: str  # as published next to the weights
    reported: str  # published per-stem numbers and where they come from


MODELS: dict[str, ModelSpec] = {
    "scnet_xl_ihf": ModelSpec(
        model_type="scnet",
        config=f"{_MSST_RELEASES}/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml",
        checkpoints=(Checkpoint(
            url=f"{_MSST_RELEASES}/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt",
            sha256="ac25975f0f5704f3d1a3c3c251505b7a0f417a22eafe82773440ee4f7e14b74f"),),
        licence=_MSST_WEIGHTS_LICENCE,
        reported=("SCNet XL IHF, MSST pretrained_models.md + release v1.0.15: Multisong bass 11.94, "
                  "drums 11.58, vocals 9.69, other 6.49 (avg 9.93); MUSDB avg 10.09"),
    ),
    "scnet_masked_xl_ihf": ModelSpec(
        model_type="scnet_masked",
        config=f"{_MSST_RELEASES}/v1.0.17/config_musdb18_scnet_xl_ihf.yaml",
        checkpoints=(Checkpoint(
            url=f"{_MSST_RELEASES}/v1.0.17/model_scnet_masked_ep_111_sdr_9.8286.ckpt",
            sha256="b4675b0269809de27172a050e8767a857077635eda1738db0874d63a79f2b6dd"),),
        licence=_MSST_WEIGHTS_LICENCE,
        reported=("SCNet Masked XL IHF, MSST pretrained_models.md: Multisong bass 11.57, "
                  "drums 11.38, vocals 9.43, other 6.30 (avg 9.67); MUSDB avg 9.82"),
    ),
    "bs_roformer_ep17": ModelSpec(
        model_type="bs_roformer",
        config=f"{_MSST_RELEASES}/v1.0.12/config_bs_roformer_384_8_2_485100.yaml",
        checkpoints=(Checkpoint(
            url=f"{_MSST_RELEASES}/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
            sha256=None),),
        licence=_MSST_WEIGHTS_LICENCE,
        reported=("BS-RoFormer (MUSDB18-HQ only), MSST pretrained_models.md: Multisong bass "
                  "11.08, drums 11.29, vocals 9.19, other 5.96 (avg 9.38); MUSDB avg 9.65"),
    ),
    "mel_roformer_kj_vocals": ModelSpec(
        model_type="mel_band_roformer",
        config=f"{_MSST_RAW}/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        checkpoints=(Checkpoint(
            url=f"{_KJ_REPO}/resolve/{_KJ_REVISION}/MelBandRoformer.ckpt",
            sha256="87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"),),
        licence=("MIT, as stated on the Hugging Face model card (KimberleyJSN/melbandroformer); "
                 "community weights, training data not stated"),
        reported=("Mel-Band RoFormer vocals (Kimberley Jensen), MSST pretrained_models.md: "
                  "Multisong vocals 10.98"),
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
        licence="Demucs repository: MIT; no separate licence stated for the weights",
        reported=("HTDemucs4 FT per-stem checkpoints, MSST pretrained_models.md: Multisong "
                  "drums 11.13, bass 11.96, other 5.85, vocals 8.38"),
    ),
}


@dataclass(frozen=True)
class AveragedSet:
    """Every model separates all four stems; the stems are averaged (MSST ensemble.py avg_wave)."""
    models: tuple[str, ...]
    stem_names = STEM_NAMES


@dataclass(frozen=True)
class SpecialistSet:
    """Each stem but one comes from the model best at it; the remaining stem is the mix minus them,
    so the four stems sum back to the mix."""
    picks: tuple[tuple[str, str], ...]  # (stem, model)
    residual: str
    stem_names = STEM_NAMES

    def __post_init__(self):
        stems = [stem for stem, _ in self.picks] + [self.residual]
        if sorted(stems) != sorted(STEM_NAMES):
            raise ValueError(f"a specialist set must cover {', '.join(STEM_NAMES)} exactly once, "
                             f"got {', '.join(stems)}")

    @property
    def models(self) -> tuple[str, ...]:
        """Each model once, in first-use order."""
        return tuple(dict.fromkeys(model for _, model in self.picks))


@dataclass(frozen=True)
class SpatialSet:
    """Four stems by spatial job (SPACE4_NAMES): LOW = bass + kick, TOPS = drums - kick,
    VOCALS from the vocal model, MUSIC = the mix minus the other three.

    The drum model splits the rhythm model's drums stem; only its kick is used, so the drums stem
    survives whole as LOW + TOPS and the four stems sum back to the mix.
    """
    rhythm: str  # a 4-stem model: its drums and bass are used
    vocals: str  # its vocals stem is used
    drumsep: str  # run on the drums stem
    kick_stem: str  # the drum model's name for the kick
    stem_names = SPACE4_NAMES

    @property
    def models(self) -> tuple[str, ...]:
        """In run order; each once."""
        return tuple(dict.fromkeys((self.rhythm, self.drumsep, self.vocals)))


ModelSet = AveragedSet | SpecialistSet | SpatialSet

MODEL_SETS: dict[str, ModelSet] = {
    # Drums and bass from the best published open 4-stem model; vocals from the best published
    # open vocal model the image can load as it is. BS PolarFormer reports 11.00 against 10.98,
    # but needs the extra package PoPE_pytorch in the image, which is not approved.
    "specialist": SpecialistSet(
        picks=(("drums", "scnet_xl_ihf"), ("bass", "scnet_xl_ihf"),
               ("vocals", "mel_roformer_kj_vocals")),
        residual="other"),
    "ensemble4": AveragedSet(("scnet_masked_xl_ihf", "bs_roformer_ep17")),
    "htdemucs_ft": AveragedSet(("htdemucs_ft",)),
}

DEFAULT_MODEL_SET = "specialist"
