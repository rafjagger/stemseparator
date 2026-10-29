# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU price table and cost figures."""
from stemseparator.remote.registry import MODEL_SETS, MODELS

GPU_PRICES_PER_SECOND = {"A10": 0.000306, "L4": 0.000222}  # modal.com/pricing, 2026-10-28
# Per checkpoint run; a placeholder until S0 measures the real figure. Both 2-model sets keep it:
# the specialist swaps BS-RoFormer (dim 384, depth 8, 4 stems) for a Mel-Band RoFormer that is no
# larger (dim 384, depth 6, 1 stem), and SCNet XL IHF for its unmasked twin of the same size.
GPU_SECONDS_PER_AUDIO_SECOND = 0.3


def _price_per_second(gpu: str) -> float | None:
    matches = [name for name in GPU_PRICES_PER_SECOND if name in gpu]
    if not matches:
        return None
    # Longest key wins so a future "A100" never matches as "A10".
    return GPU_PRICES_PER_SECOND[max(matches, key=len)]


def cost_usd(gpu_seconds: float, gpu: str) -> float | None:
    price = _price_per_second(gpu)
    if price is None:
        return None
    return gpu_seconds * price


def checkpoint_runs(model_set: str) -> int:
    return sum(len(MODELS[model].checkpoints) for model in MODEL_SETS[model_set].models)


def estimate_usd(audio_seconds: float, model_set: str) -> float:
    gpu_seconds = audio_seconds * GPU_SECONDS_PER_AUDIO_SECOND * checkpoint_runs(model_set)
    return gpu_seconds * GPU_PRICES_PER_SECOND["A10"]
