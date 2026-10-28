# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU price table and cost figures."""

GPU_PRICES_PER_SECOND = {"A10": 0.000306, "L4": 0.000222}  # modal.com/pricing, 2026-10-28
GPU_SECONDS_PER_AUDIO_SECOND = 0.3  # placeholder until S0 measures the real figure


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


def estimate_usd(audio_seconds: float, models: tuple[str, ...]) -> float:
    gpu_seconds = audio_seconds * GPU_SECONDS_PER_AUDIO_SECOND * len(models)
    return gpu_seconds * GPU_PRICES_PER_SECOND["A10"]
