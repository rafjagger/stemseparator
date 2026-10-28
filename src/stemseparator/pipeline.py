# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""One track from source file to a committed stem set in the library."""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from stemseparator.audio_io import Track, UndecodableError, read_track, write_stem_24bit
from stemseparator.layout import TrackPlan, is_inside, plan_track
from stemseparator.separator import Separation, Separator, SeparatorError
from stemseparator.staging import StagedSet, commit

MAX_SECONDS = 1200.0
ATTEMPTS = 3
OUTPUT_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class Naming:
    artist: str | None = None
    album: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class TrackResult:
    source: Path
    ok: bool
    reason: str = ""
    plan: TrackPlan | None = None
    gpu_seconds: float = 0.0
    gpu: str = ""
    residual_db: float = float("-inf")
    clipped: int = 0


def resolve_names(track: Track, naming: Naming) -> tuple[str, str, str]:
    """Raw names, flag > tag > fallback; plan_track sanitises them."""
    tags = track.tags
    artist = naming.artist or tags.albumartist or tags.artist or "Unknown Artist"
    album = naming.album or tags.album or "Unknown Album"
    title = naming.title or tags.title or track.path.stem
    return artist, album, title


def _failed(source: Path, reason: str) -> TrackResult:
    return TrackResult(source=source, ok=False, reason=reason)


def _separate_with_retries(track: Track, separator: Separator, models: tuple[str, ...],
                           sleep: Callable[[float], None]) -> Separation:
    for attempt in range(ATTEMPTS):
        try:
            return separator.separate(track.audio, track.sample_rate, models)
        except SeparatorError as error:
            is_last = attempt == ATTEMPTS - 1
            if not error.transient or is_last:
                raise
            sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _write_stems(staged: StagedSet, stems: np.ndarray) -> int:
    return sum(write_stem_24bit(path, stem, OUTPUT_SAMPLE_RATE)
               for path, stem in zip(staged.stem_paths, stems))


def _commit_set(source: Path, library: Path, separation: Separation,
                names: tuple[str, str, str]) -> tuple[TrackPlan, int]:
    artist, album, title = names
    with StagedSet(library) as staged:
        clipped = _write_stems(staged, separation.stems)
        shutil.copy2(source, staged.original_path)
        plan = commit(staged, lambda: plan_track(library, artist, album, title, source))
    return plan, clipped


def process_track(source: Path, *, library: Path, separator: Separator, models: tuple[str, ...],
                  naming: Naming, allow_long: bool,
                  sleep: Callable[[float], None] = time.sleep) -> TrackResult:
    source = Path(source)
    if is_inside(source, library):
        return _failed(source, "is inside the library")
    try:
        track = read_track(source)
    except UndecodableError as error:
        return _failed(source, str(error))
    if track.seconds > MAX_SECONDS and not allow_long:
        return _failed(source, "longer than 20 min (use --long)")
    try:
        separation = _separate_with_retries(track, separator, models, sleep)
    except SeparatorError as error:
        return _failed(source, str(error))
    if separation.sample_rate != OUTPUT_SAMPLE_RATE:
        return _failed(source, f"separator returned {separation.sample_rate} Hz, "
                               f"expected {OUTPUT_SAMPLE_RATE} Hz")
    try:
        plan, clipped = _commit_set(source, library, separation, resolve_names(track, naming))
    except Exception as error:  # the staged set is already gone; the batch continues
        return _failed(source, f"could not write the set: {error}")
    return TrackResult(source=source, ok=True, plan=plan, gpu_seconds=separation.gpu_seconds,
                       gpu=separation.gpu, residual_db=separation.residual_db, clipped=clipped)
