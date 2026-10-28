# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""One track from source file to a committed stem set in the library."""

import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from stemseparator import cost
from stemseparator.audio_io import (
    AUDIO_SUFFIXES, REFUSED_AUDIO_SUFFIXES, Track, UndecodableError, read_seconds, read_track,
    unsupported_format_message, write_stem_24bit)
from stemseparator.layout import TrackPlan, is_inside, plan_track
from stemseparator.separator import Separation, Separator, SeparatorError
from stemseparator.staging import StagedSet, commit

MAX_SECONDS = 1200.0
SUM_WARN_DB = -20.0  # placeholder; S0 measures the real ensemble residual, Task 12 sets it
ATTEMPTS = 3
OUTPUT_SAMPLE_RATE = 44100
# Resampling may round the length a little; a larger gap means the stems are not this track.
FRAME_TOLERANCE_MIN = 64
FRAME_TOLERANCE_RATIO = 0.001
ABANDONED_REASON = "abandoned after a second Ctrl-C"


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


class _Abandoned(Exception):
    pass


def _is_cancelled(cancel: threading.Event | None) -> bool:
    return cancel is not None and cancel.is_set()


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
                names: tuple[str, str, str],
                cancel: threading.Event | None) -> tuple[TrackPlan, int]:
    artist, album, title = names
    with StagedSet(library) as staged:
        clipped = _write_stems(staged, separation.stems)
        shutil.copy2(source, staged.original_path)
        # Checked as late as possible: writing takes seconds, the rename does not.
        if _is_cancelled(cancel):
            raise _Abandoned
        plan = commit(staged, lambda: plan_track(library, artist, album, title, source))
    return plan, clipped


def refusal_before_decode(source: Path, library: Path, allow_long: bool) -> str | None:
    """Reasons to skip a track that cost no decode and no GPU time; None if there is none."""
    if is_inside(source, library):
        return "is inside the library"
    try:
        seconds = read_seconds(source)
    except UndecodableError as error:
        return str(error)
    if seconds > MAX_SECONDS and not allow_long:
        return "longer than 20 min (use --long)"
    return None


def _result_problem(track: Track, separation: Separation) -> str | None:
    if separation.sample_rate != OUTPUT_SAMPLE_RATE:
        return (f"separator returned {separation.sample_rate} Hz, "
                f"expected {OUTPUT_SAMPLE_RATE} Hz")
    _, frames, channels = separation.stems.shape
    if channels != 2:
        return f"separator returned {channels} channel(s), expected 2"
    expected = round(len(track.audio) * OUTPUT_SAMPLE_RATE / track.sample_rate)
    tolerance = max(FRAME_TOLERANCE_MIN, expected * FRAME_TOLERANCE_RATIO)
    if abs(frames - expected) > tolerance:
        return (f"separator returned the wrong length ({frames} frames, "
                f"expected about {expected})")
    return None


def process_track(source: Path, *, library: Path, separator: Separator, models: tuple[str, ...],
                  naming: Naming, allow_long: bool,
                  sleep: Callable[[float], None] = time.sleep,
                  cancel: threading.Event | None = None) -> TrackResult:
    """`cancel` set means the user abandoned the batch: nothing more may land in the library."""
    source = Path(source)
    refusal = refusal_before_decode(source, library, allow_long)
    if refusal is not None:
        return _failed(source, refusal)
    try:
        track = read_track(source)
    except UndecodableError as error:
        return _failed(source, str(error))
    if _is_cancelled(cancel):
        return _failed(source, ABANDONED_REASON)
    try:
        separation = _separate_with_retries(track, separator, models, sleep)
    except SeparatorError as error:
        return _failed(source, str(error))
    problem = _result_problem(track, separation)
    if problem is not None:
        return _failed(source, problem)
    try:
        plan, clipped = _commit_set(source, library, separation, resolve_names(track, naming),
                                    cancel)
    except _Abandoned:
        return _failed(source, ABANDONED_REASON)
    except Exception as error:  # the staged set is already gone; the batch continues
        return _failed(source, f"could not write the set: {error}")
    return TrackResult(source=source, ok=True, plan=plan, gpu_seconds=separation.gpu_seconds,
                       gpu=separation.gpu, residual_db=separation.residual_db, clipped=clipped)


def _process_track_safely(source: Path, **track_kwargs) -> TrackResult:
    """A bug in one track must not cost the others their GPU time; Ctrl-C still stops."""
    try:
        return process_track(source, **track_kwargs)
    except Exception as error:
        return _failed(Path(source), f"internal error: {error!r}")


def _suffix_name(path: Path) -> str:
    return path.suffix.lstrip(".").upper()


def _refusal_for_file(path: Path) -> str:
    if path.suffix.lower() in REFUSED_AUDIO_SUFFIXES:
        return unsupported_format_message(_suffix_name(path))
    return "not an audio file"


def _scan_folder(folder: Path) -> tuple[list[Path], list[TrackResult]]:
    accepted, refused = [], []
    for child in sorted(folder.iterdir()):
        if child.name.startswith(".") or not child.is_file():
            continue
        suffix = child.suffix.lower()
        if suffix in AUDIO_SUFFIXES:
            accepted.append(child)
        elif suffix in REFUSED_AUDIO_SUFFIXES:
            refused.append(_failed(child, unsupported_format_message(_suffix_name(child))))
    return accepted, refused


def collect_inputs(paths: list[Path]) -> tuple[list[Path], list[TrackResult]]:
    """Expand folders (direct children only) and report what cannot be processed."""
    files: list[Path] = []
    refused: list[TrackResult] = []
    for path in map(Path, paths):
        if path.is_dir():
            found, not_accepted = _scan_folder(path)
            files += found
            refused += not_accepted
        elif not path.exists():
            refused.append(_failed(path, "not found"))
        elif path.suffix.lower() in AUDIO_SUFFIXES:
            files.append(path)
        else:
            refused.append(_failed(path, _refusal_for_file(path)))
    return files, refused


@dataclass(frozen=True)
class BatchSummary:
    results: tuple[TrackResult, ...]
    interrupted: bool = False
    unfinished: int = 0  # not started or abandoned after Ctrl-C

    @property
    def done(self) -> int:
        return sum(1 for result in self.results if result.ok)

    @property
    def failed(self) -> int:
        return len(self.results) - self.done

    @property
    def gpu_seconds(self) -> float:
        return sum(result.gpu_seconds for result in self.results)

    @property
    def cost_usd(self) -> float | None:
        """None if any track that used the GPU ran on a GPU without a known price."""
        costs = [cost.cost_usd(result.gpu_seconds, result.gpu)
                 for result in self.results if result.gpu_seconds > 0]
        if any(value is None for value in costs):
            return None
        return sum(costs)


class _BatchProgress:
    """Futures in input order; each result is reported once, on the calling thread."""

    def __init__(self, total: int, report: Callable[[TrackResult], None]):
        self._total = total
        self._report = report
        self._index_of: dict[Future, int] = {}
        self._results: list[TrackResult | None] = []

    def add(self, future: Future) -> None:
        self._index_of[future] = len(self._results)
        self._results.append(None)

    def _unreported(self) -> set[Future]:
        return {future for future, index in self._index_of.items()
                if self._results[index] is None and not future.cancelled()}

    def running_count(self) -> int:
        return sum(1 for future in self._unreported() if not future.done())

    def collect(self) -> None:
        waiting = self._unreported()
        while waiting:
            finished, waiting = wait(waiting, return_when=FIRST_COMPLETED)
            for future in finished:
                self._record(future)

    def _record(self, future: Future) -> None:
        result = future.result()
        # Stored before reporting, so an interrupt inside report never reports it twice.
        self._results[self._index_of[future]] = result
        self._report(result)

    def summary(self, interrupted: bool = False) -> BatchSummary:
        results = tuple(result for result in self._results if result is not None)
        return BatchSummary(results, interrupted, self._total - len(results))


def _ignore_count(count: int) -> None:
    pass


def _stop_after_interrupt(executor: ThreadPoolExecutor, progress: _BatchProgress,
                          cancel: threading.Event, on_stopping: Callable[[int], None],
                          on_abandoning: Callable[[int], None]) -> BatchSummary:
    executor.shutdown(wait=False, cancel_futures=True)
    try:
        on_stopping(progress.running_count())
        progress.collect()  # their GPU time is paid; keep what they deliver
    except KeyboardInterrupt:
        cancel.set()
        on_abandoning(progress.running_count())
    return progress.summary(interrupted=True)


def run_batch(sources: list[Path], *, parallel: int, report: Callable[[TrackResult], None],
              cancel: threading.Event | None = None,
              on_stopping: Callable[[int], None] = _ignore_count,
              on_abandoning: Callable[[int], None] = _ignore_count,
              **track_kwargs) -> BatchSummary:
    """Threads, not processes: the work is waiting on the network.

    A first Ctrl-C drops queued tracks and waits for the running ones (on_stopping gets their
    count); a second one sets `cancel`, so those write nothing, and returns without them.
    """
    cancel = cancel if cancel is not None else threading.Event()
    progress = _BatchProgress(len(sources), report)
    executor = ThreadPoolExecutor(max_workers=parallel)
    try:
        for source in sources:
            progress.add(executor.submit(_process_track_safely, source, cancel=cancel,
                                         **track_kwargs))
        progress.collect()
    except KeyboardInterrupt:
        return _stop_after_interrupt(executor, progress, cancel, on_stopping, on_abandoning)
    except BaseException:
        cancel.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    return progress.summary()
