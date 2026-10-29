# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""One track through several model sets, each set into its own folder, for comparing them.

`<library>/shootout/<set>/<Title> - N - <stem>.wav`, plus runs.json with each set's GPU time,
cost and residual. A set folder is written next to its final place and renamed in, so a failed
or interrupted set leaves its previous folder (or nothing) behind.
"""
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stemseparator import cost
from stemseparator.audio_io import Track, UndecodableError, lower_to_avoid_clipping, read_seconds, \
    read_track, write_stem_24bit
from stemseparator.compare import RUNS_FILE
from stemseparator.layout import is_inside
from stemseparator.naming import DEFAULT_OUTPUT_FORMAT, sanitise_name, stem_file_name
from stemseparator.pipeline import (
    MAX_SECONDS, OUTPUT_SAMPLE_RATE, Naming, result_problem, title_for, separate_with_retries)
from stemseparator.separator import Separation, Separator, SeparatorError

SHOOTOUT_DIR = "shootout"
_PARTIAL_PREFIX = ".partial-"


class ShootoutError(Exception):
    """The shootout cannot start; nothing was separated."""


@dataclass(frozen=True)
class SetRun:
    set_name: str
    ok: bool
    reason: str = ""
    folder: Path | None = None
    gpu_seconds: float = 0.0
    gpu: str = ""
    residual_db: float = float("-inf")
    gain: float = 1.0
    stem_names: tuple[str, ...] = ()


def shootout_dir(library: Path) -> Path:
    return library / SHOOTOUT_DIR


def _load_runs(folder: Path) -> dict:
    path = folder / RUNS_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ShootoutError(f"cannot read {path}: {error}") from error


def _check_source(source: Path, folder: Path, allow_long: bool) -> None:
    if is_inside(source, folder):
        raise ShootoutError(f"{source} is inside the shootout folder {folder}")
    try:
        seconds = read_seconds(source)
    except UndecodableError as error:
        raise ShootoutError(f"{source}: {error}") from error
    if seconds > MAX_SECONDS and not allow_long:
        raise ShootoutError(f"{source}: longer than 20 min (use --long)")


def _check_same_track(runs: dict, source: Path, folder: Path) -> None:
    earlier = runs.get("track")
    if earlier is None or Path(earlier) == source.resolve():
        return
    raise ShootoutError(f"{folder} holds a shootout of {earlier}; move it away or use another "
                        "--library")


def _write_set(folder: Path, set_name: str, title: str, separation: Separation,
               output_format: str) -> tuple[Path, float]:
    partial = folder / f"{_PARTIAL_PREFIX}{set_name}"
    target = folder / set_name
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        lowered, gain = lower_to_avoid_clipping(separation.stems)
        for index, stem in enumerate(lowered):
            write_stem_24bit(
                partial / stem_file_name(title, index, separation.stem_names, output_format),
                stem, OUTPUT_SAMPLE_RATE)
        shutil.rmtree(target, ignore_errors=True)
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target, gain


def _run_set(set_name: str, track: Track, title: str, folder: Path, separator: Separator,
             sleep: Callable[[float], None], output_format: str) -> SetRun:
    """Returns no audio: a finished future would otherwise hold every set's stems."""
    try:
        separation = separate_with_retries(track, separator, set_name, sleep)
    except SeparatorError as error:
        return SetRun(set_name, ok=False, reason=str(error))
    problem = result_problem(track, separation)
    if problem is not None:
        return SetRun(set_name, ok=False, reason=problem, gpu_seconds=separation.gpu_seconds,
                      gpu=separation.gpu)
    try:
        target, gain = _write_set(folder, set_name, title, separation, output_format)
    except OSError as error:
        return SetRun(set_name, ok=False, reason=f"could not write the set: {error}",
                      gpu_seconds=separation.gpu_seconds, gpu=separation.gpu)
    return SetRun(set_name, ok=True, folder=target, gpu_seconds=separation.gpu_seconds,
                  gpu=separation.gpu, residual_db=separation.residual_db, gain=gain,
                  stem_names=separation.stem_names)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _entry(run: SetRun) -> dict:
    return {"gpu_seconds": run.gpu_seconds, "gpu": run.gpu,
            "cost_usd": cost.cost_usd(run.gpu_seconds, run.gpu),
            "residual_db": _finite_or_none(run.residual_db), "gain": run.gain,
            "stem_names": list(run.stem_names)}


def _write_runs(folder: Path, source: Path, title: str, sets: dict, order: list[str]) -> None:
    # Requested sets in the order asked for; sets from earlier runs keep their place in front.
    ordered = {name: sets[name] for name in sets if name not in order}
    ordered.update({name: sets[name] for name in order if name in sets})
    runs = {"track": str(source.resolve()), "title": title, "sets": ordered}
    partial = folder / f"{RUNS_FILE}.partial"
    partial.write_text(json.dumps(runs, indent=2) + "\n")
    partial.replace(folder / RUNS_FILE)


def run_shootout(source: Path, set_names: list[str], *, library: Path, separator: Separator,
                 naming: Naming, parallel: int, allow_long: bool,
                 sleep: Callable[[float], None] = time.sleep,
                 report: Callable[[SetRun], None] = lambda run: None,
                 output_format: str = DEFAULT_OUTPUT_FORMAT) -> list[SetRun]:
    """Separates `source` with every set, up to `parallel` at once; returns the runs in set order."""
    source = Path(source)
    folder = shootout_dir(library)
    _check_source(source, folder, allow_long)
    runs = _load_runs(folder)
    _check_same_track(runs, source, folder)
    try:
        track = read_track(source)
    except UndecodableError as error:
        raise ShootoutError(f"{source}: {error}") from error
    title = sanitise_name(title_for(source, naming), "Untitled")
    folder.mkdir(parents=True, exist_ok=True)
    entries = dict(runs.get("sets", {}))
    finished: dict[str, SetRun] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(set_names)))) as executor:
        futures = [executor.submit(_run_set, name, track, title, folder, separator, sleep,
                                   output_format)
                   for name in set_names]
        for future in as_completed(futures):
            run = future.result()
            finished[run.set_name] = run
            if run.ok:
                entries[run.set_name] = _entry(run)
                _write_runs(folder, source, title, entries, set_names)
            report(run)
    return [finished[name] for name in set_names]
