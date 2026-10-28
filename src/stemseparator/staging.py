# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a track's files outside the library and rename them in all at once."""

import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Callable

from stemseparator.layout import ORIGINALS_DIR, TrackPlan
from stemseparator.naming import STEM_NAMES

STAGING_DIR_NAME = ".stemseparator-staging"

# One lock for the whole process: planning a free name and renaming into it must be one step.
_commit_lock = threading.Lock()
# Creating a job folder and removing an empty staging root must not interleave.
_root_lock = threading.Lock()


class StagingError(Exception):
    pass


def staging_root(library: Path) -> Path:
    return library.parent / STAGING_DIR_NAME


class StagedSet:
    def __init__(self, library: Path):
        library.mkdir(parents=True, exist_ok=True)
        self._root = staging_root(library)
        self._job = self._root / str(uuid.uuid4())
        with _root_lock:
            self._job.mkdir(parents=True)
        try:
            self._require_same_filesystem(library)
        except BaseException:
            self._remove()
            raise
        self.stem_paths = tuple(self._job / f"{n}.wav" for n in range(1, len(STEM_NAMES) + 1))
        self.original_path = self._job / "original"

    def _require_same_filesystem(self, library: Path) -> None:
        if self._job.stat().st_dev == library.stat().st_dev:
            return
        raise StagingError(
            f"{library} is on a different filesystem than its staging folder; "
            "an atomic rename is not possible")

    def __enter__(self) -> "StagedSet":
        return self

    def __exit__(self, *exc_info) -> None:
        self._remove()

    def _remove(self) -> None:
        shutil.rmtree(self._job, ignore_errors=True)
        with _root_lock:
            try:
                self._root.rmdir()  # fails while another job is still in there, which is intended
            except OSError:
                pass


def _rename_all(moves: list[tuple[Path, Path]]) -> None:
    done: list[tuple[Path, Path]] = []
    try:
        for source, target in moves:
            os.rename(source, target)
            done.append((source, target))
    except BaseException:
        # Best effort: a failing rollback step must neither stop the others nor mask the cause.
        for source, target in reversed(done):
            try:
                os.rename(target, source)
            except OSError:
                pass
        raise


def commit(staged: StagedSet, plan_for: Callable[[], TrackPlan]) -> TrackPlan:
    with _commit_lock:
        plan = plan_for()
        (plan.album_dir / ORIGINALS_DIR).mkdir(parents=True, exist_ok=True)
        # Stems last: StemDeck only shows a group of exactly four.
        moves = [(staged.original_path, plan.original_path),
                 *zip(staged.stem_paths, plan.stem_paths)]
        _rename_all(moves)
        return plan
