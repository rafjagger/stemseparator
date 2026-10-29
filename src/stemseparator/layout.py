# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a track's stem set and original go inside the library."""

from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Callable

from stemseparator.naming import STEM_NAMES, sanitise_name, stem_file_name

ORIGINALS_DIR = "originals"


@dataclass(frozen=True)
class TrackPlan:
    album_dir: Path
    title: str
    stem_paths: tuple[Path, Path, Path, Path]
    original_path: Path


def _target_paths(album_dir: Path, title: str, suffix: str,
                  stem_names: tuple[str, ...]) -> tuple[tuple[Path, ...], Path]:
    stems = tuple(album_dir / stem_file_name(title, i, stem_names) for i in range(len(stem_names)))
    return stems, album_dir / ORIGINALS_DIR / f"{title}{suffix}"


def plan_track(library: Path, artist: str, album: str, title: str, source: Path,
               exists: Callable[[Path], bool] = Path.exists,
               stem_names: tuple[str, ...] = STEM_NAMES) -> TrackPlan:
    album_dir = (library / sanitise_name(artist, "Unknown Artist")
                 / sanitise_name(album, "Unknown Album"))
    base_title = sanitise_name(title, "Untitled")
    for attempt in count(1):
        candidate = base_title if attempt == 1 else f"{base_title} ({attempt})"
        stems, original = _target_paths(album_dir, candidate, source.suffix, stem_names)
        # StemDeck only checks stem 1, but an orphaned original must not be overwritten either.
        if not any(exists(path) for path in (*stems, original)):
            return TrackPlan(album_dir, candidate, stems, original)
    raise AssertionError("unreachable")


def is_inside(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())
