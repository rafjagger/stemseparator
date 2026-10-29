# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a track's stem set and original go inside the library."""

from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Callable

from stemseparator.naming import DEFAULT_OUTPUT_FORMAT, STEM_NAMES, sanitise_name, stem_file_name

ORIGINALS_DIR = "originals"


@dataclass(frozen=True)
class TrackPlan:
    album_dir: Path
    title: str
    stem_paths: tuple[Path, Path, Path, Path]
    original_path: Path


def _target_paths(album_dir: Path, title: str, suffix: str, stem_names: tuple[str, ...],
                  extension: str) -> tuple[tuple[Path, ...], Path]:
    stems = tuple(album_dir / stem_file_name(title, i, stem_names, extension)
                  for i in range(len(stem_names)))
    return stems, album_dir / ORIGINALS_DIR / f"{title}{suffix}"


def plan_track(library: Path, folder: Path, title: str, source: Path,
               exists: Callable[[Path], bool] = Path.exists,
               stem_names: tuple[str, ...] = STEM_NAMES,
               extension: str = DEFAULT_OUTPUT_FORMAT) -> TrackPlan:
    # `folder` is relative to the library: the source folder's name, plus the path below the
    # folder that was given on the command line.
    album_dir = library.joinpath(*(sanitise_name(part, "_") for part in folder.parts))
    base_title = sanitise_name(title, "Untitled")
    for attempt in count(1):
        candidate = base_title if attempt == 1 else f"{base_title} ({attempt})"
        stems, original = _target_paths(album_dir, candidate, source.suffix, stem_names,
                                       extension)
        # StemDeck only checks stem 1, but an orphaned original must not be overwritten either.
        if not any(exists(path) for path in (*stems, original)):
            return TrackPlan(album_dir, candidate, stems, original)
    raise AssertionError("unreachable")


def default_folder(source: Path) -> Path:
    """A file given on its own goes into a folder named like its parent folder."""
    return Path(source.resolve().parent.name or "stems")


def output_folders(given: list[Path], sources: list[Path]) -> dict[Path, Path]:
    """Per source, its folder in the library: mirrors the path below the folder it was found in."""
    folders = {}
    roots = [Path(path).resolve() for path in given if Path(path).is_dir()]
    for source in sources:
        inside = [root for root in roots if source.resolve().is_relative_to(root)]
        if not inside:
            folders[source] = default_folder(source)
            continue
        root = inside[0]
        folders[source] = Path(root.name) / source.resolve().parent.relative_to(root)
    return folders


def is_inside(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())
