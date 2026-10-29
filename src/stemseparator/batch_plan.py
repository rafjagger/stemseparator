# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decide, before any upload, which sources are already converted.

A source is done when the manifest knows its content (so a renamed or moved file still counts)
or when its set stands in the library under the name its tags give, all four stems present.
"""

from dataclasses import dataclass
from pathlib import Path

from stemseparator.audio_io import UndecodableError, read_seconds
from stemseparator.layout import ORIGINALS_DIR, plan_track
from stemseparator.manifest import Fingerprint, Manifest, fingerprint
from stemseparator.naming import OUTPUT_FORMATS, stem_file_name
from stemseparator.pipeline import Naming, refusal_before_decode, title_for

DONE = "done"
PENDING = "pending"
CHANGED = "changed"  # the file is not the one the library's set was made from
DUPLICATE = "duplicate"  # the same content as an earlier source of this run
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Item:
    source: Path
    state: str
    fingerprint: Fingerprint | None = None
    location: str = ""  # DONE: the set in the library; DUPLICATE: the earlier source
    reason: str = ""
    seconds: float = 0.0


def _stem_present(album_dir: Path, title: str, index: int, stem_names: tuple[str, ...]) -> bool:
    return any((album_dir / stem_file_name(title, index, stem_names, extension)).exists()
               for extension in OUTPUT_FORMATS)


def set_present(album_dir: Path, title: str, stem_names: tuple[str, ...]) -> bool:
    return all(_stem_present(album_dir, title, i, stem_names) for i in range(len(stem_names)))


def _location(library: Path, album_dir: Path, title: str) -> str:
    return str(album_dir.relative_to(library) / title)


def _manifest_hit(source_fp: Fingerprint, library: Path, model_set: str,
                  stem_names: tuple[str, ...], manifest: Manifest) -> str | None:
    for entry in manifest.with_content(source_fp, model_set):
        album_dir = library / entry.output_folder
        if set_present(album_dir, entry.title, stem_names):
            return _location(library, album_dir, entry.title)
    return None


def _original_files(album_dir: Path, title: str) -> list[Path]:
    originals = album_dir / ORIGINALS_DIR
    if not originals.is_dir():
        return []
    return [p for p in originals.iterdir() if p.stem == title]


def _name_hit(source: Path, source_fp: Fingerprint, library: Path, folder: Path, naming: Naming,
              stem_names: tuple[str, ...]) -> tuple[str | None, bool]:
    """(location of a set under the source's own name, whether that set is of another file)."""
    plan = plan_track(library, folder, title_for(source, naming), source, exists=lambda path: False,
                      stem_names=stem_names)
    if not set_present(plan.album_dir, plan.title, stem_names):
        return None, False
    location = _location(library, plan.album_dir, plan.title)
    originals = _original_files(plan.album_dir, plan.title)
    # Without an original there is nothing to compare with; the set stands as it is.
    other_file = bool(originals) and not any(
        fingerprint(path).same_content(source_fp) for path in originals)
    return location, other_file


def _was_converted_from_here(source: Path, model_set: str, manifest: Manifest) -> bool:
    return bool(manifest.with_source(source, model_set))


def classify_source(source: Path, source_fp: Fingerprint, *, library: Path, folder: Path,
                    model_set: str,
                    stem_names: tuple[str, ...], naming: Naming, manifest: Manifest,
                    allow_long: bool) -> Item:
    hit = _manifest_hit(source_fp, library, model_set, stem_names, manifest)
    if hit is not None:
        return Item(source, DONE, source_fp, location=hit)
    location, other_file = _name_hit(source, source_fp, library, folder, naming, stem_names)
    if location is not None and not other_file:
        return Item(source, DONE, source_fp, location=location)
    refusal = refusal_before_decode(source, library, allow_long)
    if refusal is not None:
        return Item(source, UNSUPPORTED, source_fp, reason=refusal)
    try:
        seconds = read_seconds(source)
    except UndecodableError as error:
        return Item(source, UNSUPPORTED, source_fp, reason=str(error))
    changed = other_file or _was_converted_from_here(source, model_set, manifest)
    return Item(source, CHANGED if changed else PENDING, source_fp, seconds=seconds)


def classify(sources: list[Path], *, library: Path, folders: dict[Path, Path], model_set: str,
             stem_names: tuple[str, ...], naming: Naming, manifest: Manifest,
             allow_long: bool) -> list[Item]:
    items: list[Item] = []
    first_of: dict[tuple[int, str], Path] = {}
    for source in sources:
        try:
            source_fp = fingerprint(source)
        except OSError as error:
            items.append(Item(source, UNSUPPORTED, reason=f"cannot be read: {error.strerror}"))
            continue
        key = (source_fp.size, source_fp.sha256)
        if key in first_of:
            items.append(Item(source, DUPLICATE, source_fp, location=str(first_of[key])))
            continue
        first_of[key] = source
        items.append(classify_source(source, source_fp, library=library, folder=folders[source],
                                     model_set=model_set,
                                     stem_names=stem_names, naming=naming, manifest=manifest,
                                     allow_long=allow_long))
    return items
