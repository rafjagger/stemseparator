# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Taking away a set a redo has replaced in another place (the old Artist/Album layout).

Only what the stemseparator wrote is removed, and only what the manifest names: the four stems
(in either format) and the copy of the original. Folders go only when they are empty.
"""

from pathlib import Path

from stemseparator.layout import ORIGINALS_DIR
from stemseparator.manifest import Entry
from stemseparator.naming import OUTPUT_FORMATS, stem_file_name


def location_of(entry: Entry) -> Path:
    return Path(entry.output_folder) / entry.title


def _files_of(library: Path, entry: Entry, stem_names: tuple[str, ...]) -> list[Path]:
    folder = library / entry.output_folder
    stems = [folder / stem_file_name(entry.title, index, stem_names, extension)
             for index in range(len(stem_names)) for extension in OUTPUT_FORMATS]
    originals = folder / ORIGINALS_DIR
    copies = [path for path in originals.iterdir() if path.stem == entry.title] \
        if originals.is_dir() else []
    return [path for path in (*stems, *copies) if path.is_file()]


def _remove_empty_folders(library: Path, folder: Path) -> None:
    library = library.resolve()
    current = folder.resolve()
    while current != library and current.is_relative_to(library):
        try:
            current.rmdir()  # fails unless empty, which is the rule
        except OSError:
            return
        current = current.parent


def remove_replaced_set(library: Path, entry: Entry, stem_names: tuple[str, ...]) -> None:
    for path in _files_of(library, entry, stem_names):
        path.unlink()
    folder = library / entry.output_folder
    _remove_empty_folders(library, folder / ORIGINALS_DIR)
    _remove_empty_folders(library, folder)
