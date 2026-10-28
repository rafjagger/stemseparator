# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""File names StemDeck recognises as one stem set (mirrors stemdeck/Source/StemNames.cpp)."""

# Demucs' own order and NI Stems': bus 1 is drums.
STEM_NAMES: tuple[str, ...] = ("drums", "bass", "other", "vocals")

_TRAILING_SEPARATORS = " -_"
_REPLACED_CHARACTERS = "/\\\0"


def sanitise_name(name: str, fallback: str) -> str:
    clean = "".join("_" if c in _REPLACED_CHARACTERS else c for c in name)
    # Leading dots would hide the file; trailing separators would be cut off
    # by StemDeck's splitStemName and change the set's prefix.
    clean = clean.lstrip(". ").rstrip(_TRAILING_SEPARATORS)
    return fallback if clean.strip("_ ") == "" else clean


def stem_file_name(title: str, stem: int, extension: str = "wav") -> str:
    return f"{title} - {stem + 1} - {STEM_NAMES[stem]}.{extension}"
