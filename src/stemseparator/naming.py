# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""File names StemDeck recognises as one stem set (mirrors stemdeck/Source/StemNames.cpp)."""

# Demucs' own order and NI Stems': bus 1 is drums.
STEM_NAMES: tuple[str, ...] = ("drums", "bass", "other", "vocals")
# Four stems by spatial job, the space4 set: kick + bass, the rest of the kit, the music, the voice.
SPACE4_NAMES: tuple[str, ...] = ("low", "tops", "music", "vocals")

# Stems are written as 24-bit FLAC or WAV; StemDeck reads both (FLAC decoded into RAM on load).
OUTPUT_FORMATS: tuple[str, ...] = ("flac", "wav")
DEFAULT_OUTPUT_FORMAT = "flac"

_TRAILING_SEPARATORS = " -_"
_REPLACED_CHARACTERS = "/\\\0"


def sanitise_name(name: str, fallback: str) -> str:
    clean = "".join("_" if c in _REPLACED_CHARACTERS else c for c in name)
    # Leading dots would hide the file; trailing separators would be cut off
    # by StemDeck's splitStemName and change the set's prefix.
    clean = clean.lstrip(". ").rstrip(_TRAILING_SEPARATORS)
    return fallback if clean.strip("_ ") == "" else clean


def stem_file_name(title: str, stem: int, names: tuple[str, ...] = STEM_NAMES,
                   extension: str = "wav") -> str:
    return f"{title} - {stem + 1} - {names[stem]}.{extension}"
