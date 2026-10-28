# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test-only line-by-line port of stemdeck/Source/StemNames.cpp:splitStemName.

It lets the tests check that our file names are read back by StemDeck as one
set. Never used by the CLI. If it disagrees with a title, fix sanitise_name.
"""

SEPARATORS = " -_"
NPOS = -1


def _find_last_of(text: str, characters: str) -> int:
    """std::string::find_last_of: index of the last character in the set, or NPOS."""
    return max((text.rfind(c) for c in characters), default=NPOS)


def _find_first_not_of(text: str, characters: str, start: int = 0) -> int:
    for index in range(start, len(text)):
        if text[index] not in characters:
            return index
    return NPOS


def _trim_end(text: str, characters: str) -> str:
    end = max((i for i, c in enumerate(text) if c not in characters), default=NPOS)
    return "" if end == NPOS else text[: end + 1]


def _is_number(text: str) -> bool:
    return text != "" and len(text) <= 2 and all(c in "0123456789" for c in text)


def split_stem_name(base: str) -> tuple[str, str] | None:
    split_at = _find_last_of(base, SEPARATORS)
    if split_at == NPOS or split_at == 0 or split_at == len(base) - 1:
        return None

    prefix = _trim_end(base[:split_at], SEPARATORS)
    suffix = base[split_at + 1:]
    if prefix == "":
        return None

    def dash_between(start: int, end: int) -> bool:
        return "-" in base[start:end]

    if not _is_number(suffix) and dash_between(len(prefix), split_at + 1):
        number_at = _find_last_of(prefix, SEPARATORS)
        if number_at != NPOS and _is_number(prefix[number_at + 1:]):
            title = _trim_end(prefix[:number_at], SEPARATORS)
            if title != "" and dash_between(len(title), number_at + 1):
                suffix = base[_find_first_not_of(base, SEPARATORS, len(title)):]
                prefix = title
    return prefix, suffix
