# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the library already holds: a small manifest of converted sources.

The manifest lets a renamed or moved source count as done (the identity is a hash of the first
and last megabyte plus the size), and lets an edited file count as new.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = ".stemseparator-manifest.json"
MANIFEST_VERSION = 1
HASHED_BYTES = 1024 * 1024


class ManifestError(Exception):
    pass


@dataclass(frozen=True)
class Fingerprint:
    size: int
    sha256: str  # of the first and the last HASHED_BYTES of the file

    def same_content(self, other: "Fingerprint") -> bool:
        return self.size == other.size and self.sha256 == other.sha256


def fingerprint(path: Path) -> Fingerprint:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        digest.update(file.read(HASHED_BYTES))
        if size > HASHED_BYTES:
            file.seek(max(HASHED_BYTES, size - HASHED_BYTES))
            digest.update(file.read(HASHED_BYTES))
    return Fingerprint(size, digest.hexdigest())


@dataclass(frozen=True)
class Entry:
    source: str
    size: int
    mtime: float
    sha256: str
    model_set: str
    date: str
    output_folder: str  # relative to the library
    title: str
    # Entries from before the no-clipping fix have neither: those sets were clamped if they were
    # hot, and nothing recorded whether they were.
    gain: float | None = None  # the common factor the stems were lowered by; 1.0 = unchanged
    clipped: int | None = None  # samples that were clamped when the set was written

    @property
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(self.size, self.sha256)


class Manifest:
    def __init__(self, library: Path):
        self.path = library / MANIFEST_NAME
        self._entries: list[Entry] = []
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = [Entry(**item) for item in raw["entries"]]
        except FileNotFoundError:
            return
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ManifestError(f"cannot read {self.path}: {error} "
                                "(fix or remove it; nothing was changed)") from error

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def with_content(self, fp: Fingerprint, model_set: str) -> list[Entry]:
        return [e for e in self._entries
                if e.fingerprint.same_content(fp) and e.model_set == model_set]

    def may_have_clipped(self, fp: Fingerprint, model_set: str) -> bool:
        """True unless the set's entry says it was written without clamping."""
        entries = self.with_content(fp, model_set)
        if not entries:
            return True
        latest = entries[-1]
        return latest.gain is None or (latest.clipped or 0) > 0

    def with_source(self, source: Path, model_set: str) -> list[Entry]:
        key = str(source.resolve())
        return [e for e in self._entries if e.source == key and e.model_set == model_set]

    def record(self, source: Path, fp: Fingerprint, model_set: str, output_folder: str,
               title: str, gain: float | None = None) -> None:
        entry = Entry(source=str(source.resolve()), size=fp.size, mtime=source.stat().st_mtime,
                      sha256=fp.sha256, model_set=model_set,
                      date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      output_folder=output_folder, title=title, gain=gain,
                      clipped=None if gain is None else 0)
        kept = [e for e in self._entries
                if not (e.fingerprint.same_content(fp) and e.model_set == model_set)]
        self._entries = [*kept, entry]
        self._save()

    def _save(self) -> None:
        # Replaced in one step, so an interrupted run never leaves a half-written manifest.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        document = {"version": MANIFEST_VERSION,
                    "entries": [asdict(entry) for entry in self._entries]}
        temp.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        os.replace(temp, self.path)
