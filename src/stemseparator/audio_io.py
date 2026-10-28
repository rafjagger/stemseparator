# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read tracks (FLAC, WAV, AIFF) and write clamped 24-bit stems."""

from dataclasses import dataclass
from pathlib import Path

import mutagen
import numpy as np
import soundfile as sf

AUDIO_SUFFIXES = (".flac", ".wav", ".aif", ".aiff")
ACCEPTED_HEADER_FORMATS = ("WAV", "WAVEX", "RF64", "FLAC", "AIFF")
# Largest value PCM_24 can hold; +1.0 would wrap to negative full scale.
MAX_24BIT = 1.0 - 2.0**-23


@dataclass(frozen=True)
class Tags:
    title: str = ""
    artist: str = ""
    album: str = ""
    albumartist: str = ""


@dataclass(frozen=True)
class Track:
    path: Path
    audio: np.ndarray  # float32, shape (frames, 2)
    sample_rate: int
    tags: Tags

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sample_rate


class UndecodableError(Exception):
    """The message is the reason shown to the user."""


def unsupported_format_message(format_name: str) -> str:
    return f"{format_name} is not supported: use FLAC, WAV or AIFF"


def _check_suffix(path: Path) -> None:
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise UndecodableError(unsupported_format_message(path.suffix.lstrip(".").upper()))


_ACCEPTED_MAGIC = (b"fLaC", b"RIFF", b"RF64", b"FORM")


def _reject_unaccepted_magic(path: Path) -> None:
    # Checked before sf.info: libsndfile probes unknown bytes as MP3 and
    # prints decoder notes to stderr.
    try:
        with open(path, "rb") as file:
            head = file.read(4)
    except OSError as error:
        raise UndecodableError(f"cannot be read: {error}") from error
    if head in _ACCEPTED_MAGIC:
        return
    if head == b"OggS":
        raise UndecodableError(unsupported_format_message("OGG"))
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        raise UndecodableError(unsupported_format_message("MP3"))
    raise UndecodableError("cannot be decoded: unrecognised audio file")


def _read_header(path: Path):
    _reject_unaccepted_magic(path)
    try:
        return sf.info(path)
    except (sf.LibsndfileError, RuntimeError) as error:
        raise UndecodableError(f"cannot be decoded: {error}") from error


def _check_header(info) -> None:
    if info.format not in ACCEPTED_HEADER_FORMATS:
        raise UndecodableError(unsupported_format_message(info.format))
    if info.channels not in (1, 2):
        raise UndecodableError(f"{info.channels} channels: only mono or stereo is supported")
    if info.frames == 0:
        raise UndecodableError("no audio")


def _decode(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, rate = sf.read(path, always_2d=True, dtype="float32")
    except (sf.LibsndfileError, RuntimeError) as error:
        raise UndecodableError(f"cannot be decoded: {error}") from error
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio, rate


def _first_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return _first_text(value[0]) if value else ""
    text = getattr(value, "text", None)
    if text is not None:
        return _first_text(text)
    return str(value).strip()


def _mutagen_tags(path: Path) -> Tags:
    try:
        file = mutagen.File(path)
    except Exception:  # mutagen raises many types on damaged tags
        return Tags()
    if file is None or file.tags is None:
        return Tags()
    tags = file.tags
    if hasattr(tags, "getall"):  # ID3 (WAV, AIFF)
        def frame(name):
            return _first_text(tags.getall(name))
        return Tags(frame("TIT2"), frame("TPE1"), frame("TALB"), frame("TPE2"))
    def comment(name):  # Vorbis comments (FLAC)
        return _first_text(tags.get(name))
    return Tags(comment("title"), comment("artist"), comment("album"), comment("albumartist"))


def _soundfile_tags(path: Path) -> Tags:
    try:
        with sf.SoundFile(path) as file:
            return Tags(
                title=(file.title or "").strip(),
                artist=(file.artist or "").strip(),
                album=(file.album or "").strip(),
            )
    except (sf.LibsndfileError, RuntimeError):
        return Tags()


def read_tags(path: Path) -> Tags:
    primary = _mutagen_tags(path)
    fallback = _soundfile_tags(path)
    return Tags(
        title=primary.title or fallback.title,
        artist=primary.artist or fallback.artist,
        album=primary.album or fallback.album,
        albumartist=primary.albumartist,
    )


def read_track(path: Path) -> Track:
    path = Path(path)
    _check_suffix(path)
    _check_header(_read_header(path))
    audio, rate = _decode(path)
    return Track(path=path, audio=audio, sample_rate=rate, tags=read_tags(path))


def write_stem_24bit(path: Path, audio: np.ndarray, sample_rate: int) -> int:
    """Write a 24-bit WAV; returns how many samples had to be clipped."""
    clipped = int(np.count_nonzero((audio > MAX_24BIT) | (audio < -1.0)))
    # libsndfile does not clip float to int, it wraps.
    sf.write(path, np.clip(audio, -1.0, MAX_24BIT), sample_rate, subtype="PCM_24")
    return clipped
