# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline measures of several stem sets of one track, and listening excerpts to judge them by ear.

No reference stems exist for a real track, so nothing here is an SDR. What can be measured against
the mix alone: how far a set's sum is from the mix (residual), how loud each stem is, and how much
of each frequency band of the mix a stem carries (a vocals stem with sub-bass is bleed). Between
sets: how different the same stem is, and how closely it correlates.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import soundfile as sf

from stemseparator.cost import format_usd

RATE = 44100
RUNS_FILE = "runs.json"
REPORT_FILE = "report.md"
EXCERPTS_DIR = "excerpts"
EXCERPT_SECONDS = 12.0
GAP_SECONDS = 1.0
BLOCK = 8192
BANDS = {"sub <60": (0.0, 60.0), "low 60-250": (60.0, 250.0), "mid 250-2k": (250.0, 2000.0),
         "high 2k-6k": (2000.0, 6000.0), "air >6k": (6000.0, RATE / 2 + 1)}
# Stems that do the same job under another name, compared and played next to each other.
ALIASES = {"music": "other"}
_STEM_FILE = re.compile(r" - ([1-4]) - ([a-z]+)$")
_FLOOR = 1e-30


class CompareError(Exception):
    pass


@dataclass(frozen=True)
class StemMeasures:
    level_db: float  # rms re the mix
    band_share_db: dict[str, float]  # the stem's energy in a band re the mix's in that band


@dataclass(frozen=True)
class SetMeasures:
    residual_db: float  # mix minus the sum of the stems, re the mix
    stems: dict[str, StemMeasures]


@dataclass(frozen=True)
class PairMeasures:
    difference_db: float  # rms(a - b) re rms(a)
    correlation: float


def find_set(folder: Path) -> dict[str, Path]:
    """The four '<title> - N - name.wav' files below `folder` (originals/ skipped), in bus order."""
    found: dict[int, tuple[str, Path]] = {}
    for path in sorted(folder.rglob("*.wav")):
        if "originals" in path.relative_to(folder).parts:
            continue
        match = _STEM_FILE.search(path.stem)
        if match is None:
            continue
        number = int(match.group(1))
        if number in found:
            raise CompareError(f"{folder}: more than one stem {number}")
        found[number] = (match.group(2), path)
    if sorted(found) != [1, 2, 3, 4]:
        raise CompareError(f"{folder}: expected stems 1-4, found {sorted(found) or 'none'}")
    return {name: path for _, (name, path) in sorted(found.items())}


def role(stem: str) -> str:
    return ALIASES.get(stem, stem)


def read_mono(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE:
        raise CompareError(f"{path}: {rate} Hz; compare needs {RATE} Hz")
    return audio.mean(axis=1)


def _db(power_ratio: float) -> float:
    return float(10.0 * np.log10(max(power_ratio, _FLOOR)))


def _power(signal: np.ndarray) -> float:
    return float(np.mean(np.square(signal, dtype=np.float64))) if signal.size else 0.0


def band_energy(signal: np.ndarray) -> dict[str, float]:
    blocks = len(signal) // BLOCK
    if blocks == 0:
        return {name: 0.0 for name in BANDS}
    # Hann window: a rectangular one would smear a loud bass across every band.
    framed = signal[:blocks * BLOCK].reshape(blocks, BLOCK) * np.hanning(BLOCK)
    spectrum = np.square(np.abs(np.fft.rfft(framed, axis=1))).sum(axis=0)
    freqs = np.fft.rfftfreq(BLOCK, 1.0 / RATE)
    return {name: float(spectrum[(freqs >= low) & (freqs < high)].sum())
            for name, (low, high) in BANDS.items()}


def _fit(signal: np.ndarray, frames: int) -> np.ndarray:
    if len(signal) >= frames:
        return signal[:frames]
    return np.pad(signal, (0, frames - len(signal)))


def measure_set(mix: np.ndarray, stems: dict[str, Path]) -> SetMeasures:
    """One stem in memory at a time; the mix is mono at RATE."""
    mix_power = _power(mix)
    mix_bands = band_energy(mix)
    remainder = mix.astype(np.float64)
    measures = {}
    for name, path in stems.items():
        signal = _fit(read_mono(path), len(mix))
        remainder -= signal
        bands = band_energy(signal)
        measures[name] = StemMeasures(
            level_db=_db(_power(signal) / max(mix_power, _FLOOR)),
            band_share_db={band: _db(bands[band] / max(mix_bands[band], _FLOOR))
                           for band in BANDS})
    return SetMeasures(_db(_power(remainder) / max(mix_power, _FLOOR)), measures)


def compare_sets(reference: dict[str, Path], other: dict[str, Path]) -> dict[str, PairMeasures]:
    """Stems both sets carry for the same job, keyed by role (music counts as other)."""
    others = {role(name): path for name, path in other.items()}
    pairs = {}
    for name, path in reference.items():
        if role(name) not in others:
            continue
        a = read_mono(path).astype(np.float64)
        b = _fit(read_mono(others[role(name)]), len(a)).astype(np.float64)
        difference = _db(_power(a - b) / max(_power(a), _FLOOR))
        silent = np.std(a) == 0.0 or np.std(b) == 0.0
        correlation = 0.0 if silent else float(np.corrcoef(a, b)[0, 1])
        pairs[role(name)] = PairMeasures(difference, correlation)
    return pairs


def loudest_window(mono: np.ndarray, rate: int, seconds: float) -> int:
    """First frame of the loudest `seconds` of the track, in whole seconds; 0 if it is shorter."""
    window = int(seconds)
    blocks = len(mono) // rate
    if blocks <= window:
        return 0
    power = np.square(mono[:blocks * rate].reshape(blocks, rate), dtype=np.float64).sum(axis=1)
    sums = np.convolve(power, np.ones(window), mode="valid")
    return int(np.argmax(sums)) * rate


def _excerpt(path: Path, start: int, frames: int) -> np.ndarray:
    with sf.SoundFile(path) as handle:
        if handle.samplerate != RATE:
            raise CompareError(f"{path}: {handle.samplerate} Hz; compare needs {RATE} Hz")
        handle.seek(min(start, handle.frames))
        audio = handle.read(frames, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return np.pad(audio, ((0, frames - len(audio)), (0, 0)))


def _excerpt_groups(sets: dict[str, dict[str, Path]]) -> dict[str, list[tuple[str, int, str, Path]]]:
    """role -> [(set, bus number, stem name, path)] in set order."""
    groups: dict[str, list[tuple[str, int, str, Path]]] = {}
    for set_name, stems in sets.items():
        for number, (stem, path) in enumerate(stems.items(), start=1):
            groups.setdefault(role(stem), []).append((set_name, number, stem, path))
    return groups


def _excerpt_name(entries: list[tuple[str, int, str, Path]]) -> str:
    names = "+".join(dict.fromkeys(stem for _, _, stem, _ in entries))
    return f"{entries[0][1]} - {names}.wav"


def write_excerpts(sets: dict[str, dict[str, Path]], start_seconds: float,
                   out_dir: Path) -> list[Path]:
    """Per stem one WAV: the same EXCERPT_SECONDS of every set that has it, GAP_SECONDS apart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start, frames = int(start_seconds * RATE), int(EXCERPT_SECONDS * RATE)
    gap = np.zeros((int(GAP_SECONDS * RATE), 2), np.float32)
    written = []
    for entries in _excerpt_groups(sets).values():
        pieces = []
        for _, _, _, path in entries:
            pieces += [gap, _excerpt(path, start, frames)] if pieces else [_excerpt(path, start, frames)]
        target = out_dir / _excerpt_name(entries)
        sf.write(target, np.concatenate(pieces), RATE, subtype="PCM_24")
        written.append(target)
    return written


def _load_runs(shootout: Path) -> dict:
    path = shootout / RUNS_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise CompareError(f"cannot read {path}: {error}") from error


def _set_folders(shootout: Path, runs: dict) -> dict[str, dict[str, Path]]:
    folders = [p for p in sorted(shootout.iterdir())
               if p.is_dir() and not p.name.startswith(".") and p.name != EXCERPTS_DIR]
    order = list(runs.get("sets", {}))
    folders.sort(key=lambda p: (order.index(p.name) if p.name in order else len(order), p.name))
    if not folders:
        raise CompareError(f"{shootout}: no stem sets in it")
    return {folder.name: find_set(folder) for folder in folders}


def _number(value, digits: int) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _runs_table(sets: dict[str, dict[str, Path]], runs: dict,
                measures: dict[str, SetMeasures]) -> list[str]:
    lines = ["| set | stems | GPU s | GPU | cost | residual dB (remote) | residual dB (files) | "
             "clipped |", "|---|---|---|---|---|---|---|---|"]
    for name, stems in sets.items():
        run = runs.get("sets", {}).get(name, {})
        cost = run.get("cost_usd")
        lines.append(
            f"| {name} | {' / '.join(stems)} | {_number(run.get('gpu_seconds'), 0)} | "
            f"{run.get('gpu', '–')} | {format_usd(cost) if 'cost_usd' in run else '–'} | "
            f"{_number(run.get('residual_db'), 1) if run else '–'} | "
            f"{measures[name].residual_db:.1f} | {run.get('clipped', '–')} |")
    return lines


def _stems_table(measures: dict[str, SetMeasures]) -> list[str]:
    lines = ["| set | stem | level dB | " + " | ".join(BANDS) + " |",
             "|---|---|---|" + "---|" * len(BANDS)]
    for name, set_measures in measures.items():
        for stem, stem_measures in set_measures.stems.items():
            shares = " | ".join(f"{stem_measures.band_share_db[band]:.1f}" for band in BANDS)
            lines.append(f"| {name} | {stem} | {stem_measures.level_db:.1f} | {shares} |")
    return lines


def _pairs_table(sets: dict[str, dict[str, Path]]) -> list[str]:
    names = list(sets)
    reference = names[0]
    lines = [f"Against `{reference}`: difference dB re its stem, and correlation r.", "",
             "| set | stem | difference dB | correlation |", "|---|---|---|---|"]
    for name in names[1:]:
        for stem, pair in compare_sets(sets[reference], sets[name]).items():
            lines.append(f"| {name} | {stem} | {pair.difference_db:.1f} | {pair.correlation:.3f} |")
    return lines


def _report(title: str, mix: Path, sets, runs, measures, start: int, excerpts) -> str:
    excerpt_lines = [f"- `{EXCERPTS_DIR}/{p.name}`" for p in excerpts]
    return "\n".join([
        f"# Shootout: {title}", "", f"Mix: `{mix}`", "",
        "## Runs", "", *_runs_table(sets, runs, measures), "",
        "Residual: the mix minus the sum of the four stems; below -60 dB they add up.", "",
        "## Stems", "",
        "Level re the mix; then each band's energy in the stem re the mix's energy in that band "
        "(0 dB: the stem carries all of the band; a vocals stem near 0 dB in sub <60 is bleed).", "",
        *_stems_table(measures), "",
        "## Set against set", "", *_pairs_table(sets), "",
        "## Listening excerpts", "",
        f"{EXCERPT_SECONDS:.0f} s from {start // RATE // 60}:{start // RATE % 60:02d} of every set "
        f"that has the stem, {GAP_SECONDS:.0f} s apart, in this order: {', '.join(sets)}.", "",
        *excerpt_lines, ""])


def run_compare(shootout: Path, *, mix: Path | None = None, at_seconds: float | None = None,
                out: TextIO) -> Path:
    """Measures every set folder in `shootout`, writes report.md and excerpts/; returns the report."""
    runs = _load_runs(shootout)
    mix = mix if mix is not None else (Path(runs["track"]) if "track" in runs else None)
    if mix is None:
        raise CompareError(f"no {RUNS_FILE} in {shootout}: give the mix with --mix")
    sets = _set_folders(shootout, runs)
    mix_mono = read_mono(mix)
    measures = {name: measure_set(mix_mono, stems) for name, stems in sets.items()}
    start = loudest_window(mix_mono, RATE, EXCERPT_SECONDS) if at_seconds is None \
        else int(at_seconds * RATE)
    excerpts = write_excerpts(sets, start / RATE, shootout / EXCERPTS_DIR)
    report = shootout / REPORT_FILE
    report.write_text(_report(runs.get("title", mix.stem), mix, sets, runs, measures, start,
                              excerpts))
    print(f"report: {report}", file=out)
    print(f"excerpts: {shootout / EXCERPTS_DIR}", file=out)
    return report
