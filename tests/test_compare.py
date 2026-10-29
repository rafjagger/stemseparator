# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from stemseparator import compare
from stemseparator.naming import SPACE4_NAMES, STEM_NAMES, stem_file_name

RATE = 44100


def _tone(hz: float, seconds: float, level: float = 0.25) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    mono = level * np.sin(2 * np.pi * hz * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _write(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, RATE, subtype="FLOAT")
    return path


class Fixture:
    """A 30 s mix of four tones and two stem sets of it; the second puts half the bass in other."""

    def __init__(self, root: Path):
        self.root = root
        self.parts = {"drums": _tone(8000, 30), "bass": _tone(45, 30),
                      "other": _tone(600, 30), "vocals": _tone(1500, 30)}
        self.mix_audio = sum(self.parts.values())
        self.mix = _write(root / "mix.wav", self.mix_audio)

    def write_set(self, name: str, stems: dict[str, np.ndarray], names=STEM_NAMES) -> Path:
        folder = self.root / "shootout" / name
        for index, stem in enumerate(names):
            _write(folder / stem_file_name("Song", index, names), stems[stem])
        return folder

    def exact(self, name="exact") -> Path:
        return self.write_set(name, self.parts)

    def leaky(self, name="leaky") -> Path:
        stems = dict(self.parts)
        stems["bass"] = 0.5 * self.parts["bass"]
        stems["other"] = self.parts["other"] + 0.5 * self.parts["bass"]
        return self.write_set(name, stems)


class FindSetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = Fixture(Path(self._tmp.name))

    def test_finds_the_four_stems_in_bus_order_and_skips_originals(self):
        folder = self.fixture.exact()
        _write(folder / "originals" / "Song - 1 - drums.wav", _tone(100, 1))
        found = compare.find_set(folder)
        self.assertEqual(list(found), list(STEM_NAMES))
        self.assertEqual(found["bass"].name, "Song - 2 - bass.wav")

    def test_space4_names_are_found_too(self):
        stems = dict(zip(SPACE4_NAMES, self.fixture.parts.values()))
        folder = self.fixture.write_set("space4", stems, SPACE4_NAMES)
        self.assertEqual(list(compare.find_set(folder)), list(SPACE4_NAMES))

    def test_a_folder_without_four_stems_raises(self):
        folder = self.fixture.exact()
        (folder / "Song - 4 - vocals.wav").unlink()
        with self.assertRaises(compare.CompareError):
            compare.find_set(folder)


class MeasureTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = Fixture(Path(self._tmp.name))
        self.mix = compare.read_mono(self.fixture.mix)

    def test_a_set_that_sums_to_the_mix_has_no_residual(self):
        measures = compare.measure_set(self.mix, compare.find_set(self.fixture.exact()))
        self.assertLess(measures.residual_db, -100.0)

    def test_a_missing_part_shows_in_the_residual(self):
        stems = dict(self.fixture.parts)
        stems["vocals"] = np.zeros_like(stems["vocals"])
        measures = compare.measure_set(self.mix, compare.find_set(
            self.fixture.write_set("novocals", stems)))
        # One of four equal tones missing: rms ratio 1/2, -6 dB.
        self.assertAlmostEqual(measures.residual_db, -6.02, delta=0.2)

    def test_level_and_band_shares_show_where_a_stem_sits(self):
        measures = compare.measure_set(self.mix, compare.find_set(self.fixture.exact()))
        bass = measures.stems["bass"]
        self.assertAlmostEqual(bass.level_db, -6.02, delta=0.2)
        self.assertGreater(bass.band_share_db["sub <60"], -0.5)
        self.assertLess(bass.band_share_db["air >6k"], -60.0)

    def test_bass_leaking_into_other_shows_as_sub_band_bleed(self):
        measures = compare.measure_set(self.mix, compare.find_set(self.fixture.leaky()))
        self.assertAlmostEqual(measures.stems["other"].band_share_db["sub <60"], -6.02, delta=0.3)


class PairTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = Fixture(Path(self._tmp.name))

    def test_same_stems_correlate_fully(self):
        pairs = compare.compare_sets(compare.find_set(self.fixture.exact()),
                                     compare.find_set(self.fixture.exact("again")))
        self.assertAlmostEqual(pairs["vocals"].correlation, 1.0, places=4)
        self.assertLess(pairs["vocals"].difference_db, -100.0)

    def test_music_is_compared_with_other(self):
        space4 = dict(zip(SPACE4_NAMES, self.fixture.parts.values()))
        pairs = compare.compare_sets(
            compare.find_set(self.fixture.exact()),
            compare.find_set(self.fixture.write_set("space4", space4, SPACE4_NAMES)))
        self.assertEqual(sorted(pairs), ["other", "vocals"])

    def test_a_changed_stem_shows_its_difference(self):
        pairs = compare.compare_sets(compare.find_set(self.fixture.exact()),
                                     compare.find_set(self.fixture.leaky()))
        self.assertAlmostEqual(pairs["bass"].difference_db, -6.02, delta=0.2)


class ExcerptTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_the_loudest_window_is_found(self):
        mono = np.concatenate([np.full(20 * RATE, 0.01), np.full(12 * RATE, 0.5),
                               np.full(20 * RATE, 0.01)]).astype(np.float32)
        self.assertAlmostEqual(compare.loudest_window(mono, RATE, 12.0) / RATE, 20.0, delta=1.0)

    def test_a_short_track_starts_at_zero(self):
        self.assertEqual(compare.loudest_window(np.ones(5 * RATE, np.float32), RATE, 12.0), 0)

    def test_each_stem_gets_one_file_with_every_set_back_to_back(self):
        fixture = Fixture(self.root)
        sets = {"exact": compare.find_set(fixture.exact()),
                "leaky": compare.find_set(fixture.leaky())}
        written = compare.write_excerpts(sets, start_seconds=5.0, out_dir=self.root / "excerpts")
        self.assertEqual(sorted(p.name for p in written),
                         ["1 - drums.wav", "2 - bass.wav", "3 - other.wav", "4 - vocals.wav"])
        audio, rate = sf.read(self.root / "excerpts" / "2 - bass.wav", dtype="float32")
        segment, gap = 12 * RATE, RATE
        self.assertEqual(rate, RATE)
        self.assertEqual(audio.shape, (2 * segment + gap, 2))
        start = 5 * RATE
        np.testing.assert_allclose(audio[:segment], fixture.parts["bass"][start:start + segment],
                                   atol=1e-6)
        self.assertEqual(float(np.abs(audio[segment:segment + gap]).max()), 0.0)
        np.testing.assert_allclose(audio[segment + gap:],
                                   0.5 * fixture.parts["bass"][start:start + segment], atol=1e-6)

    def test_music_and_other_share_one_excerpt_and_space4_stems_get_their_own(self):
        fixture = Fixture(self.root)
        space4 = dict(zip(SPACE4_NAMES, fixture.parts.values()))
        sets = {"exact": compare.find_set(fixture.exact()),
                "space4": compare.find_set(fixture.write_set("space4", space4, SPACE4_NAMES))}
        written = compare.write_excerpts(sets, start_seconds=0.0, out_dir=self.root / "x")
        self.assertIn("3 - other+music.wav", [p.name for p in written])
        self.assertIn("1 - low.wav", [p.name for p in written])
        low, _ = sf.read(self.root / "x" / "1 - low.wav")
        self.assertEqual(len(low), 12 * RATE)


class RunCompareTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = Fixture(Path(self._tmp.name))
        self.shootout = self.fixture.root / "shootout"

    def test_writes_a_report_and_excerpts_using_the_runs_file(self):
        self.fixture.exact("specialist")
        self.fixture.leaky("space4x")
        (self.shootout / compare.RUNS_FILE).write_text(json.dumps({
            "track": str(self.fixture.mix), "title": "Song",
            "sets": {"specialist": {"gpu_seconds": 61.0, "gpu": "NVIDIA A10", "cost_usd": 0.0187,
                                    "residual_db": None, "clipped": 0},
                     "space4x": {"gpu_seconds": 95.0, "gpu": "NVIDIA A10", "cost_usd": 0.029,
                                 "residual_db": -140.2, "clipped": 3}}}))
        out = io.StringIO()
        report = compare.run_compare(self.shootout, out=out)
        text = report.read_text()
        for expected in ("specialist", "space4x", "61", "0.03", "residual", "sub <60",
                         "correlation", "excerpts"):
            self.assertIn(expected, text)
        self.assertTrue((self.shootout / "excerpts" / "2 - bass.wav").is_file())
        self.assertIn(str(report), out.getvalue())

    def test_without_a_runs_file_the_mix_must_be_given(self):
        self.fixture.exact("a")
        with self.assertRaises(compare.CompareError):
            compare.run_compare(self.shootout, out=io.StringIO())
        report = compare.run_compare(self.shootout, mix=self.fixture.mix, out=io.StringIO())
        self.assertIn("| a |", report.read_text())

    def test_a_mix_at_another_rate_is_refused_with_a_reason(self):
        self.fixture.exact("a")
        other = self.fixture.root / "mix48.wav"
        sf.write(other, np.zeros((48000, 2), np.float32), 48000)
        with self.assertRaises(compare.CompareError) as caught:
            compare.run_compare(self.shootout, mix=other, out=io.StringIO())
        self.assertIn("48000", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
