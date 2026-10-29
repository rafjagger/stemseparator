# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""A set that would exceed full scale is lowered as a whole, never clamped."""

import io
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

from stemseparator import cli
from stemseparator.audio_io import MAX_24BIT
from stemseparator.manifest import MANIFEST_NAME
from stemseparator.naming import SPACE4_NAMES
from tests import fakes
from tests.fakes import FakeSeparator

# Sums to 1, so the stems add up to the mix, but the first stem alone is 1.6 times the mix.
HOT = (1.6, -0.8, 0.1, 0.1)
QUIET = (0.4, 0.3, 0.2, 0.1)
TARGET_PEAK = 10 ** (-0.1 / 20)


def refuse_factory():
    raise AssertionError("the separator must not be built")


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.music = self.root / "music"
        self.music.mkdir()
        self.library = self.root / "lib"
        config = self.root / "config.toml"
        config.write_text(f'library = "{self.library}"\nmodels = "space4"\nparallel = 1\n'
                          'format = "wav"\n')
        self.config_path = config
        self.out, self.err = io.StringIO(), io.StringIO()

    def _track(self, name, seed=None, amplitude=0.9):
        path = self.music / name
        audio = fakes.test_signal(0.5, seed=zlib.crc32(name.encode()) if seed is None else seed)
        audio = (audio * (amplitude / np.max(np.abs(audio)))).astype(np.float32)
        sf.write(path, audio, 44100, format="FLAC", subtype="PCM_24")
        # What the program decodes, not what was meant: 24-bit rounding is part of the source.
        return path, sf.read(path, dtype="float32", always_2d=True)[0].astype(np.float64)

    def _run(self, *argv, weights=QUIET):
        self.out.seek(0), self.out.truncate(), self.err.seek(0), self.err.truncate()
        separator = FakeSeparator(weights=weights, stem_names=SPACE4_NAMES)
        code = cli.main([str(a) for a in argv], separator_factory=lambda: separator,
                        config_path=self.config_path, known_sets=None,
                        out=self.out, err=self.err)
        return code, separator

    def _stems(self):
        files = sorted(p for p in self.library.rglob("*.wav") if "originals" not in p.parts)
        return np.stack([sf.read(p, dtype="float64", always_2d=True)[0] for p in files])

    def _entries(self):
        return json.loads((self.library / MANIFEST_NAME).read_text())["entries"]


class NoClippingTest(_Base):
    def test_all_four_stems_get_the_same_factor_and_the_peak_lands_at_minus_point_one(self):
        _, mix = self._track("a.flac")
        code, _ = self._run(self.music, weights=HOT)
        self.assertEqual(code, 0, self.err.getvalue())
        stems = self._stems()
        factor = TARGET_PEAK / (1.6 * float(np.max(np.abs(mix))))
        self.assertAlmostEqual(float(np.max(np.abs(stems))), TARGET_PEAK, places=5)
        for stem, weight in zip(stems, HOT):
            np.testing.assert_allclose(stem, weight * factor * mix, atol=2e-7)

    def test_the_lowered_stems_still_add_up_to_the_mix_times_the_factor(self):
        _, mix = self._track("a.flac")
        self._run(self.music, weights=HOT)
        gain = self._entries()[0]["gain"]
        np.testing.assert_allclose(self._stems().sum(axis=0), mix * gain, atol=1e-6)

    def test_nothing_in_the_written_files_exceeds_full_scale(self):
        self._track("a.flac")
        self._run(self.music, weights=HOT)
        raw = [sf.read(p, dtype="int32", always_2d=True)[0]
               for p in self.library.rglob("*.wav") if "originals" not in p.parts]
        for samples in raw:
            self.assertLess(int(np.max(samples)), 2**31 - 256)
            self.assertGreater(int(np.min(samples)), -(2**31))
        self.assertLessEqual(float(np.max(np.abs(self._stems()))), MAX_24BIT)

    def test_the_run_names_the_reduction_instead_of_warning_about_clipping(self):
        self._track("a.flac")
        self._run(self.music, weights=HOT)
        self.assertRegex(self.out.getvalue(), r"lowered by \d+\.\d dB to avoid clipping")
        self.assertNotIn("sample(s) clipped", self.out.getvalue())

    def test_a_set_within_full_scale_is_written_unchanged(self):
        _, mix = self._track("a.flac", amplitude=0.5)
        self._run(self.music, weights=QUIET)
        for stem, weight in zip(self._stems(), QUIET):
            np.testing.assert_allclose(stem, weight * mix, atol=2e-7)
        self.assertEqual(self._entries()[0]["gain"], 1.0)
        self.assertNotIn("lowered", self.out.getvalue())

    def test_the_manifest_records_the_factor(self):
        _, mix = self._track("a.flac")
        self._run(self.music, weights=HOT)
        entry = self._entries()[0]
        self.assertAlmostEqual(entry["gain"], TARGET_PEAK / (1.6 * float(np.max(np.abs(mix)))),
                               places=5)
        self.assertEqual(entry["clipped"], 0)


class RedoClippedTest(_Base):
    def _age_entry(self, index, **fields):
        path = self.library / MANIFEST_NAME
        document = json.loads(path.read_text())
        entry = document["entries"][index]
        for key in ("gain", "clipped"):
            entry.pop(key, None)
        entry.update(fields)
        path.write_text(json.dumps(document))

    def test_entries_without_a_factor_are_converted_again_and_clean_ones_are_not(self):
        self._track("old.flac", amplitude=0.5)
        self._track("new.flac", amplitude=0.5)
        self._run(self.music)
        old = next(i for i, e in enumerate(self._entries()) if e["source"].endswith("old.flac"))
        self._age_entry(old)  # as written before the fix: no gain, no clip count
        code, separator = self._run(self.music, "--redo-clipped")
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertEqual(len(separator.calls), 1)
        self.assertTrue(all(e["gain"] is not None for e in self._entries()))
        code, separator = self._run(self.music, "--redo-clipped")
        self.assertEqual(separator.calls, [])

    def test_an_entry_with_a_clip_count_is_converted_again(self):
        self._track("a.flac", amplitude=0.5)
        self._run(self.music)
        self._age_entry(0, gain=1.0, clipped=100)
        _, separator = self._run(self.music, "--redo-clipped")
        self.assertEqual(len(separator.calls), 1)
        self.assertEqual(self._entries()[0]["clipped"], 0)

    def test_without_the_flag_nothing_is_converted_again(self):
        self._track("a.flac", amplitude=0.5)
        self._run(self.music)
        self._age_entry(0)
        _, separator = self._run(self.music)
        self.assertEqual(separator.calls, [])

    def test_redo_clipped_replaces_the_set_under_its_own_name(self):
        self._track("a.flac", amplitude=0.9)
        self._run(self.music, weights=HOT)
        before = sorted(p.name for p in self.library.rglob("*.wav"))
        self._age_entry(0)
        self._run(self.music, "--redo-clipped", weights=HOT)
        self.assertEqual(sorted(p.name for p in self.library.rglob("*.wav")), before)

    def test_dry_run_plans_the_same_tracks(self):
        self._track("a.flac", amplitude=0.5)
        self._run(self.music)
        self._age_entry(0)
        self._run(self.music, "--redo-clipped", "--dry-run")
        self.assertIn("1 track(s)", self.out.getvalue())
        self.assertNotIn("0 track(s)", self.out.getvalue())


if __name__ == "__main__":
    unittest.main()
