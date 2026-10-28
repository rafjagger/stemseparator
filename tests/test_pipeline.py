# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import dataclasses
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from stemseparator import pipeline
from stemseparator.audio_io import Tags, Track, write_stem_24bit
from stemseparator.pipeline import Naming, process_track, resolve_names
from stemseparator.separator import SeparatorError
from stemseparator.staging import staging_root
from tests import fakes
from tests.fakes import FakeSeparator

MODELS = ("m",)


def no_sleep(seconds):
    pass


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.library = root / "lib"
        self.inbox = root / "in"
        self.inbox.mkdir()

    def _source(self, name="song.flac", audio=None, rate=44100, tags=None):
        audio = fakes.test_signal(1.0, rate) if audio is None else audio
        path = self.inbox / name
        with sf.SoundFile(path, "w", rate, 2, format=path.suffix[1:].upper()) as f:
            for key, value in (tags or {}).items():
                setattr(f, key, value)
            f.write(audio)
        return path

    def _run(self, source, separator=None, naming=Naming(), allow_long=False):
        return process_track(
            source, library=self.library, separator=separator or FakeSeparator(), models=MODELS,
            naming=naming, allow_long=allow_long, sleep=no_sleep)

    def _library_files(self):
        return sorted(p for p in self.library.rglob("*") if p.is_file()) if self.library.exists() else []

    def _assert_nothing_left(self):
        self.assertEqual(self._library_files(), [])
        self.assertFalse(staging_root(self.library).exists())

    def test_happy_path_writes_one_set(self):
        audio = fakes.test_signal(1.0)
        source = self._source(audio=audio, tags={"title": "T", "artist": "A", "album": "B"})
        result = self._run(source)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.gpu, "FAKE")
        self.assertEqual(result.gpu_seconds, 10.0)
        self.assertEqual(result.plan.album_dir, self.library / "A" / "B")
        self.assertEqual(result.plan.title, "T")
        written = [sf.read(p, dtype="float64") for p in result.plan.stem_paths]
        for path in result.plan.stem_paths:
            info = sf.info(path)
            self.assertEqual((info.subtype, info.samplerate), ("PCM_24", 44100))
        self.assertTrue(result.plan.original_path.is_file())
        total = sum(data for data, _ in written)
        self.assertLessEqual(float(np.max(np.abs(total - sf.read(source, dtype="float64")[0]))), 2**-22)
        self.assertLess(result.residual_db, -100)

    def test_flags_override_tags(self):
        source = self._source(tags={"title": "T", "artist": "A", "album": "B"})
        result = self._run(source, naming=Naming(artist="FA", album="FB", title="FT"))
        self.assertEqual(result.plan.album_dir, self.library / "FA" / "FB")
        self.assertEqual(result.plan.title, "FT")

    def test_no_tags_uses_file_stem_and_unknowns(self):
        result = self._run(self._source("my song.flac"))
        self.assertEqual(result.plan.album_dir, self.library / "Unknown Artist" / "Unknown Album")
        self.assertEqual(result.plan.title, "my song")

    def test_albumartist_wins_over_artist(self):
        track = Track(Path("x.flac"), fakes.test_signal(0.1), 44100,
                      Tags(title="T", artist="Feat", album="B", albumartist="Band"))
        self.assertEqual(resolve_names(track, Naming()), ("Band", "B", "T"))
        self.assertEqual(resolve_names(track, Naming(artist="Flag"))[0], "Flag")
        solo = dataclasses.replace(track, tags=Tags(artist="Feat"))
        self.assertEqual(resolve_names(solo, Naming()), ("Feat", "Unknown Album", "x"))

    def test_undecodable_is_skipped_with_reason(self):
        bad = self.inbox / "bad.flac"
        bad.write_bytes(b"not audio")
        fake = FakeSeparator()
        result = self._run(bad, fake)
        self.assertFalse(result.ok)
        self.assertTrue(result.reason)
        self.assertEqual(fake.calls, [])

    def test_source_inside_library_refused(self):
        self.library.mkdir()
        source = self.library / "song.flac"
        sf.write(source, fakes.test_signal(0.2), 44100, format="FLAC")
        fake = FakeSeparator()
        result = self._run(source, fake)
        self.assertFalse(result.ok)
        self.assertIn("library", result.reason)
        self.assertEqual(fake.calls, [])

    def test_long_track_refused_unless_long(self):
        source = self._source()
        fake = FakeSeparator()
        with mock.patch.object(pipeline, "MAX_SECONDS", 0.5):
            refused = self._run(source, fake)
            self.assertFalse(refused.ok)
            self.assertEqual(refused.reason, "longer than 20 min (use --long)")
            self.assertEqual(fake.calls, [])
            self.assertTrue(self._run(source, fake, allow_long=True).ok)

    def test_long_track_refused_from_the_header_without_decoding(self):
        source = self._source()
        with mock.patch.object(pipeline, "MAX_SECONDS", 0.5), \
                mock.patch.object(pipeline, "read_track", side_effect=AssertionError("decoded")):
            result = self._run(source)
        self.assertEqual(result.reason, "longer than 20 min (use --long)")

    def test_format_refusal_comes_before_the_length_check(self):
        mp3 = self.inbox / "x.flac"
        mp3.write_bytes(b"ID3" + bytes(64))
        with mock.patch.object(pipeline, "MAX_SECONDS", 0.0):
            result = self._run(mp3)
        self.assertEqual(result.reason, "MP3 is not supported: use FLAC, WAV or AIFF")

    def test_transient_error_retried_twice_then_fails(self):
        fake = FakeSeparator(fail_times=3)
        sleeps = []
        result = process_track(self._source(), library=self.library, separator=fake, models=MODELS,
                               naming=Naming(), allow_long=False, sleep=sleeps.append)
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "boom")
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(sleeps, [1, 2])
        self._assert_nothing_left()

    def test_two_failures_then_success(self):
        fake = FakeSeparator(fail_times=2)
        self.assertTrue(self._run(self._source(), fake).ok)
        self.assertEqual(len(fake.calls), 3)

    def test_permanent_error_not_retried(self):
        fake = FakeSeparator(fail_times=3, error=SeparatorError("bad", transient=False))
        result = self._run(self._source(), fake)
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "bad")
        self.assertEqual(len(fake.calls), 1)
        self._assert_nothing_left()

    def test_interrupt_leaves_no_files(self):
        written = []

        def interrupting_write(path, audio, rate):
            if len(written) == 2:
                raise KeyboardInterrupt
            written.append(path)
            return write_stem_24bit(path, audio, rate)

        with mock.patch.object(pipeline, "write_stem_24bit", interrupting_write):
            with self.assertRaises(KeyboardInterrupt):
                self._run(self._source())
        self.assertEqual(len(written), 2)
        self._assert_nothing_left()

    def test_cancel_during_separation_writes_nothing(self):
        cancel = threading.Event()

        class CancelWhileRunning(FakeSeparator):
            def separate(self, audio, sample_rate, models):
                separation = super().separate(audio, sample_rate, models)
                cancel.set()  # the second Ctrl-C arrives while the GPU call runs
                return separation

        result = process_track(self._source(), library=self.library,
                               separator=CancelWhileRunning(), models=MODELS, naming=Naming(),
                               allow_long=False, sleep=no_sleep, cancel=cancel)
        self.assertFalse(result.ok)
        self.assertIn("abandoned", result.reason)
        self._assert_nothing_left()

    def test_cancel_before_start_never_calls_the_separator(self):
        cancel = threading.Event()
        cancel.set()
        fake = FakeSeparator()
        result = process_track(self._source(), library=self.library, separator=fake,
                               models=MODELS, naming=Naming(), allow_long=False,
                               sleep=no_sleep, cancel=cancel)
        self.assertFalse(result.ok)
        self.assertEqual(fake.calls, [])

    def test_48k_track_written_at_44k(self):
        result = self._run(self._source(rate=48000))
        self.assertTrue(result.ok, result.reason)
        for path in result.plan.stem_paths:
            info = sf.info(path)
            self.assertEqual((info.samplerate, info.subtype), (44100, "PCM_24"))

    def test_wrong_rate_from_separator_fails(self):
        class Wrong(FakeSeparator):
            def separate(self, audio, sample_rate, models):
                good = super().separate(audio, sample_rate, models)
                return dataclasses.replace(good, sample_rate=48000)

        result = self._run(self._source(), Wrong())
        self.assertFalse(result.ok)
        self.assertIn("48000", result.reason)
        self._assert_nothing_left()

    def _reshaping_separator(self, reshape):
        class Reshaped(FakeSeparator):
            def separate(self, audio, sample_rate, models):
                good = super().separate(audio, sample_rate, models)
                return dataclasses.replace(good, stems=reshape(good.stems))
        return Reshaped()

    def test_mono_result_fails_and_writes_nothing(self):
        result = self._run(self._source(), self._reshaping_separator(lambda s: s[:, :, :1]))
        self.assertFalse(result.ok)
        self.assertIn("channel", result.reason)
        self._assert_nothing_left()

    def test_result_of_the_wrong_length_fails_and_writes_nothing(self):
        result = self._run(self._source(), self._reshaping_separator(lambda s: s[:, :-1000]))
        self.assertFalse(result.ok)
        self.assertIn("length", result.reason)
        self._assert_nothing_left()

    def test_result_a_few_frames_off_is_accepted(self):
        result = self._run(self._source(), self._reshaping_separator(lambda s: s[:, :-10]))
        self.assertTrue(result.ok, result.reason)

    def test_residual_reported(self):
        result = self._run(self._source(), FakeSeparator(weights=(0.4, 0.3, 0.2, 0.0)))
        self.assertAlmostEqual(result.residual_db, 20 * np.log10(0.1), delta=0.1)

    def test_clipped_counted(self):
        audio = fakes.test_signal(1.0)
        audio = (audio * (0.9 / np.max(np.abs(audio)))).astype(np.float32)
        result = self._run(self._source(audio=audio), FakeSeparator(weights=(2, 0, 0, 0)))
        self.assertTrue(result.ok, result.reason)
        self.assertGreater(result.clipped, 0)


if __name__ == "__main__":
    unittest.main()
