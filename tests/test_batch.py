# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import tempfile
import threading
import time
import unittest
from pathlib import Path

import soundfile as sf

from stemseparator.pipeline import BatchSummary, Naming, TrackResult, collect_inputs, run_batch
from tests import fakes
from tests.fakes import FakeSeparator


class TrackingSeparator(FakeSeparator):
    """Holds each call briefly so overlapping calls are observable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active = 0
        self.max_active = 0
        self._count_lock = threading.Lock()

    def separate(self, audio, sample_rate, models):
        with self._count_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.03)
            return super().separate(audio, sample_rate, models)
        finally:
            with self._count_lock:
                self._active -= 1


class BatchTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.library = root / "lib"
        self.inbox = root / "in"
        self.inbox.mkdir()

    def _flac(self, name):
        path = self.inbox / name
        sf.write(path, fakes.test_signal(0.2), 44100, format="FLAC")
        return path

    def _touch(self, name, data=b"x"):
        path = self.inbox / name
        path.write_bytes(data)
        return path


class CollectInputsTest(BatchTestBase):
    def test_collect_inputs_folder_filters_and_sorts(self):
        self._flac("b.flac")
        self._flac("a.WAV")
        self._touch(".hidden.flac")
        self._touch("notes.txt")
        (self.inbox / "sub").mkdir()
        files, refused = collect_inputs([self.inbox])
        self.assertEqual([p.name for p in files], ["a.WAV", "b.flac"])
        self.assertEqual(refused, [])

    def test_mp3_in_folder_reported_as_refused(self):
        self._flac("a.flac")
        self._touch("song.MP3")
        self._touch("cover.jpg")
        files, refused = collect_inputs([self.inbox])
        self.assertEqual([p.name for p in files], ["a.flac"])
        self.assertEqual([r.source.name for r in refused], ["song.MP3"])
        self.assertFalse(refused[0].ok)
        self.assertEqual(refused[0].reason, "MP3 is not supported: use FLAC, WAV or AIFF")

    def test_missing_path_reported(self):
        files, refused = collect_inputs([self.inbox / "gone.flac"])
        self.assertEqual(files, [])
        self.assertEqual(len(refused), 1)
        self.assertFalse(refused[0].ok)
        self.assertIn("not found", refused[0].reason)

    def test_explicit_file_with_other_suffix_is_reported(self):
        mp3 = self._touch("x.mp3")
        txt = self._touch("x.txt")
        files, refused = collect_inputs([mp3, txt])
        self.assertEqual(files, [])
        self.assertEqual([r.reason for r in refused],
                         ["MP3 is not supported: use FLAC, WAV or AIFF", "not an audio file"])

    def test_explicit_file_is_kept_in_order(self):
        b, a = self._flac("b.flac"), self._flac("a.flac")
        files, _ = collect_inputs([b, a])
        self.assertEqual(files, [b, a])


class RunBatchTest(BatchTestBase):
    def _run(self, sources, separator, parallel=2, report=None):
        return run_batch(
            sources, parallel=parallel, report=report or (lambda result: None),
            library=self.library, separator=separator, models=("m",), naming=Naming(),
            allow_long=False, sleep=lambda seconds: None)

    def test_batch_continues_after_a_failure(self):
        sources = [self._flac("a.flac"), self._touch("b.flac", b"garbage!"), self._flac("c.flac")]
        reported = []
        summary = self._run(sources, FakeSeparator(), report=reported.append)
        self.assertEqual((summary.done, summary.failed), (2, 1))
        self.assertEqual(len(reported), 3)
        self.assertEqual([r.source for r in summary.results], sources)

    def test_unexpected_error_in_one_track_fails_only_that_track(self):
        sources = [self._flac("a.flac"), self._flac("b.flac"), self._flac("c.flac")]

        class BrokenOnB(FakeSeparator):
            def separate(self, audio, sample_rate, models):
                if len(self.calls) == 1:
                    self.calls.append((0, models))
                    raise ValueError("odd")
                return super().separate(audio, sample_rate, models)

        summary = self._run(sources, BrokenOnB(), parallel=1)
        self.assertEqual((summary.done, summary.failed), (2, 1))
        self.assertEqual(summary.results[1].reason, "internal error: ValueError('odd')")

    def test_batch_never_exceeds_parallel(self):
        sources = [self._flac(f"t{i}.flac") for i in range(5)]
        separator = TrackingSeparator()
        summary = self._run(sources, separator, parallel=2)
        self.assertEqual(summary.done, 5)
        self.assertEqual(separator.max_active, 2)

    def test_report_runs_on_the_calling_thread(self):
        sources = [self._flac(f"t{i}.flac") for i in range(3)]
        threads = set()
        self._run(sources, FakeSeparator(), report=lambda r: threads.add(threading.current_thread()))
        self.assertEqual(threads, {threading.current_thread()})

    def test_keyboard_interrupt_cancels_and_reraises(self):
        sources = [self._flac(f"t{i}.flac") for i in range(4)]
        separator = TrackingSeparator()

        def interrupt(result):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self._run(sources, separator, parallel=1, report=interrupt)
        self.assertLess(len(separator.calls), 4)


class SummaryTest(unittest.TestCase):
    def _result(self, ok=True, gpu_seconds=0.0, gpu=""):
        return TrackResult(Path("x"), ok, gpu_seconds=gpu_seconds, gpu=gpu)

    def test_summary_totals(self):
        summary = BatchSummary((
            self._result(gpu_seconds=10.0, gpu="NVIDIA A10"),
            self._result(gpu_seconds=20.0, gpu="NVIDIA L4"),
            self._result(ok=False),
        ))
        self.assertEqual((summary.done, summary.failed), (2, 1))
        self.assertEqual(summary.gpu_seconds, 30.0)
        self.assertAlmostEqual(summary.cost_usd, 10 * 0.000306 + 20 * 0.000222)

    def test_cost_is_none_if_any_gpu_unknown(self):
        summary = BatchSummary((self._result(gpu_seconds=1.0, gpu="A10"),
                                self._result(gpu_seconds=1.0, gpu="FAKE")))
        self.assertIsNone(summary.cost_usd)

    def test_cost_of_empty_batch_is_zero(self):
        self.assertEqual(BatchSummary(()).cost_usd, 0.0)
