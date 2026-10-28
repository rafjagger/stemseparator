# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import soundfile as sf

from stemseparator import cli
from stemseparator.separator import SeparatorUnavailable
from tests import fakes
from tests.fakes import FakeSeparator


def refuse_factory():
    raise AssertionError("the separator must not be built")


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.library = self.root / "lib"
        self.inbox = self.root / "in"
        self.inbox.mkdir()
        self.config_path = self.root / "config.toml"
        self._write_config(f'library = "{self.library}"\nmodels = ["m1"]\nparallel = 2\n')
        self.out, self.err = io.StringIO(), io.StringIO()

    def _write_config(self, text):
        self.config_path.write_text(text)

    def _source(self, name="song.flac", seconds=1.0):
        path = self.inbox / name
        with sf.SoundFile(path, "w", 44100, 2, format=path.suffix[1:].upper()) as handle:
            handle.write(fakes.test_signal(seconds))
        return path

    def _main(self, *argv, separator=None, known_models=None):
        factory = (lambda: separator) if separator is not None else refuse_factory
        return cli.main([str(a) for a in argv], separator_factory=factory,
                        config_path=self.config_path, known_models=known_models,
                        out=self.out, err=self.err)

    def _library_files(self):
        return sorted(p for p in self.library.rglob("*") if p.is_file()) if self.library.exists() else []

    def test_dry_run_prints_names_and_estimate_and_never_builds_a_separator(self):
        source = self._source(seconds=252)
        code = self._main("--dry-run", "--artist", "Ann", "--album", "Rec", "--title", "Tune",
                          source)
        self.assertEqual(code, 0)
        text = self.out.getvalue()
        self.assertIn("Ann/Rec/Tune - 1 - drums.wav", text)
        self.assertIn("4:12", text)
        self.assertIn("~$", text)
        self.assertEqual(self._library_files(), [])

    def test_run_writes_sets_and_exits_0(self):
        a, b = self._source("a.flac"), self._source("b.flac")
        separator = FakeSeparator()
        with mock.patch("stemseparator.cost.cost_usd", return_value=0.0345):
            code = self._main(a, b, separator=separator)
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertEqual(len(separator.calls), 2)
        self.assertEqual(len([p for p in self._library_files() if p.suffix == ".wav"]), 8)
        text = self.out.getvalue()
        self.assertIn("[1/2]", text)
        self.assertIn("[2/2]", text)
        self.assertIn("2 done, 0 failed", text)
        self.assertIn("~$0.07 (approx.)", text)

    def test_one_failure_exits_1_and_lists_reason(self):
        good = self._source("good.flac")
        bad = self.inbox / "bad.flac"
        bad.write_bytes(b"not audio at all")
        code = self._main(good, bad, separator=FakeSeparator())
        self.assertEqual(code, 1)
        text = self.out.getvalue()
        self.assertIn("1 done, 1 failed", text)
        self.assertIn("bad.flac", text)

    def test_refused_mp3_counts_as_failed(self):
        mp3 = self.inbox / "x.mp3"
        mp3.write_bytes(b"x")
        good = self._source()
        code = self._main(good, mp3, separator=FakeSeparator())
        self.assertEqual(code, 1)
        self.assertIn("x.mp3", self.out.getvalue())

    def test_all_inputs_refused_skips_preflight(self):
        code = self._main(self.inbox / "missing.flac")
        self.assertEqual(code, 1)
        self.assertIn("not found", self.out.getvalue())

    def test_preflight_failure_stops_before_upload(self):
        source = self._source()
        separator = FakeSeparator(
            unavailable=SeparatorUnavailable("No Modal token: run `modal token new`"))
        code = self._main(source, separator=separator)
        self.assertEqual(code, 2)
        self.assertIn("modal token new", self.err.getvalue())
        self.assertEqual(separator.calls, [])

    def test_title_with_two_inputs_is_usage_error(self):
        a, b = self._source("a.flac"), self._source("b.flac")
        code = self._main("--title", "X", a, b, separator=FakeSeparator())
        self.assertEqual(code, 2)
        self.assertIn("--title", self.err.getvalue())

    def test_unknown_model_in_config_exits_2(self):
        source = self._source()
        code = self._main(source, known_models=frozenset({"other"}))
        self.assertEqual(code, 2)
        self.assertIn("m1", self.err.getvalue())

    def test_default_known_models_are_the_registry(self):
        code = cli.main([str(self._source())], separator_factory=refuse_factory,
                        config_path=self.config_path, out=self.out, err=self.err)
        self.assertEqual(code, 2)
        self.assertIn("m1", self.err.getvalue())
        self.assertIn("scnet_xl_ihf", self.err.getvalue())

    def test_separator_that_cannot_be_built_exits_2_without_traceback(self):
        def broken_factory():
            raise ImportError("No module named 'modal'")
        code = cli.main([str(self._source())], separator_factory=broken_factory,
                        config_path=self.config_path, known_models=None,
                        out=self.out, err=self.err)
        self.assertEqual(code, 2)
        self.assertIn("No module named 'modal'", self.err.getvalue())
        self.assertNotIn("Traceback", self.err.getvalue())

    def test_bad_config_exits_2_with_path(self):
        self._write_config("parallel = 0\n")
        code = self._main(self._source())
        self.assertEqual(code, 2)
        self.assertIn(str(self.config_path), self.err.getvalue())

    def test_library_flag_overrides_config(self):
        other = self.root / "other-lib"
        code = self._main("--library", other, self._source(), separator=FakeSeparator())
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertTrue(any(other.rglob("*.wav")))
        self.assertFalse(self.library.exists())

    def test_long_flag_passes_through(self):
        source = self._source()
        with mock.patch("stemseparator.pipeline.MAX_SECONDS", 0.5):
            refused = self._main(source, separator=FakeSeparator())
            allowed = self._main("--long", source, separator=FakeSeparator())
        self.assertEqual(refused, 1)
        self.assertIn("--long", self.out.getvalue())
        self.assertEqual(allowed, 0)

    def test_ctrl_c_exits_130(self):
        source = self._source()
        with mock.patch("stemseparator.cli.run_batch", side_effect=KeyboardInterrupt):
            code = self._main(source, separator=FakeSeparator())
        self.assertEqual(code, 130)
        self.assertIn("interrupted, nothing half-written", self.err.getvalue())

    def test_unknown_cost_prints_cost_unknown(self):
        source = self._source()
        separator = FakeSeparator()
        with mock.patch("stemseparator.cost.cost_usd", return_value=None):
            code = self._main(source, separator=separator)
        self.assertEqual(code, 0)
        self.assertIn("cost unknown", self.out.getvalue())

    def test_format_helpers(self):
        self.assertEqual(cli.format_duration(252), "4:12")
        self.assertEqual(cli.format_duration(65.4), "1:05")
        self.assertEqual(cli.format_cost(0.0312), "~$0.03")
        self.assertEqual(cli.format_cost(None), "cost unknown")


if __name__ == "__main__":
    unittest.main()
