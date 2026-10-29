# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from stemseparator import cli, shootout
from stemseparator.pipeline import Naming
from stemseparator.remote.registry import MODEL_SETS
from stemseparator.separator import SeparatorError
from tests import fakes
from tests.fakes import FakeSeparator


def no_sleep(seconds):
    pass


class RegistryNamesFake(FakeSeparator):
    """Names its stems as the registered set would; fails the sets in `failing` permanently."""

    def __init__(self, failing=(), **kwargs):
        super().__init__(**kwargs)
        self.failing = set(failing)

    def separate(self, audio, sample_rate, model_set):
        if model_set in self.failing:
            self._register_call(audio.shape[0], model_set)
            raise SeparatorError("no GPU today", transient=False)
        separation = super().separate(audio, sample_rate, model_set)
        return dataclasses.replace(separation, stem_names=MODEL_SETS[model_set].stem_names)


class ShootoutTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.library = self.root / "lib"
        self.library.mkdir()
        self.track = self._track(self.root / "Big Party.wav")

    def _track(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, fakes.test_signal(1.0), 44100)
        return path

    def _run(self, sets, separator=None, track=None, naming=Naming()):
        return shootout.run_shootout(
            track or self.track, list(sets), library=self.library,
            separator=separator or RegistryNamesFake(), naming=naming, parallel=2,
            allow_long=False, sleep=no_sleep)

    def _folder(self, set_name):
        return self.library / shootout.SHOOTOUT_DIR / set_name

    def _runs(self):
        return json.loads((self.library / shootout.SHOOTOUT_DIR / "runs.json").read_text())

    def test_each_set_lands_in_its_own_folder_named_after_its_stems(self):
        runs = self._run(["specialist", "space4"])
        self.assertTrue(all(run.ok for run in runs), [run.reason for run in runs])
        self.assertEqual(sorted(p.name for p in self._folder("space4").iterdir()),
                         ["Big Party - 1 - low.wav", "Big Party - 2 - tops.wav",
                          "Big Party - 3 - music.wav", "Big Party - 4 - vocals.wav"])
        self.assertEqual(sorted(p.name for p in self._folder("specialist").iterdir()),
                         ["Big Party - 1 - drums.wav", "Big Party - 2 - bass.wav",
                          "Big Party - 3 - other.wav", "Big Party - 4 - vocals.wav"])

    def test_every_set_is_separated_once_with_its_name(self):
        separator = RegistryNamesFake()
        self._run(["specialist", "space4", "sw4"], separator)
        self.assertEqual(sorted(name for _, name in separator.calls),
                         ["space4", "specialist", "sw4"])

    def test_the_written_stems_sum_back_to_the_track(self):
        self._run(["space4"])
        total = sum(sf.read(p, dtype="float64")[0] for p in self._folder("space4").iterdir())
        mix = sf.read(self.track, dtype="float64")[0]
        self.assertLessEqual(float(np.max(np.abs(total - mix))), 2 ** -14)

    def test_runs_file_records_gpu_time_cost_and_residual_per_set(self):
        self._run(["specialist", "space4"])
        runs = self._runs()
        self.assertEqual(Path(runs["track"]), self.track.resolve())
        self.assertEqual(runs["title"], "Big Party")
        self.assertEqual(list(runs["sets"]), ["specialist", "space4"])
        entry = runs["sets"]["space4"]
        self.assertEqual(entry["gpu_seconds"], 10.0)
        self.assertEqual(entry["gpu"], "FAKE")
        self.assertIn("cost_usd", entry)
        self.assertEqual(entry["stem_names"], ["low", "tops", "music", "vocals"])
        self.assertIn("residual_db", entry)

    def test_a_failed_set_leaves_no_folder_and_the_others_still_land(self):
        runs = {run.set_name: run for run in self._run(["specialist", "space4"],
                                                       RegistryNamesFake(failing={"space4"}))}
        self.assertFalse(runs["space4"].ok)
        self.assertIn("no GPU today", runs["space4"].reason)
        self.assertTrue(runs["specialist"].ok)
        self.assertFalse(self._folder("space4").exists())
        self.assertEqual(sorted(p.name for p in (self.library / "shootout").iterdir()),
                         ["runs.json", "specialist"])

    def test_running_one_set_again_replaces_it_and_keeps_the_others(self):
        self._run(["specialist", "space4"])
        stale = self._folder("space4") / "stale.txt"
        stale.write_text("x")
        self._run(["space4"])
        self.assertFalse(stale.exists())
        self.assertEqual(list(self._runs()["sets"]), ["specialist", "space4"])

    def test_a_track_inside_the_library_is_fine_but_not_one_inside_the_shootout(self):
        inside = self._track(self.library / "Big Party.wav")
        self.assertTrue(self._run(["space4"], track=inside)[0].ok)
        nested = self._track(self.library / "shootout" / "x.wav")
        with self.assertRaises(shootout.ShootoutError):
            self._run(["space4"], track=nested)

    def test_a_shootout_of_another_track_is_not_mixed_in(self):
        self._run(["space4"])
        other = self._track(self.root / "Other.wav")
        separator = RegistryNamesFake()
        with self.assertRaises(shootout.ShootoutError) as caught:
            self._run(["specialist"], separator, track=other)
        self.assertIn("Big Party", str(caught.exception))
        self.assertEqual(separator.calls, [])


class ShootoutCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.library = self.root / "lib"
        self.config_path = self.root / "config.toml"
        self.config_path.write_text(f'library = "{self.library}"\n')
        self.track = self.root / "Big Party.wav"
        sf.write(self.track, fakes.test_signal(1.0), 44100)
        self.out, self.err = io.StringIO(), io.StringIO()

    def _main(self, *argv, separator=None):
        def refuse():
            raise AssertionError("the separator must not be built")
        factory = (lambda: separator) if separator is not None else refuse
        return cli.main([str(a) for a in argv], separator_factory=factory,
                        config_path=self.config_path, out=self.out, err=self.err)

    def test_shootout_runs_the_sets_and_writes_the_report(self):
        code = self._main("--shootout", self.track, "--sets", "specialist,space4",
                          separator=RegistryNamesFake())
        self.assertEqual(code, 0, self.err.getvalue())
        text = self.out.getvalue()
        self.assertIn("specialist: done", text)
        self.assertIn("space4: done", text)
        report = self.library / "shootout" / "report.md"
        self.assertTrue(report.is_file())
        self.assertIn("space4", report.read_text())

    def test_a_failed_set_exits_1(self):
        code = self._main("--shootout", self.track, "--sets", "specialist,space4",
                          separator=RegistryNamesFake(failing={"space4"}))
        self.assertEqual(code, 1)
        self.assertIn("space4: failed", self.out.getvalue())

    def test_shootout_dry_run_estimates_each_set_and_builds_nothing(self):
        code = self._main("--dry-run", "--shootout", self.track, "--sets", "specialist,space4")
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertIn("space4", self.out.getvalue())
        self.assertIn("~$", self.out.getvalue())
        self.assertFalse((self.library / "shootout").exists())

    def test_unknown_set_exits_2(self):
        self.assertEqual(self._main("--shootout", self.track, "--sets", "space4,nope"), 2)
        self.assertIn("nope", self.err.getvalue())

    def test_shootout_needs_sets_and_takes_no_other_paths(self):
        self.assertEqual(self._main("--shootout", self.track), 2)
        self.assertEqual(self._main("--shootout", self.track, "--sets", "space4", self.track), 2)

    def test_no_paths_without_shootout_is_a_usage_error(self):
        self.assertEqual(self._main(), 2)

    def test_compare_writes_the_report_of_a_shootout_folder(self):
        self._main("--shootout", self.track, "--sets", "specialist,space4",
                   separator=RegistryNamesFake())
        report = self.library / "shootout" / "report.md"
        report.unlink()
        self.out = io.StringIO()
        self.assertEqual(self._main("compare", self.library / "shootout"), 0, self.err.getvalue())
        self.assertTrue(report.is_file())
        self.assertIn("report.md", self.out.getvalue())

    def test_compare_of_a_folder_without_sets_exits_2(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(self._main("compare", empty, "--mix", self.track), 2)
        self.assertIn("no stem sets", self.err.getvalue())


if __name__ == "__main__":
    unittest.main()
