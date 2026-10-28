# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from stemseparator.layout import is_inside, plan_track
from stemseparator.staging import StagedSet, StagingError, commit, staging_root


def _fill(staged: StagedSet) -> None:
    for path in (*staged.stem_paths, staged.original_path):
        path.write_bytes(path.name.encode())


def _files_under(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file()]


class StagingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.library = Path(self._tmp.name) / "library"
        self.library.mkdir()
        self.source = Path("song.flac")

    def _plan_for(self, title="Intro"):
        return lambda: plan_track(self.library, "Artist", "Album", title, self.source)

    def test_commit_moves_five_files_and_cleans_staging(self):
        with StagedSet(self.library) as staged:
            _fill(staged)
            plan = commit(staged, self._plan_for())
        for path in (*plan.stem_paths, plan.original_path):
            self.assertTrue(path.is_file(), path)
        self.assertEqual(len(_files_under(self.library)), 5)
        self.assertFalse(staging_root(self.library).exists())

    def test_staging_is_outside_library(self):
        self.assertFalse(is_inside(staging_root(self.library), self.library))
        with StagedSet(self.library) as staged:
            self.assertFalse(is_inside(staged.original_path, self.library))

    def test_failed_rename_rolls_back(self):
        real_rename = os.rename
        calls = []

        def flaky(src, dst):
            calls.append(src)
            if len(calls) == 3:
                raise OSError("disk on fire")
            real_rename(src, dst)

        with StagedSet(self.library) as staged:
            _fill(staged)
            with mock.patch("stemseparator.staging.os.rename", flaky):
                with self.assertRaises(OSError):
                    commit(staged, self._plan_for())
            self.assertEqual(_files_under(self.library), [])
            for path in (*staged.stem_paths, staged.original_path):
                self.assertTrue(path.is_file(), path)
        self.assertFalse(staging_root(self.library).exists())

    def test_parallel_commits_with_one_title_get_distinct_names(self):
        barrier = threading.Barrier(2)
        plans, errors = [], []

        def worker():
            try:
                with StagedSet(self.library) as staged:
                    _fill(staged)
                    barrier.wait(timeout=10)
                    plans.append(commit(staged, self._plan_for()))
            except Exception as error:  # surfaced below
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertEqual(sorted(p.title for p in plans), ["Intro", "Intro (2)"])
        stems = [p for p in _files_under(self.library) if p.suffix == ".wav"]
        self.assertEqual(len(stems), 8)
        self.assertEqual(len(_files_under(self.library)), 10)

    def test_exit_without_commit_leaves_library_untouched(self):
        with StagedSet(self.library) as staged:
            _fill(staged)
        self.assertEqual(list(self.library.iterdir()), [])
        self.assertFalse(staging_root(self.library).exists())

    def test_exit_keeps_other_jobs_folders(self):
        with StagedSet(self.library) as first:
            with StagedSet(self.library):
                pass
            self.assertTrue(first.original_path.parent.is_dir())
        self.assertFalse(staging_root(self.library).exists())

    def test_missing_library_is_created(self):
        library = Path(self._tmp.name) / "fresh" / "library"
        with StagedSet(library):
            self.assertTrue(library.is_dir())

    def test_different_filesystem_is_refused(self):
        real_stat = Path.stat

        def fake_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if path.name != "library":
                return result
            return mock.Mock(st_dev=result.st_dev + 1)

        with mock.patch.object(Path, "stat", fake_stat):
            with self.assertRaises(StagingError):
                StagedSet(self.library)
        self.assertFalse(staging_root(self.library).exists())


if __name__ == "__main__":
    unittest.main()
