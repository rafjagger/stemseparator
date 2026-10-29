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
        return lambda: plan_track(self.library, Path("Artist/Album"), title, self.source)

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
        planned = [threading.Event(), threading.Event()]
        plans, errors = [], []

        def plan_for(index):
            def plan():
                result = plan_track(self.library, Path("Artist/Album"), "Intro", self.source)
                planned[index].set()
                # Without the lock the other worker plans now, before any rename happened.
                planned[1 - index].wait(timeout=0.3)
                return result
            return plan

        def worker(index):
            try:
                with StagedSet(self.library) as staged:
                    _fill(staged)
                    plans.append(commit(staged, plan_for(index)))
            except Exception as error:  # surfaced below
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertEqual(sorted(p.title for p in plans), ["Intro", "Intro (2)"])
        stems = [p for p in _files_under(self.library) if p.suffix == ".flac" and "originals" not in p.parts]
        self.assertEqual(len(stems), 8)
        self.assertEqual(len(_files_under(self.library)), 10)

    def test_concurrent_job_creation_and_exit_do_not_race(self):
        errors = []

        def churn():
            try:
                for _ in range(100):
                    with StagedSet(self.library):
                        pass
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=churn) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(errors, [])
        self.assertFalse(staging_root(self.library).exists())

    def test_stat_failure_removes_job_folder(self):
        with mock.patch.object(Path, "stat", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                StagedSet(self.library)
        self.assertFalse(staging_root(self.library).exists())

    def test_failing_rollback_does_not_mask_original_error(self):
        real_rename = os.rename
        calls = []

        def flaky(src, dst):
            calls.append(src)
            if len(calls) == 3:
                raise OSError("first failure")
            if len(calls) == 4:
                raise OSError("rollback failure")
            real_rename(src, dst)

        with StagedSet(self.library) as staged:
            _fill(staged)
            with mock.patch("stemseparator.staging.os.rename", flaky):
                with self.assertRaisesRegex(OSError, "first failure"):
                    commit(staged, self._plan_for())
            self.assertEqual(len(calls), 5)  # the second rollback step still ran

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
            with self.assertRaises(StagingError) as caught:
                StagedSet(self.library)
        self.assertIn("inside another folder on the same drive", str(caught.exception))
        self.assertFalse(staging_root(self.library).exists())


if __name__ == "__main__":
    unittest.main()
