# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""Batch conversion of local folders: recursion, the done check, status, dry-run, resume."""

import io
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import soundfile as sf

from stemseparator import cli
from stemseparator.manifest import MANIFEST_NAME, fingerprint
from tests import fakes
from tests.fakes import FakeSeparator


def refuse_factory():
    raise AssertionError("the separator must not be built")


class Interrupting(FakeSeparator):
    """Ctrl-C arrives while the given call is running."""

    def __init__(self, on_call):
        super().__init__()
        self.on_call = on_call

    def separate(self, audio, sample_rate, model_set):
        if len(self.calls) + 1 == self.on_call:
            self._register_call(audio.shape[0], model_set)
            raise KeyboardInterrupt
        return super().separate(audio, sample_rate, model_set)


class ConvertFoldersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.music = self.root / "music"
        self.music.mkdir()
        self.library = self.root / "lib"
        self.config_path = self.root / "config.toml"
        self.config_path.write_text(
            f'library = "{self.library}"\nmodels = "specialist"\nparallel = 1\n')
        self.out, self.err = io.StringIO(), io.StringIO()

    def _track(self, relative, seed=None, seconds=0.3, title=None):
        path = self.music / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        seed = zlib.crc32(relative.encode()) if seed is None else seed
        sf.write(path, fakes.test_signal(seconds, seed=seed), 44100, format="FLAC")
        if title is not None:
            self._tag(path, title)
        return path

    def _tag(self, path, title):
        import mutagen.flac
        flac = mutagen.flac.FLAC(path)
        flac["title"], flac["artist"], flac["album"] = title, "Ann", "Rec"
        flac.save()

    def _run(self, *argv, separator=None, command=()):
        self.out.seek(0), self.out.truncate(), self.err.seek(0), self.err.truncate()
        factory = (lambda: separator) if separator is not None else refuse_factory
        return cli.main([*command, *map(str, argv)], separator_factory=factory,
                        config_path=self.config_path, known_sets=None,
                        out=self.out, err=self.err)

    def _convert(self, *argv, separator=None):
        separator = separator or FakeSeparator()
        code = self._run(self.music, *argv, separator=separator)
        return code, separator

    def _status(self, *argv):
        return self._run(self.music, *argv, command=["status"])

    def _stem_files(self):
        if not self.library.exists():
            return []
        return sorted(p for p in self.library.rglob("*") if p.is_file()
                      and not p.name.startswith(".") and "originals" not in p.parts)

    def _manifest(self):
        return json.loads((self.library / MANIFEST_NAME).read_text())["entries"]


class RecursionTest(ConvertFoldersTest):
    def test_folders_are_walked_recursively(self):
        self._track("top.flac", title="Top")
        self._track("a/deep.flac", title="Deep")
        self._track("a/b/deeper.flac", title="Deeper")
        code, separator = self._convert()
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertEqual(len(separator.calls), 3)
        self.assertEqual(len(self._stem_files()), 12)

    def test_unsupported_formats_are_skipped_but_mp3_and_ogg_are_listed(self):
        self._track("a/song.flac", title="Song")
        for name in ("a/cover.jpg", "a/notes.txt"):
            (self.music / name).write_bytes(b"x")
        (self.music / "a" / "old.mp3").write_bytes(b"x")
        (self.music / "a" / "old.ogg").write_bytes(b"x")
        code, _ = self._convert()
        text = self.out.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("old.mp3: MP3 is not supported", text)
        self.assertIn("old.ogg: OGG is not supported", text)
        self.assertNotIn("cover.jpg", text)
        self.assertNotIn("notes.txt", text)

    def test_a_library_inside_the_music_folder_is_not_walked(self):
        self.library = self.music / "stems"
        self.config_path.write_text(
            f'library = "{self.library}"\nmodels = "specialist"\nparallel = 1\n')
        self._track("song.flac", title="Song")
        self.assertEqual(self._convert()[0], 0, self.err.getvalue())
        code, separator = self._convert()
        self.assertEqual(code, 0, self.out.getvalue())
        self.assertEqual(separator.calls, [])


class DoneCheckTest(ConvertFoldersTest):
    def test_second_run_skips_an_existing_set_and_writes_no_duplicate(self):
        self._track("song.flac", title="Song")
        self._convert()
        code, separator = self._convert()
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertEqual(separator.calls, [])
        self.assertEqual(len(self._stem_files()), 4)
        self.assertFalse(any("(2)" in p.name for p in self.library.rglob("*")))
        self.assertIn("skipping 1 track(s) already done", self.out.getvalue())

    def test_a_set_found_by_name_needs_no_manifest(self):
        self._track("song.flac", title="Song")
        self._convert()
        (self.library / MANIFEST_NAME).unlink()
        _, separator = self._convert()
        self.assertEqual(separator.calls, [])

    def test_a_set_with_a_missing_stem_is_not_done(self):
        self._track("song.flac", title="Song")
        self._convert()
        next(p for p in self._stem_files() if "4 - vocals" in p.name).unlink()
        (self.library / MANIFEST_NAME).unlink()
        _, separator = self._convert()
        self.assertEqual(len(separator.calls), 1)

    def test_stems_in_another_format_still_count(self):
        self._track("song.flac", title="Song")
        self._convert("--format", "wav")
        _, separator = self._convert("--format", "flac")
        self.assertEqual(separator.calls, [])

    def test_a_renamed_source_still_counts_as_done_through_the_manifest(self):
        source = self._track("song.flac", title="Song")
        self._convert()
        (self.music / "moved").mkdir()
        source.rename(self.music / "moved" / "other name.flac")
        # --title makes the name differ from the set's: only the manifest can know it is done.
        code, separator = self._convert("--title", "Different Title")
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertEqual(separator.calls, [])
        self.assertEqual(len(self._stem_files()), 4)

    def test_a_changed_file_is_new(self):
        source = self._track("song.flac", seed=1, title="Song")
        self._convert()
        sf.write(source, fakes.test_signal(0.3, seed=2), 44100, format="FLAC")
        self._tag(source, "Song")
        self.assertEqual(self._status(), 0)
        self.assertIn("changed", self.out.getvalue())
        _, separator = self._convert()
        self.assertEqual(len(separator.calls), 1)

    def test_redo_converts_again_and_replaces_the_set(self):
        self._track("song.flac", title="Song")
        self._convert()
        before = self._stem_files()
        code, separator = self._convert("--redo")
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertEqual(len(separator.calls), 1)
        self.assertEqual(self._stem_files(), before)
        self.assertEqual(len(self._manifest()), 1)

    def test_the_same_content_twice_in_one_run_is_converted_once(self):
        self._track("a.flac", seed=5, title="Song")
        self._track("copy/a.flac", seed=5, title="Song")
        _, separator = self._convert()
        self.assertEqual(len(separator.calls), 1)
        self.assertEqual(len(self._stem_files()), 4)

    def test_two_different_tracks_with_one_name_both_stay(self):
        self._track("a.flac", seed=5, title="Song")
        self._track("b.flac", seed=6, title="Song")
        _, separator = self._convert()
        self.assertEqual(len(separator.calls), 2)
        self.assertEqual(len(self._stem_files()), 8)

    def test_a_corrupt_manifest_stops_the_run_untouched(self):
        self._track("song.flac", title="Song")
        self.library.mkdir()
        (self.library / MANIFEST_NAME).write_text("{not json")
        code = self._run(self.music, separator=None)
        self.assertEqual(code, 2)
        self.assertIn(MANIFEST_NAME, self.err.getvalue())

    def test_the_manifest_records_what_the_issue_asks_for(self):
        source = self._track("song.flac", title="Song")
        self._convert()
        entry, = self._manifest()
        self.assertEqual(entry["source"], str(source.resolve()))
        self.assertEqual(entry["size"], source.stat().st_size)
        self.assertEqual(entry["sha256"], fingerprint(source).sha256)
        self.assertEqual(entry["model_set"], "specialist")
        self.assertEqual(entry["output_folder"], "Ann/Rec")
        for key in ("mtime", "date", "title"):
            self.assertIn(key, entry)


class StatusAndDryRunTest(ConvertFoldersTest):
    def test_status_lists_each_track_with_totals_and_uploads_nothing(self):
        self._track("done.flac", title="Done")
        self._convert()
        self._track("new.flac", title="New")
        self._track("sub/new2.flac", title="New Two")
        (self.music / "x.mp3").write_bytes(b"x")
        code = self._status()
        text = self.out.getvalue()
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertRegex(text, r"done +\S*done\.flac  → Ann/Rec/Done")
        self.assertRegex(text, r"pending +\S*new\.flac")
        self.assertRegex(text, r"unsupported +\S*x\.mp3")
        last = text.strip().splitlines()[-1]
        self.assertIn("1 done, 2 pending, 1 unsupported", last)
        self.assertIn("to do: 2 track(s), estimated ~$", last)

    def test_status_writes_nothing(self):
        self._track("song.flac", title="Song")
        self._status()
        self.assertFalse(self.library.exists())

    def test_dry_run_skips_done_tracks_and_prices_only_the_rest(self):
        self._track("done.flac", title="Done")
        self._convert()
        self._track("new.flac", title="New")
        code = self._run("--dry-run", self.music)
        text = self.out.getvalue()
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertIn("skipped: already done (Ann/Rec/Done)", text)
        self.assertIn("Ann/Rec/New - 1 - ", text)
        self.assertIn("1 track(s), estimated", text)
        self.assertIn("1 already done", text)
        self.assertEqual(len(self._stem_files()), 4)

    def test_dry_run_with_redo_plans_the_done_tracks_too(self):
        self._track("done.flac", title="Done")
        self._convert()
        self._run("--dry-run", "--redo", self.music)
        self.assertIn("1 track(s), estimated", self.out.getvalue())


class ResumeTest(ConvertFoldersTest):
    def test_an_interrupted_run_resumes_with_the_rest(self):
        for name in ("a", "b", "c"):
            self._track(f"{name}.flac", title=name.upper())
        code, _ = self._convert(separator=Interrupting(on_call=2))
        self.assertEqual(code, 130)
        self.assertEqual([e["title"] for e in self._manifest()], ["A"])
        code, separator = self._convert()
        self.assertEqual(code, 0, self.err.getvalue())
        self.assertEqual(len(separator.calls), 2)
        self.assertEqual(len(self._stem_files()), 12)
        self.assertFalse(any("(2)" in p.name for p in self.library.rglob("*")))
