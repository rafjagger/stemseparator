# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The library mirrors the source folders; sets in the old Artist/Album layout are replaced."""

import io
import json
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path

import soundfile as sf

from stemseparator import cli
from stemseparator.manifest import MANIFEST_NAME
from tests import fakes
from tests.fakes import FakeSeparator


class LayoutBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.music = self.root / "Music" / "sets"
        self.library = self.root / "lib"
        self.config_path = self.root / "config.toml"
        self.config_path.write_text(f'library = "{self.library}"\nmodels = "specialist"\n'
                                    'parallel = 1\n')
        self.out, self.err = io.StringIO(), io.StringIO()

    def _track(self, relative):
        path = self.music / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, fakes.test_signal(0.3, seed=zlib.crc32(relative.encode())), 44100,
                 format="FLAC")
        return path

    def _run(self, *argv):
        self.out.seek(0), self.out.truncate(), self.err.seek(0), self.err.truncate()
        separator = FakeSeparator()
        code = cli.main([str(a) for a in argv], separator_factory=lambda: separator,
                        config_path=self.config_path, known_sets=None,
                        out=self.out, err=self.err)
        self.assertEqual(code, 0, self.err.getvalue())
        return separator

    def _relative_files(self):
        return sorted(str(p.relative_to(self.library)) for p in self.library.rglob("*")
                      if p.is_file() and not p.name.startswith("."))

    def _manifest_path(self):
        return self.library / MANIFEST_NAME


class MirrorTest(LayoutBase):
    def test_a_recursive_run_mirrors_the_folders_below_the_given_one(self):
        self._track("house/a124 Foo.flac")
        self._track("top.flac")
        self._run(self.music)
        files = self._relative_files()
        for number, name in enumerate(("drums", "bass", "other", "vocals"), 1):
            self.assertIn(f"sets/house/a124 Foo - {number} - {name}.flac", files)
        self.assertIn("sets/top - 1 - drums.flac", files)
        self.assertIn("sets/house/originals/a124 Foo.flac", files)

    def test_a_single_file_goes_into_its_parent_folders_name(self):
        source = self._track("house/a098 apaj - Big Party.flac")
        self._run(source)
        self.assertIn("house/a098 apaj - Big Party - 1 - drums.flac", self._relative_files())

    def test_no_unknown_artist_or_album_folder_appears(self):
        self._run(self._track("x.flac"))
        self.assertFalse(any("Unknown" in name for name in self._relative_files()))


class OldLayoutTest(LayoutBase):
    def _convert_in_the_old_layout(self, relative="song.flac"):
        """Moves a fresh set to Ann/Rec/Old Song and rewrites its manifest entry to match."""
        source = self._track(relative)
        self._run(self.music)
        old = self.library / "Ann" / "Rec"
        old.mkdir(parents=True)
        new_dir = self.library / "sets"
        for path in sorted(new_dir.glob("song - *")):
            shutil.move(path, old / path.name.replace("song", "Old Song"))
        (old / "originals").mkdir()
        shutil.move(new_dir / "originals" / "song.flac", old / "originals" / "Old Song.flac")
        (new_dir / "originals").rmdir()
        new_dir.rmdir()
        document = json.loads(self._manifest_path().read_text())
        document["entries"][0].update(output_folder="Ann/Rec", title="Old Song")
        document["entries"][0].pop("gain"), document["entries"][0].pop("clipped")  # as before
        self._manifest_path().write_text(json.dumps(document))
        return source, old

    def test_an_old_set_still_counts_as_done(self):
        self._convert_in_the_old_layout()
        separator = self._run(self.music)
        self.assertEqual(separator.calls, [])

    def test_redo_writes_the_new_layout_and_removes_the_old_set_and_its_empty_folders(self):
        self._convert_in_the_old_layout()
        self._run(self.music, "--redo")
        files = self._relative_files()
        self.assertEqual(len(files), 5)
        self.assertTrue(all(name.startswith("sets/") for name in files), files)
        self.assertFalse((self.library / "Ann").exists())
        entry, = json.loads(self._manifest_path().read_text())["entries"]
        self.assertEqual((entry["output_folder"], entry["title"]), ("sets", "song"))

    def test_redo_leaves_files_it_did_not_write_and_the_folder_that_holds_them(self):
        _, old = self._convert_in_the_old_layout()
        (old / "notes.txt").write_text("mine")
        (old / "originals" / "Other.flac").write_bytes(b"x")
        self._run(self.music, "--redo")
        self.assertEqual((old / "notes.txt").read_text(), "mine")
        self.assertTrue((old / "originals" / "Other.flac").exists())
        self.assertFalse((old / "Old Song - 1 - drums.flac").exists())
        self.assertFalse((old / "originals" / "Old Song.flac").exists())

    def test_redo_clipped_moves_an_old_set_too(self):
        self._convert_in_the_old_layout()
        self._run(self.music, "--redo-clipped")
        self.assertFalse((self.library / "Ann").exists())
        self.assertIn("sets/song - 1 - drums.flac", self._relative_files())

    def test_the_dry_run_shows_the_move_and_changes_nothing(self):
        self._convert_in_the_old_layout()
        before = self._relative_files()
        self._run(self.music, "--redo", "--dry-run")
        self.assertIn("sets/song - 1 - drums.flac", self.out.getvalue())
        self.assertIn("removes the old one at Ann/Rec/Old Song", self.out.getvalue())
        self.assertEqual(self._relative_files(), before)

    def test_a_redo_in_the_same_place_removes_nothing(self):
        self._track("song.flac")
        self._run(self.music)
        before = self._relative_files()
        self._run(self.music, "--redo")
        self.assertEqual(self._relative_files(), before)


if __name__ == "__main__":
    unittest.main()
