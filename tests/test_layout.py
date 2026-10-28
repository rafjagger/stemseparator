# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import tempfile
import unittest
from pathlib import Path

from stemseparator.layout import is_inside, plan_track


class PlanTrackTest(unittest.TestCase):
    def test_plan_paths(self):
        plan = plan_track(Path("/lib"), "AC/DC", "", "Intro", Path("/in/x.FLAC"),
                          exists=lambda p: False)
        album = Path("/lib/AC_DC/Unknown Album")
        self.assertEqual(plan.album_dir, album)
        self.assertEqual(plan.title, "Intro")
        self.assertEqual(plan.stem_paths[0], album / "Intro - 1 - drums.wav")
        self.assertEqual(plan.stem_paths[3], album / "Intro - 4 - vocals.wav")
        self.assertEqual(plan.original_path, album / "originals" / "Intro.FLAC")

    def test_empty_names_use_fallbacks(self):
        plan = plan_track(Path("/lib"), "", "", "", Path("/in/x.wav"),
                          exists=lambda p: False)
        self.assertEqual(plan.album_dir, Path("/lib/Unknown Artist/Unknown Album"))
        self.assertEqual(plan.title, "Untitled")

    def test_clash_gets_2_then_3(self):
        album = Path("/lib/A/B")
        taken = {album / "Intro - 1 - drums.wav", album / "originals" / "Intro (2).flac"}
        plan = plan_track(Path("/lib"), "A", "B", "Intro", Path("/in/x.flac"),
                          exists=lambda p: p in taken)
        self.assertEqual(plan.title, "Intro (3)")

    def test_clash_on_original_alone_counts(self):
        album = Path("/lib/A/B")
        taken = {album / "originals" / "Intro.flac"}
        plan = plan_track(Path("/lib"), "A", "B", "Intro", Path("/in/x.flac"),
                          exists=lambda p: p in taken)
        self.assertEqual(plan.title, "Intro (2)")

    def test_clash_on_any_stem_counts(self):
        album = Path("/lib/A/B")
        taken = {album / "Intro - 4 - vocals.wav"}
        plan = plan_track(Path("/lib"), "A", "B", "Intro", Path("/in/x.flac"),
                          exists=lambda p: p in taken)
        self.assertEqual(plan.title, "Intro (2)")


class IsInsideTest(unittest.TestCase):
    def test_inside_and_outside(self):
        self.assertTrue(is_inside(Path("/lib/A/x.flac"), Path("/lib")))
        self.assertFalse(is_inside(Path("/library2/x.flac"), Path("/lib")))

    def test_symlink_into_library_is_inside(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "lib"
            library.mkdir()
            (library / "x.flac").write_bytes(b"")
            link = root / "link.flac"
            link.symlink_to(library / "x.flac")
            self.assertTrue(is_inside(link, library))


if __name__ == "__main__":
    unittest.main()
