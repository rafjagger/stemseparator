# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import tempfile
import unittest
from pathlib import Path

from stemseparator.layout import default_folder, is_inside, output_folders, plan_track
from stemseparator.naming import SPACE4_NAMES


class PlanTrackTest(unittest.TestCase):
    def test_plan_paths(self):
        plan = plan_track(Path("/lib"), Path("AC/DC"), "Intro", Path("/in/x.FLAC"),
                          exists=lambda p: False)
        album = Path("/lib/AC/DC")
        self.assertEqual(plan.album_dir, album)
        self.assertEqual(plan.title, "Intro")
        self.assertEqual(plan.stem_paths[0], album / "Intro - 1 - drums.flac")
        self.assertEqual(plan.stem_paths[3], album / "Intro - 4 - vocals.flac")
        self.assertEqual(plan.original_path, album / "originals" / "Intro.FLAC")

    def test_plan_paths_carry_the_sets_stem_names(self):
        plan = plan_track(Path("/lib"), Path("sets/house"), "Intro", Path("/in/x.flac"),
                          exists=lambda p: False, stem_names=SPACE4_NAMES)
        self.assertEqual([p.name for p in plan.stem_paths],
                         ["Intro - 1 - low.flac", "Intro - 2 - tops.flac",
                          "Intro - 3 - music.flac", "Intro - 4 - vocals.flac"])

    def test_wav_sets_end_in_wav(self):
        plan = plan_track(Path("/lib"), Path("sets/house"), "Intro", Path("/in/x.wav"),
                          exists=lambda p: False, extension="wav")
        self.assertEqual(plan.stem_paths[0].name, "Intro - 1 - drums.wav")
        self.assertEqual(plan.original_path.name, "Intro.wav")

    def test_empty_names_use_fallbacks(self):
        plan = plan_track(Path("/lib"), Path("..."), "", Path("/in/x.wav"),
                          exists=lambda p: False)
        self.assertEqual(plan.album_dir, Path("/lib/_"))
        self.assertEqual(plan.title, "Untitled")

    def test_clash_gets_2_then_3(self):
        album = Path("/lib/sets/house")
        taken = {album / "Intro - 1 - drums.flac", album / "originals" / "Intro (2).flac"}
        plan = plan_track(Path("/lib"), Path("sets/house"), "Intro", Path("/in/x.flac"),
                          exists=lambda p: p in taken)
        self.assertEqual(plan.title, "Intro (3)")

    def test_clash_on_original_alone_counts(self):
        album = Path("/lib/sets/house")
        taken = {album / "originals" / "Intro.flac"}
        plan = plan_track(Path("/lib"), Path("sets/house"), "Intro", Path("/in/x.flac"),
                          exists=lambda p: p in taken)
        self.assertEqual(plan.title, "Intro (2)")

    def test_clash_on_any_stem_counts(self):
        album = Path("/lib/sets/house")
        taken = {album / "Intro - 4 - vocals.flac"}
        plan = plan_track(Path("/lib"), Path("sets/house"), "Intro", Path("/in/x.flac"),
                          exists=lambda p: p in taken)
        self.assertEqual(plan.title, "Intro (2)")


class OutputFoldersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        (self.root / "sets" / "house").mkdir(parents=True)

    def test_a_recursive_run_mirrors_the_path_below_the_given_folder(self):
        top, deep = self.root / "sets" / "a.flac", self.root / "sets" / "house" / "b.flac"
        folders = output_folders([self.root / "sets"], [top, deep])
        self.assertEqual(folders[top], Path("sets"))
        self.assertEqual(folders[deep], Path("sets/house"))

    def test_a_file_given_directly_uses_its_parent_folders_name(self):
        single = self.root / "sets" / "house" / "c.flac"
        self.assertEqual(output_folders([single], [single])[single], Path("house"))
        self.assertEqual(default_folder(single), Path("house"))


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
