# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

from stemseparator.naming import STEM_NAMES, sanitise_name, stem_file_name
from tests.stemdeck_reference import split_stem_name


class NamingTest(unittest.TestCase):
    def test_stem_order_is_bus_order(self):
        self.assertEqual(STEM_NAMES, ("drums", "bass", "other", "vocals"))

    def test_stem_file_names(self):
        self.assertEqual(stem_file_name("Archangel", 0), "Archangel - 1 - drums.wav")
        self.assertEqual(stem_file_name("Archangel", 3), "Archangel - 4 - vocals.wav")

    def test_sanitise(self):
        cases = [
            ("AC/DC", "x", "AC_DC"), ("..hidden", "x", "hidden"),
            ("Title - ", "x", "Title"), ("  Mix_ ", "x", "Mix"),
            ("", "Unknown Artist", "Unknown Artist"),
            ("///", "Unknown Album", "Unknown Album"),
            ("Sigur Rós", "x", "Sigur Rós"), ("a\0b", "x", "a_b"),
            ("a\\b", "x", "a_b"),
        ]
        for raw, fallback, want in cases:
            with self.subTest(raw=raw):
                self.assertEqual(sanitise_name(raw, fallback), want)

    def test_reference_port_matches_stemdeck_tests(self):
        self.assertEqual(split_stem_name("Archangel - 1.drums"), ("Archangel", "1.drums"))
        self.assertEqual(split_stem_name("Set - 01 - 02"), ("Set - 01", "02"))
        self.assertIsNone(split_stem_name("NoSeparator"))
        p1, _ = split_stem_name("Title - 1 drums")
        p2, _ = split_stem_name("Title - 2 bass")
        self.assertNotEqual(p1, p2)

    def test_tricky_titles_make_one_set_in_bus_order(self):
        for raw in ["Archangel", "Track 2", "Set - 01", "1", "Foo - 1 -", "", "Mix_", "A - B - C"]:
            with self.subTest(raw=raw):
                title = sanitise_name(raw, "Untitled")
                splits = [
                    split_stem_name(stem_file_name(title, i).removesuffix(".wav"))
                    for i in range(4)
                ]
                self.assertEqual({prefix for prefix, _ in splits}, {title})
                suffixes = [suffix for _, suffix in splits]
                self.assertEqual(suffixes, sorted(suffixes))


if __name__ == "__main__":
    unittest.main()
