# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import tempfile
import unittest
from pathlib import Path

from stemseparator.config import Config, ConfigError, load_config


class LoadConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "config.toml"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def test_missing_file_gives_defaults(self):
        config = load_config(self.path)
        self.assertEqual(config, Config())
        self.assertEqual(config.library, Path("~/Music/stems").expanduser())
        self.assertEqual(config.models, "space4")
        self.assertEqual(config.parallel, 4)

    def test_library_expands_tilde(self):
        self.write('library = "~/x"\n')
        self.assertEqual(load_config(self.path).library, Path("~/x").expanduser())

    def test_models_names_a_model_set(self):
        self.write('models = "ensemble4"\n')
        self.assertEqual(load_config(self.path).models, "ensemble4")

    def test_a_list_of_models_is_refused_with_the_new_form(self):
        self.write('models = ["scnet_xl_ihf", "bs_roformer_ep17"]\n')
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.path)
        self.assertIn('models = "space4"', str(ctx.exception))

    def test_parallel_is_read(self):
        self.write("parallel = 2\n")
        self.assertEqual(load_config(self.path).parallel, 2)

    def test_unknown_key_names_the_key(self):
        self.write('libary = "~/x"\n')
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.path)
        self.assertIn("libary", str(ctx.exception))

    def test_parallel_zero_is_refused(self):
        self.write("parallel = 0\n")
        with self.assertRaises(ConfigError):
            load_config(self.path)

    def test_wrong_types_are_refused(self):
        for text in ('parallel = "4"\n', 'models = ""\n', "library = 3\n",
                     "models = 1\n", "parallel = true\n"):
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaises(ConfigError):
                    load_config(self.path)

    def test_broken_toml_names_the_path(self):
        self.write("library = \n[")
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.path)
        self.assertIn(str(self.path), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
