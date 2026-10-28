# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import re
import unittest

from stemseparator.naming import STEM_NAMES
from stemseparator.remote.registry import DEFAULT_MODELS, MODELS, MSST_COMMIT, MSST_REPO


def _urls(spec):
    urls = [checkpoint.url for checkpoint in spec.checkpoints]
    if "://" in spec.config:
        urls.append(spec.config)
    return urls


class RegistryTest(unittest.TestCase):
    def test_default_models_are_registered_and_pinned_to_the_commit(self):
        self.assertTrue(set(DEFAULT_MODELS) <= set(MODELS))
        self.assertRegex(MSST_COMMIT, r"^[0-9a-f]{40}$")
        self.assertTrue(MSST_REPO.startswith("https://"))
        for name, spec in MODELS.items():
            self.assertTrue(spec.checkpoints, name)
            for url in _urls(spec):
                self.assertTrue(url.startswith("https://"), f"{name}: {url}")

    def test_htdemucs_ft_has_one_checkpoint_per_stem(self):
        spec = MODELS["htdemucs_ft"]
        self.assertEqual(tuple(c.stem for c in spec.checkpoints), STEM_NAMES)
        self.assertFalse("://" in spec.config)

    def test_sha256_pins_are_hex_when_set(self):
        for spec in MODELS.values():
            for checkpoint in spec.checkpoints:
                if checkpoint.sha256 is not None:
                    self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", checkpoint.sha256))


if __name__ == "__main__":
    unittest.main()
