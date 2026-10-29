# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import re
import unittest

from stemseparator.naming import SPACE4_NAMES, STEM_NAMES
from stemseparator.remote.registry import (
    DEFAULT_MODEL_SET, MODEL_SETS, MODELS, MSST_COMMIT, MSST_REPO, AveragedSet, SpatialSet,
    SpecialistSet)


def _urls(spec):
    urls = [checkpoint.url for checkpoint in spec.checkpoints]
    if "://" in spec.config:
        urls.append(spec.config)
    return urls


class ManifestTest(unittest.TestCase):
    def test_msst_is_pinned_to_a_commit(self):
        self.assertRegex(MSST_COMMIT, r"^[0-9a-f]{40}$")
        self.assertTrue(MSST_REPO.startswith("https://"))

    def test_every_model_has_https_urls_a_licence_and_its_published_numbers(self):
        for name, spec in MODELS.items():
            self.assertTrue(spec.checkpoints, name)
            for url in _urls(spec):
                self.assertTrue(url.startswith("https://"), f"{name}: {url}")
            self.assertTrue(spec.licence.strip(), f"{name}: licence missing")
            self.assertTrue(spec.reported.strip(), f"{name}: published numbers missing")

    def test_sha256_is_a_placeholder_or_a_hex_digest(self):
        for name, spec in MODELS.items():
            for checkpoint in spec.checkpoints:
                if checkpoint.sha256 is not None:
                    self.assertRegex(checkpoint.sha256, r"^[0-9a-f]{64}$", name)

    def test_checkpoint_file_names_do_not_collide_in_the_weights_volume(self):
        names = [url.rsplit("/", 1)[-1] for spec in MODELS.values() for url in _urls(spec)]
        self.assertEqual(len(names), len(set(names)))

    def test_htdemucs_ft_has_one_checkpoint_per_stem(self):
        spec = MODELS["htdemucs_ft"]
        self.assertEqual(tuple(c.stem for c in spec.checkpoints), STEM_NAMES)
        self.assertFalse("://" in spec.config)


class ModelSetTest(unittest.TestCase):
    def test_space4_is_the_default_and_specialist_and_ensemble4_stay_selectable(self):
        self.assertEqual(DEFAULT_MODEL_SET, "space4")
        self.assertIsInstance(MODEL_SETS["specialist"], SpecialistSet)
        self.assertEqual(MODEL_SETS["ensemble4"],
                         AveragedSet(("scnet_masked_xl_ihf", "bs_roformer_ep17")))

    def test_every_set_names_only_registered_models(self):
        for name, model_set in MODEL_SETS.items():
            self.assertTrue(model_set.models, name)
            self.assertTrue(set(model_set.models) <= set(MODELS), name)

    def test_specialist_picks_vocals_drums_bass_and_leaves_other_as_the_residual(self):
        specialist = MODEL_SETS["specialist"]
        self.assertEqual(specialist.residual, "other")
        self.assertEqual(dict(specialist.picks), {"vocals": "mel_roformer_kj_vocals",
                                                  "drums": "scnet_xl_ihf",
                                                  "bass": "scnet_xl_ihf"})
        self.assertEqual(specialist.models, ("scnet_xl_ihf", "mel_roformer_kj_vocals"))

    def test_space4_takes_drums_and_bass_from_scnet_vocals_from_kj_and_splits_the_drums(self):
        self.assertEqual(MODEL_SETS["space4"],
                         SpatialSet(rhythm="scnet_xl_ihf", vocals="mel_roformer_kj_vocals",
                                    drumsep="mdx23c_drumsep_6stem", kick_stem="kick"))
        self.assertEqual(MODEL_SETS["space4"].stem_names, SPACE4_NAMES)

    def test_every_drum_model_checkpoint_is_hash_pinned(self):
        for name, model_set in MODEL_SETS.items():
            if isinstance(model_set, SpatialSet):
                for checkpoint in MODELS[model_set.drumsep].checkpoints:
                    self.assertIsNotNone(checkpoint.sha256, name)

    def test_the_vocal_shootout_sets_swap_only_the_vocal_model_of_space4(self):
        space4 = MODEL_SETS["space4"]
        for name, vocals in (("space4polar", "bs_polarformer_vocals"),
                             ("space4deux", "mel_roformer_deux_vocals")):
            with self.subTest(name=name):
                self.assertEqual(MODEL_SETS[name],
                                 SpatialSet(rhythm=space4.rhythm, vocals=vocals,
                                            drumsep=space4.drumsep, kick_stem=space4.kick_stem))

    def test_the_new_vocal_models_are_hash_pinned(self):
        for name in ("bs_polarformer_vocals", "mel_roformer_deux_vocals"):
            for checkpoint in MODELS[name].checkpoints:
                self.assertIsNotNone(checkpoint.sha256, name)

    def test_sw4_maps_the_six_stem_model_to_four_with_other_as_the_rest(self):
        sw4 = MODEL_SETS["sw4"]
        self.assertEqual(dict(sw4.picks), {"drums": "bs_roformer_sw", "bass": "bs_roformer_sw",
                                           "vocals": "bs_roformer_sw"})
        self.assertEqual(sw4.residual, "other")
        self.assertEqual(MODEL_SETS["space4sw"].rhythm, "bs_roformer_sw")

    def test_third_party_files_are_pinned_to_a_revision(self):
        for name, spec in MODELS.items():
            for url in _urls(spec):
                if "huggingface.co" in url or "MSST-WebUI" in url:
                    self.assertRegex(url, r"/[0-9a-f]{40}/", f"{name}: {url}")

    def test_a_specialist_set_must_cover_every_stem_exactly_once(self):
        with self.assertRaises(ValueError):
            SpecialistSet(picks=(("vocals", "a"), ("drums", "b")), residual="other")
        with self.assertRaises(ValueError):
            SpecialistSet(picks=(("vocals", "a"), ("drums", "b"), ("bass", "b"),
                                 ("other", "c")), residual="other")
        with self.assertRaises(ValueError):
            SpecialistSet(picks=(("vocals", "a"), ("vocals", "b"), ("bass", "b")),
                          residual="other")


if __name__ == "__main__":
    unittest.main()
