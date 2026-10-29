# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from stemseparator import sumcheck
from stemseparator.naming import STEM_NAMES
from stemseparator.remote.assemble import separate_with_set
from stemseparator.remote.registry import MODEL_SETS, AveragedSet, SpecialistSet

_RNG = np.random.default_rng(7)
MIX = (0.3 * _RNG.standard_normal((2, 4096))).astype(np.float32)


class FakeModels:
    """Each fake model returns every stem as its own fixed fraction of the mix, and logs its runs."""

    def __init__(self, fractions: dict[str, dict[str, float]]):
        self.fractions = fractions
        self.runs: list[str] = []

    def __call__(self, model: str) -> dict[str, np.ndarray]:
        self.runs.append(model)
        return {stem: (share * MIX).astype(np.float32)
                for stem, share in self.fractions[model].items()}


def _index(stem: str) -> int:
    return STEM_NAMES.index(stem)


class SpecialistTest(unittest.TestCase):
    def setUp(self):
        self.models = FakeModels({
            "four": {"drums": 0.25, "bass": 0.125, "other": 0.5, "vocals": 0.0625},
            "voice": {"vocals": 0.375},
        })
        self.specialist = SpecialistSet(
            picks=(("vocals", "voice"), ("drums", "four"), ("bass", "four")), residual="other")

    def test_each_stem_comes_from_its_specialist(self):
        stems = separate_with_set(self.specialist, MIX, self.models)
        self.assertEqual(stems.shape, (4,) + MIX.shape)
        self.assertEqual(stems.dtype, np.float32)
        np.testing.assert_array_equal(stems[_index("vocals")], 0.375 * MIX)
        np.testing.assert_array_equal(stems[_index("drums")], 0.25 * MIX)
        np.testing.assert_array_equal(stems[_index("bass")], 0.125 * MIX)

    def test_other_is_the_residual_so_the_four_stems_sum_back_to_the_mix(self):
        stems = separate_with_set(self.specialist, MIX, self.models)
        np.testing.assert_allclose(stems[_index("other")], (1 - 0.375 - 0.25 - 0.125) * MIX,
                                   rtol=0, atol=1e-7)
        self.assertLess(sumcheck.residual_db(MIX.T, np.transpose(stems, (0, 2, 1))), -120.0)

    def test_residual_ignores_what_the_specialists_say_about_the_residual_stem(self):
        self.models.fractions["four"]["other"] = 99.0
        stems = separate_with_set(self.specialist, MIX, self.models)
        self.assertLess(float(np.abs(stems[_index("other")]).max()), float(np.abs(MIX).max()))

    def test_each_model_runs_once(self):
        separate_with_set(self.specialist, MIX, self.models)
        self.assertEqual(sorted(self.models.runs), ["four", "voice"])

    def test_a_specialist_lacking_its_stem_raises(self):
        del self.models.fractions["voice"]["vocals"]
        self.models.fractions["voice"]["other"] = 0.5
        with self.assertRaises(ValueError):
            separate_with_set(self.specialist, MIX, self.models)

    def test_a_result_of_the_wrong_shape_raises(self):
        def short_run(model):
            return {stem: np.zeros((2, 100), np.float32) for stem in STEM_NAMES}
        with self.assertRaises(ValueError):
            separate_with_set(self.specialist, MIX, short_run)


class AveragedTest(unittest.TestCase):
    def test_averaged_set_is_the_per_stem_mean(self):
        models = FakeModels({
            "a": {"drums": 0.5, "bass": 0.25, "other": 0.125, "vocals": 0.125},
            "b": {"drums": 0.25, "bass": 0.25, "other": 0.25, "vocals": 0.25},
        })
        stems = separate_with_set(AveragedSet(("a", "b")), MIX, models)
        np.testing.assert_allclose(stems[_index("drums")], 0.375 * MIX, atol=1e-7)
        np.testing.assert_allclose(stems[_index("vocals")], 0.1875 * MIX, atol=1e-7)
        self.assertEqual(models.runs, ["a", "b"])


class RegisteredSetsTest(unittest.TestCase):
    def test_both_registered_sets_assemble_with_fake_models(self):
        every_stem = {stem: 0.25 for stem in STEM_NAMES}
        for name, model_set in MODEL_SETS.items():
            models = FakeModels({model: every_stem for model in model_set.models})
            stems = separate_with_set(model_set, MIX, models)
            self.assertEqual(stems.shape, (4,) + MIX.shape, name)


if __name__ == "__main__":
    unittest.main()
