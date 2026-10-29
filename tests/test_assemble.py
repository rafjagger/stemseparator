# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from stemseparator import sumcheck
from stemseparator.naming import SPACE4_NAMES, STEM_NAMES
from stemseparator.remote.assemble import separate_with_set
from stemseparator.remote.registry import MODEL_SETS, AveragedSet, SpatialSet, SpecialistSet

_RNG = np.random.default_rng(7)
MIX = (0.3 * _RNG.standard_normal((2, 4096))).astype(np.float32)


class FakeModels:
    """Each fake model returns every stem as its own fixed fraction of its input, and logs its runs."""

    def __init__(self, fractions: dict[str, dict[str, float]]):
        self.fractions = fractions
        self.runs: list[str] = []
        self.inputs: dict[str, np.ndarray] = {}

    def __call__(self, model: str, audio: np.ndarray) -> dict[str, np.ndarray]:
        self.runs.append(model)
        self.inputs[model] = audio
        return {stem: (share * audio).astype(np.float32)
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
        def short_run(model, audio):
            return {stem: np.zeros((2, 100), np.float32) for stem in STEM_NAMES}
        with self.assertRaises(ValueError):
            separate_with_set(self.specialist, MIX, short_run)


class SpatialTest(unittest.TestCase):
    """space4: LOW = bass + kick, TOPS = drums - kick, MUSIC = the mix minus the other three."""

    def setUp(self):
        self.models = FakeModels({
            "four": {"drums": 0.25, "bass": 0.125, "other": 0.5, "vocals": 0.0625},
            "voice": {"vocals": 0.375},
            "kit": {"kick": 0.5, "snare": 0.25, "hh": 0.25},
        })
        self.space4 = SpatialSet(rhythm="four", vocals="voice", drumsep="kit", kick_stem="kick")

    def _stems(self):
        return separate_with_set(self.space4, MIX, self.models)

    def _stem(self, stems, name):
        return stems[SPACE4_NAMES.index(name)]

    def test_names_are_low_tops_music_vocals(self):
        self.assertEqual(self.space4.stem_names, SPACE4_NAMES)
        self.assertEqual(self._stems().shape, (4,) + MIX.shape)

    def test_the_drum_model_splits_the_drums_stem_not_the_mix(self):
        self._stems()
        np.testing.assert_array_equal(self.models.inputs["kit"], 0.25 * MIX)

    def test_low_is_bass_plus_kick(self):
        np.testing.assert_allclose(self._stem(self._stems(), "low"), (0.125 + 0.5 * 0.25) * MIX,
                                   rtol=0, atol=1e-7)

    def test_tops_is_drums_minus_kick_so_the_drums_stay_whole(self):
        stems = self._stems()
        np.testing.assert_allclose(self._stem(stems, "tops"), 0.5 * 0.25 * MIX, rtol=0, atol=1e-7)
        np.testing.assert_allclose(self._stem(stems, "low") + self._stem(stems, "tops"),
                                   (0.25 + 0.125) * MIX, rtol=0, atol=1e-7)

    def test_vocals_come_from_the_vocal_model(self):
        np.testing.assert_array_equal(self._stem(self._stems(), "vocals"), 0.375 * MIX)

    def test_music_is_the_rest_so_the_four_sum_back_to_the_mix(self):
        stems = self._stems()
        np.testing.assert_allclose(self._stem(stems, "music"), (1 - 0.375 - 0.375) * MIX,
                                   rtol=0, atol=1e-7)
        self.assertLess(sumcheck.residual_db(MIX.T, np.transpose(stems, (0, 2, 1))), -120.0)

    def test_each_model_runs_once(self):
        self._stems()
        self.assertEqual(sorted(self.models.runs), ["four", "kit", "voice"])
        self.assertEqual(self.space4.models, ("four", "kit", "voice"))

    def test_a_drum_model_without_a_kick_raises(self):
        del self.models.fractions["kit"]["kick"]
        with self.assertRaises(ValueError):
            self._stems()

    def test_a_rhythm_model_without_bass_raises(self):
        del self.models.fractions["four"]["bass"]
        with self.assertRaises(ValueError):
            self._stems()


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
        every_stem = {stem: 0.25 for stem in STEM_NAMES + ("kick",)}
        for name, model_set in MODEL_SETS.items():
            models = FakeModels({model: every_stem for model in model_set.models})
            stems = separate_with_set(model_set, MIX, models)
            self.assertEqual(stems.shape, (4,) + MIX.shape, name)
            self.assertEqual(len(model_set.stem_names), 4, name)


if __name__ == "__main__":
    unittest.main()
