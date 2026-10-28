# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from stemseparator.naming import STEM_NAMES
from stemseparator.remote.ensemble import StemAccumulator, average_stems


def _result(value, channels=2, frames=8):
    return {name: np.full((channels, frames), value + index, dtype=np.float32)
            for index, name in enumerate(STEM_NAMES)}


class EnsembleTest(unittest.TestCase):
    def test_average_of_two_is_the_mean(self):
        averaged = average_stems([_result(0.0), _result(1.0)])
        self.assertEqual(averaged.shape, (4, 2, 8))
        self.assertEqual(averaged.dtype, np.float32)
        for index in range(4):
            np.testing.assert_allclose(averaged[index], index + 0.5)

    def test_average_of_one_is_that_result(self):
        averaged = average_stems([_result(2.0)])
        np.testing.assert_allclose(averaged[3], 5.0)

    def test_missing_stem_or_shape_mismatch_raises(self):
        incomplete = _result(0.0)
        del incomplete["bass"]
        with self.assertRaises(ValueError):
            average_stems([_result(0.0), incomplete])
        with self.assertRaises(ValueError):
            average_stems([_result(0.0), _result(0.0, frames=9)])
        with self.assertRaises(ValueError):
            average_stems([])


    def test_accumulator_sums_in_float32_and_divides_by_the_count(self):
        accumulator = StemAccumulator()
        for value in (0.0, 1.0, 2.0):
            accumulator.add(_result(value))
        averaged = accumulator.mean()
        self.assertEqual(averaged.dtype, np.float32)
        np.testing.assert_allclose(averaged[0], 1.0)

    def test_accumulator_rejects_a_mismatch_and_an_empty_mean(self):
        accumulator = StemAccumulator()
        with self.assertRaises(ValueError):
            accumulator.mean()
        accumulator.add(_result(0.0))
        with self.assertRaises(ValueError):
            accumulator.add(_result(0.0, channels=1))


if __name__ == "__main__":
    unittest.main()
