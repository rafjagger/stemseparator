# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from stemseparator.sumcheck import residual_db
from tests.fakes import test_signal


class SumcheckTest(unittest.TestCase):
    def test_exact_sum_is_minus_infinity(self):
        mix = test_signal()
        stems = np.stack([0.5 * mix, 0.25 * mix, 0.125 * mix, 0.125 * mix])
        self.assertEqual(residual_db(mix, stems), -np.inf)

    def test_silent_mix_is_minus_infinity(self):
        mix = np.zeros((100, 2), dtype=np.float32)
        stems = np.ones((4, 100, 2), dtype=np.float32)
        self.assertEqual(residual_db(mix, stems), -np.inf)

    def test_known_residual(self):
        mix = test_signal()
        stems = np.stack([0.9 * mix, 0 * mix, 0 * mix, 0 * mix])
        self.assertAlmostEqual(residual_db(mix, stems), -20.0, places=2)

    def test_long_signal_spanning_several_chunks(self):
        mix = test_signal(seconds=30.0)
        stems = np.stack([0.9 * mix, 0 * mix, 0 * mix, 0 * mix])
        self.assertAlmostEqual(residual_db(mix, stems), -20.0, places=2)
