# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from unittest import mock

from stemseparator import cost


class CostTest(unittest.TestCase):
    def test_cost_known_and_unknown_gpu(self):
        self.assertAlmostEqual(cost.cost_usd(100.0, "NVIDIA A10"), 100.0 * 0.000306)
        self.assertAlmostEqual(cost.cost_usd(10.0, "NVIDIA L4"), 10.0 * 0.000222)
        self.assertIsNone(cost.cost_usd(10.0, "FAKE"))

    def test_longest_matching_key_wins(self):
        prices = {"A10": 1.0, "A100": 5.0}
        with mock.patch.object(cost, "GPU_PRICES_PER_SECOND", prices):
            self.assertEqual(cost.cost_usd(2.0, "NVIDIA A100"), 10.0)
            self.assertEqual(cost.cost_usd(2.0, "NVIDIA A10"), 2.0)

    def test_estimate_scales_with_models(self):
        one = cost.estimate_usd(100.0, ("m1",))
        two = cost.estimate_usd(100.0, ("m1", "m2"))
        self.assertAlmostEqual(one, 100.0 * cost.GPU_SECONDS_PER_AUDIO_SECOND * 0.000306)
        self.assertAlmostEqual(two, 2 * one)
