# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from stemseparator.separator import SeparatorError, SeparatorUnavailable
from tests.fakes import FakeSeparator, test_signal


class FakeSeparatorTest(unittest.TestCase):
    def test_fake_stems_sum_to_input(self):
        audio = test_signal()
        separation = FakeSeparator().separate(audio, 44100, ("htdemucs",))
        self.assertEqual(separation.stems.shape, (4, *audio.shape))
        np.testing.assert_allclose(separation.stems.sum(axis=0), audio, atol=1e-6)
        self.assertEqual(separation.sample_rate, 44100)
        self.assertLess(separation.residual_db, -100)

    def test_fake_fails_then_succeeds(self):
        fake = FakeSeparator(fail_times=2)
        audio = test_signal(0.1)
        for _ in range(2):
            with self.assertRaises(SeparatorError) as caught:
                fake.separate(audio, 44100, ("m",))
            self.assertTrue(caught.exception.transient)
        fake.separate(audio, 44100, ("m",))
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(fake.calls[0], (4410, ("m",)))

    def test_fake_resamples_48k_to_44k(self):
        audio = test_signal(1.0, rate=48000)
        separation = FakeSeparator().separate(audio, 48000, ("m",))
        self.assertEqual(separation.stems.shape[1], 44100)
        self.assertEqual(separation.sample_rate, 44100)

    def test_fake_preflight_raises_unavailable(self):
        fake = FakeSeparator(unavailable=SeparatorUnavailable("no token"))
        with self.assertRaises(SeparatorUnavailable):
            fake.preflight()
        FakeSeparator().preflight()
