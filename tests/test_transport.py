# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from stemseparator.separator import SeparatorError
from stemseparator.transport import decode_result, decode_upload, encode_result, encode_upload
from tests.fakes import test_signal


class UploadTest(unittest.TestCase):
    def test_upload_roundtrip_24bit(self):
        audio = test_signal()
        data = encode_upload(audio, 44100)
        self.assertTrue(data.startswith(b"fLaC"))
        decoded, rate = decode_upload(data)
        self.assertEqual(rate, 44100)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.shape, audio.shape)
        self.assertLessEqual(float(np.max(np.abs(decoded - audio))), 2**-22)

    def test_upload_hot_signal_uses_float(self):
        audio = test_signal()
        audio[10, 0] = 1.2
        data = encode_upload(audio, 48000)
        self.assertTrue(data.startswith(b"RIFF"))
        decoded, rate = decode_upload(data)
        self.assertEqual(rate, 48000)
        self.assertEqual(float(np.max(np.abs(decoded))), np.float32(1.2))


class ResultTest(unittest.TestCase):
    def _stems(self):
        return np.stack([test_signal(0.1) * w for w in (0.4, 0.3, 0.2, 0.1)]).astype(np.float32)

    def test_result_roundtrip(self):
        stems = self._stems()
        result = encode_result(stems, 12.5, "NVIDIA A10", 44100, -42.5)
        separation = decode_result(result)
        np.testing.assert_array_equal(separation.stems, stems)
        self.assertEqual(separation.gpu_seconds, 12.5)
        self.assertEqual(separation.gpu, "NVIDIA A10")
        self.assertEqual(separation.sample_rate, 44100)
        self.assertEqual(separation.residual_db, -42.5)

    def test_result_roundtrip_carries_minus_infinity(self):
        result = encode_result(self._stems(), 1.0, "", 44100, -np.inf)
        self.assertEqual(decode_result(result).residual_db, -np.inf)

    def test_result_with_wrong_length_raises_separator_error(self):
        result = encode_result(self._stems(), 1.0, "", 44100, -40.0)
        result["stems"] = result["stems"][:-4]
        with self.assertRaises(SeparatorError) as caught:
            decode_result(result)
        self.assertFalse(caught.exception.transient)

    def test_result_with_wrong_stem_count_raises_separator_error(self):
        result = encode_result(self._stems(), 1.0, "", 44100, -40.0)
        result["shape"] = [3, *result["shape"][1:]]
        with self.assertRaises(SeparatorError):
            decode_result(result)

    def test_result_without_sample_rate_raises_separator_error(self):
        result = encode_result(self._stems(), 1.0, "", 44100, -40.0)
        del result["sample_rate"]
        with self.assertRaises(SeparatorError) as caught:
            decode_result(result)
        self.assertFalse(caught.exception.transient)
