# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from stemseparator import sumcheck, transport
from stemseparator.modal_separator import APP_NAME, ModalSeparator, classify
from stemseparator.separator import SeparatorError, SeparatorUnavailable
from tests import fakes


def _exception_class(name):
    return type(name, (Exception,), {})


NotFoundError = _exception_class("NotFoundError")
AuthError = _exception_class("AuthError")
ResourceExhaustedError = _exception_class("ResourceExhaustedError")
InternalFailure = _exception_class("InternalFailure")
ServiceError = _exception_class("ServiceError")
InvalidError = _exception_class("InvalidError")
ModalConnectionError = type("ConnectionError", (Exception,), {})
ModalTimeoutError = type("TimeoutError", (Exception,), {})
RemoteError = _exception_class("RemoteError")


class _Remote:
    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        return self._behaviour(*args)


def _raises(error):
    def behaviour(*args):
        raise error
    return behaviour


def _quarter_split(upload, model_set):
    mix, rate = transport.decode_upload(upload)
    stems = np.stack([0.25 * mix] * 4).astype(np.float32)
    return transport.encode_result(stems, 12.5, "NVIDIA A10", 44100,
                                   sumcheck.residual_db(mix, stems))


def _refuse_lookup(app_name, function_name):
    raise AssertionError("lookup must not be called")


class _Lookup:
    def __init__(self, functions):
        self.functions = functions
        self.names = []

    def __call__(self, app_name, function_name):
        self.names.append((app_name, function_name))
        return self.functions[function_name]


class PreflightTest(unittest.TestCase):
    def test_preflight_without_token_never_looks_up(self):
        separator = ModalSeparator(lookup=_refuse_lookup, token_present=lambda: False)
        with self.assertRaises(SeparatorUnavailable) as caught:
            separator.preflight()
        self.assertIn("modal token new", str(caught.exception))

    def test_preflight_pings_the_deployed_app(self):
        ping = _Remote(lambda: "ok")
        lookup = _Lookup({"ping": ping})
        ModalSeparator(lookup=lookup, token_present=lambda: True).preflight()
        self.assertEqual(lookup.names, [(APP_NAME, "ping")])
        self.assertEqual(ping.calls, [()])

    def test_preflight_maps_not_found_and_budget(self):
        cases = [
            (NotFoundError("App 'stemseparator' not found"), "modal deploy modal_app.py"),
            (ResourceExhaustedError("limit"), "budget or credit"),
            (InvalidError("workspace billing limit reached"), "budget or credit"),
            (AuthError("bad token"), "rejected the token"),
        ]
        for error, expected in cases:
            with self.subTest(type(error).__name__):
                lookup = _Lookup({"ping": _Remote(_raises(error))})
                separator = ModalSeparator(lookup=lookup, token_present=lambda: True)
                with self.assertRaises(SeparatorUnavailable) as caught:
                    separator.preflight()
                self.assertIn(expected, str(caught.exception))

    def test_preflight_not_found_at_lookup_is_mapped_too(self):
        def lookup(app_name, function_name):
            raise NotFoundError("no such app")
        separator = ModalSeparator(lookup=lookup, token_present=lambda: True)
        with self.assertRaises(SeparatorUnavailable) as caught:
            separator.preflight()
        self.assertIn("not deployed", str(caught.exception))

    def test_preflight_other_error_is_unavailable_with_its_text(self):
        lookup = _Lookup({"ping": _Remote(_raises(ModalConnectionError("network down")))})
        separator = ModalSeparator(lookup=lookup, token_present=lambda: True)
        with self.assertRaises(SeparatorUnavailable) as caught:
            separator.preflight()
        self.assertIn("network down", str(caught.exception))


class SeparateTest(unittest.TestCase):
    def test_separate_roundtrip_through_fake_remote(self):
        remote = _Remote(_quarter_split)
        lookup = _Lookup({"separate": remote})
        separator = ModalSeparator(lookup=lookup, token_present=lambda: True)
        mix = fakes.test_signal(0.5)
        result = separator.separate(mix, 44100, "ensemble4")
        self.assertEqual(lookup.names, [(APP_NAME, "separate")])
        upload, model_set = remote.calls[0]
        self.assertIsInstance(upload, bytes)
        self.assertEqual(model_set, "ensemble4")
        self.assertEqual(result.stems.shape, (4,) + mix.shape)
        self.assertEqual(result.sample_rate, 44100)
        self.assertEqual(result.gpu, "NVIDIA A10")
        self.assertEqual(result.gpu_seconds, 12.5)
        np.testing.assert_allclose(result.stems.sum(axis=0), mix, atol=1e-5)
        self.assertLess(result.residual_db, -60.0)

    def test_separate_maps_remote_errors(self):
        lookup = _Lookup({"separate": _Remote(_raises(ModalTimeoutError("slow")))})
        separator = ModalSeparator(lookup=lookup, token_present=lambda: True)
        with self.assertRaises(SeparatorError) as caught:
            separator.separate(fakes.test_signal(0.1), 44100, "specialist")
        self.assertTrue(caught.exception.transient)

    def test_separate_lets_a_malformed_result_fail_permanently(self):
        lookup = _Lookup({"separate": _Remote(lambda upload, model_set: {"shape": [1]})})
        separator = ModalSeparator(lookup=lookup, token_present=lambda: True)
        with self.assertRaises(SeparatorError) as caught:
            separator.separate(fakes.test_signal(0.1), 44100, "specialist")
        self.assertFalse(caught.exception.transient)

    def test_separate_turns_an_upload_that_cannot_be_encoded_into_a_permanent_error(self):
        separator = ModalSeparator(lookup=_refuse_lookup, token_present=lambda: True)
        with mock.patch.object(transport, "encode_upload", side_effect=RuntimeError("no space")):
            with self.assertRaises(SeparatorError) as caught:
                separator.separate(fakes.test_signal(0.1), 44100, "specialist")
        self.assertFalse(caught.exception.transient)
        self.assertIn("no space", str(caught.exception))


class ClassifyTest(unittest.TestCase):
    def test_classify_transient_vs_permanent(self):
        for error in (ModalConnectionError("x"), ModalTimeoutError("x"),
                      InternalFailure("x"), ServiceError("x")):
            with self.subTest(type(error).__name__):
                mapped = classify(error)
                self.assertIsInstance(mapped, SeparatorError)
                self.assertTrue(mapped.transient)
        for error in (InvalidError("x"), RemoteError("x"), NotFoundError("x"),
                      ValueError("x")):
            with self.subTest(type(error).__name__):
                self.assertFalse(classify(error).transient)

    def test_classify_keeps_the_message(self):
        self.assertIn("socket closed", str(classify(ModalConnectionError("socket closed"))))


class ImportTest(unittest.TestCase):
    def test_importing_the_package_does_not_import_modal_or_torch(self):
        code = ("import stemseparator.cli, stemseparator.modal_separator, "
                "stemseparator.remote.registry, stemseparator.remote.ensemble, sys; "
                "print('modal' in sys.modules or 'torch' in sys.modules)")
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                   text=True, check=True)
        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
