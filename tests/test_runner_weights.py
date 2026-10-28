# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stemseparator.remote import runner
from stemseparator.remote.registry import Checkpoint, ModelSpec


class WeightsTest(unittest.TestCase):
    """Downloads from file:// URLs, so nothing touches the network."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.weights_dir = root / "weights"
        source = root / "source"
        source.mkdir()
        self.checkpoint_file = source / "model.ckpt"
        self.checkpoint_file.write_bytes(b"weights")
        (source / "config.yaml").write_text("sample_rate: 44100\n")
        self.config_url = (source / "config.yaml").as_uri()
        patcher = mock.patch.object(runner, "WEIGHTS_DIR", str(self.weights_dir))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _spec(self, sha256=None):
        return ModelSpec("bs_roformer", self.config_url,
                         (Checkpoint(self.checkpoint_file.as_uri(), sha256),))

    def test_first_call_downloads_and_the_second_does_not(self):
        spec = self._spec()
        with mock.patch("builtins.print"):
            self.assertTrue(runner.download_missing(spec))
        self.assertFalse(runner.download_missing(spec))
        self.assertEqual(runner.ensure_weights(spec), [self.weights_dir / "model.ckpt"])
        self.assertEqual((self.weights_dir / "model.ckpt").read_bytes(), b"weights")
        self.assertTrue((self.weights_dir / "config.yaml").exists())

    def test_wrong_sha256_leaves_nothing_behind(self):
        with self.assertRaises(RuntimeError):
            runner.download_missing(self._spec(sha256="0" * 64))
        self.assertFalse((self.weights_dir / "model.ckpt").exists())
        self.assertEqual([p.name for p in self.weights_dir.iterdir()], ["config.yaml"])

    def test_matching_sha256_is_accepted_silently(self):
        digest = hashlib.sha256(b"weights").hexdigest()
        with mock.patch("builtins.print") as printed:
            runner.download_missing(ModelSpec("bs_roformer", "configs/x.yaml",
                                              (Checkpoint(self.checkpoint_file.as_uri(), digest),)))
        printed.assert_not_called()

    def test_download_uses_a_timeout(self):
        with mock.patch.object(runner.urllib.request, "urlopen",
                               side_effect=OSError("stalled")) as urlopen:
            with self.assertRaises(OSError):
                runner.download_missing(self._spec())
        self.assertEqual(urlopen.call_args.kwargs.get("timeout"), runner.DOWNLOAD_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
