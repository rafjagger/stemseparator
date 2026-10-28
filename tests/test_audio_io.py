# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import tempfile
import unittest
from pathlib import Path

import mutagen.id3 as id3
import numpy as np
import soundfile as sf

from stemseparator.audio_io import (
    MAX_24BIT,
    UndecodableError,
    read_seconds,
    read_track,
    unsupported_format_message,
    write_stem_24bit,
)
from tests import fakes


class AudioIoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write(self, name, data, rate=44100, **kwargs):
        path = self.dir / name
        sf.write(path, data, rate, **kwargs)
        return path

    def test_reads_flac_with_tags(self):
        path = self.dir / "a.flac"
        with sf.SoundFile(path, "w", 44100, 2, format="FLAC") as f:
            f.title, f.artist, f.album = "T", "A", "B"
            f.write(fakes.test_signal(1.0))
        track = read_track(path)
        self.assertEqual((track.tags.title, track.tags.artist, track.tags.album), ("T", "A", "B"))
        self.assertEqual(track.audio.shape, (44100, 2))
        self.assertEqual(track.audio.dtype, np.float32)
        self.assertAlmostEqual(track.seconds, 1.0)

    def test_mono_becomes_stereo(self):
        track = read_track(self._write("m.wav", fakes.test_signal(0.2, channels=1)))
        self.assertEqual(track.audio.shape[1], 2)
        np.testing.assert_array_equal(track.audio[:, 0], track.audio[:, 1])

    def test_48k_rate_kept(self):
        track = read_track(self._write("r.wav", fakes.test_signal(0.2, 48000), 48000))
        self.assertEqual(track.sample_rate, 48000)

    def test_reads_aiff(self):
        track = read_track(self._write("a.aiff", fakes.test_signal(0.2)))
        self.assertEqual(track.audio.shape, (8820, 2))

    def test_read_seconds_comes_from_the_header(self):
        path = self._write("r.wav", fakes.test_signal(0.5, 48000), 48000)
        self.assertAlmostEqual(read_seconds(path), 0.5)

    def test_read_seconds_refuses_like_read_track(self):
        mp3 = self.dir / "x.mp3"
        mp3.write_bytes(b"ID3xxxx")
        with self.assertRaisesRegex(UndecodableError, "MP3 is not supported"):
            read_seconds(mp3)

    def test_three_channels_refused(self):
        path = self._write("c.wav", fakes.test_signal(0.2, channels=3))
        with self.assertRaisesRegex(UndecodableError, "3 channels"):
            read_track(path)

    def test_garbage_refused(self):
        path = self.dir / "g.flac"
        path.write_bytes(np.random.default_rng(1).bytes(2000))
        with self.assertRaises(UndecodableError):
            read_track(path)

    def test_empty_audio_refused(self):
        path = self._write("e.wav", np.zeros((0, 2), dtype=np.float32))
        with self.assertRaisesRegex(UndecodableError, "no audio"):
            read_track(path)

    def test_mp3_suffix_refused_with_clear_message(self):
        path = self.dir / "x.mp3"
        path.write_bytes(b"whatever")
        with self.assertRaises(UndecodableError) as ctx:
            read_track(path)
        self.assertIn("MP3", str(ctx.exception))
        self.assertIn("FLAC, WAV or AIFF", str(ctx.exception))

    def test_suffix_check_is_case_insensitive(self):
        track = read_track(self._write("U.WAV", fakes.test_signal(0.1)))
        self.assertEqual(track.audio.shape[1], 2)

    def test_ogg_inside_a_flac_name_refused(self):
        path = self.dir / "o.flac"
        sf.write(path, fakes.test_signal(0.2), 44100, format="OGG", subtype="VORBIS")
        with self.assertRaisesRegex(UndecodableError, "OGG"):
            read_track(path)

    def test_message_builder(self):
        self.assertEqual(
            unsupported_format_message("OGG"), "OGG is not supported: use FLAC, WAV or AIFF"
        )

    def test_reads_id3_tags_from_wav(self):
        path = self._write("i.wav", fakes.test_signal(0.2))
        from mutagen.wave import WAVE

        wave = WAVE(path)
        wave.add_tags()
        tags = wave.tags
        tags.add(id3.TIT2(encoding=3, text="Song"))
        tags.add(id3.TPE1(encoding=3, text="Artist"))
        tags.add(id3.TALB(encoding=3, text="Album"))
        tags.add(id3.TPE2(encoding=3, text="Various"))
        wave.save()
        got = read_track(path).tags
        self.assertEqual(
            (got.title, got.artist, got.album, got.albumartist),
            ("Song", "Artist", "Album", "Various"),
        )

    def test_reads_albumartist_from_flac(self):
        from mutagen.flac import FLAC

        path = self._write("f.flac", fakes.test_signal(0.2))
        flac = FLAC(path)
        flac["albumartist"] = "Group"
        flac.save()
        self.assertEqual(read_track(path).tags.albumartist, "Group")

    def test_untagged_file_has_empty_tags(self):
        tags = read_track(self._write("n.wav", fakes.test_signal(0.1))).tags
        self.assertEqual((tags.title, tags.artist, tags.album, tags.albumartist), ("", "", "", ""))

    def test_clamp_not_wrap(self):
        path = self.dir / "out.wav"
        audio = np.array([[1.5, -1.5], [0.25, 0.25]], dtype=np.float32)
        self.assertEqual(write_stem_24bit(path, audio, 44100), 2)
        self.assertEqual(sf.info(path).subtype, "PCM_24")
        raw, _ = sf.read(path, dtype="int32", always_2d=True)
        self.assertEqual(raw[0, 0], 2**31 - 256)
        self.assertEqual(raw[0, 1], -(2**31))
        self.assertAlmostEqual(raw[1, 0] / 2**31, 0.25, places=6)

    def test_max_24bit_constant(self):
        self.assertEqual(MAX_24BIT, 1.0 - 2.0**-23)


if __name__ == "__main__":
    unittest.main()
