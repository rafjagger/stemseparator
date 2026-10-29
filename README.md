# stemseparator

A local command-line tool that sends audio (FLAC, WAV or AIFF) to a rented cloud GPU, separates it into four stems and writes the result as a stem set for StemDeck (44.1 kHz, 24-bit).

The separation runs on [Modal](https://modal.com) with models loaded through
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
(MSST, pinned to one commit). Two model sets are selectable with `--models`:

- **`specialist`** (default): vocals from Kimberley Jensen's Mel-Band RoFormer, drums and bass
  from SCNet XL IHF, and other = the mix minus those three, so the four stems sum back to the mix.
- **`ensemble4`**: SCNet Masked XL IHF and BS-RoFormer, all four stems averaged (the earlier
  default).

`htdemucs_ft` (Demucs v4, one fine-tuned checkpoint per stem) is there as a reference.

Published SDR in dB on MVSep's Multisong set, from MSST's `docs/pretrained_models.md`:

| Model | drums | bass | other | vocals |
|---|---|---|---|---|
| SCNet XL IHF (specialist: drums, bass) | 11.58 | 11.94 | 6.49 | 9.69 |
| Mel-Band RoFormer, Kimberley Jensen (specialist: vocals) | | | | 10.98 |
| SCNet Masked XL IHF (ensemble4) | 11.38 | 11.57 | 6.30 | 9.43 |
| BS-RoFormer ep 17 (ensemble4) | 11.29 | 11.08 | 5.96 | 9.19 |

No figures are published for the averaged ensemble or for the specialist's residual "other".

## Setup

Python 3.12 or newer (numpy 2.5 needs it), and a Modal account.

```
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/modal token new          # once; stores the token in ~/.modal.toml
.venv/bin/modal deploy modal_app.py
```

`modal deploy` builds the GPU image (PyTorch, MSST) on Modal's side the first time, which takes a
few minutes. Deploy again after changing `modal_app.py` or anything under `src/stemseparator/`.
The model weights are downloaded into the Modal Volume `stemseparator-weights` on the first
separation and reused from there.

## Use

The command lives in the venv: run `source .venv/bin/activate` once per terminal, or call it as
`.venv/bin/stemseparator`.

```
stemseparator ~/Music/new/                    # the audio files directly in the folder
stemseparator track.flac --artist X --album Y
stemseparator --models ensemble4 track.flac   # another model set than the config's
stemseparator --dry-run …                     # names and cost estimate, no upload
```

A folder is read one level deep: files in its subfolders are not picked up. The same file given
twice (or a folder plus a file inside it) is separated once. Ctrl-C stops starting new tracks and
waits for the running ones, whose GPU time is already paid; a second Ctrl-C abandons them, and
they write nothing.

Further flags: `--title` (single track only), `--library` (overrides the config), `--long` (allow
tracks over 20 minutes). Before the first upload the tool checks that a token exists and that the
app is deployed, and stops with a plain message if not. A failed network or Modal call is retried
twice; a track that still fails leaves nothing in the library.

- **Input:** FLAC, WAV or AIFF only, mono or stereo, any sample rate.
- **Output:** always 44.1 kHz, 24-bit WAV (the models run at 44.1 kHz; a 48 kHz track comes back
  at 44.1 kHz). Per track: `<library>/<Artist>/<Album>/<Title> - 1 - drums.wav` … `4 - vocals.wav`,
  plus the untouched source under `originals/`.
- GPU seconds and cost per track are approximate: wall time on the GPU times Modal's list price;
  cold starts and idle time are not in it.
- The `--dry-run` cost is a rough guess: it assumes 0.3 GPU seconds per second of audio and
  checkpoint run (two for `specialist` and `ensemble4`, four for `htdemucs_ft`), a placeholder
  until the first real runs have been measured.

## Config

`~/.config/stemseparator/config.toml`, every key optional:

```toml
library = "~/Music/stems"                       # where stem sets go
models = "specialist"                           # model set: specialist, ensemble4, htdemucs_ft
parallel = 4                                    # tracks on the GPU at once
```

## Privacy

The track is sent to Modal for the length of one call; this tool stores no audio there and logs
no file names. Modal routes payloads above 2 MiB — every track — through its own blob storage in
the US, and its retention for that storage is not stated on its data-residency page. Read Modal's
current statement on this before uploading material that must not leave your hands.

## Model weights

The weights are downloaded from their public release pages at run time and are never committed to
this repository. Where the host publishes a sha256 (GitHub release assets, Hugging Face LFS), the
download is checked against it. Licences as published:

| Weights | Source | Licence as published |
|---|---|---|
| SCNet XL IHF, SCNet Masked XL IHF, BS-RoFormer ep 17 | MSST GitHub releases | none stated for the weights (MSST's code is MIT); trained on MUSDB18-HQ, licensed for research and non-commercial use |
| Mel-Band RoFormer vocals (Kimberley Jensen) | Hugging Face `KimberleyJSN/melbandroformer` | MIT on the model card; community weights, training data not stated |
| htdemucs_ft | Meta's Demucs release | Demucs repository MIT; none stated separately for the weights |

These are community and research weights with unclear terms for commercial use. That is fine for
a personal DJ library; check each model's terms before using its stems commercially.

## Development

```
.venv/bin/python -m unittest discover -s tests -t . -v
```

No test touches the network, Modal, a GPU or torch. `modal_app.py` and
`src/stemseparator/remote/runner.py` run only inside the Modal image.

## Licence

GPL-3.0-or-later, see `LICENSES/`. REUSE-compliant.
