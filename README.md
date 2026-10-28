# stemseparator

A local command-line tool that sends audio (FLAC, WAV or AIFF) to a rented cloud GPU, separates it into four stems and writes the result as a stem set for StemDeck (44.1 kHz, 24-bit).

The separation runs on [Modal](https://modal.com) as an ensemble of two models from
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
(MSST, pinned to one commit): SCNet XL IHF and BS-RoFormer. Their stems are averaged.

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

```
stemseparator ~/Music/new/                    # all audio files in the folder
stemseparator track.flac --artist X --album Y
stemseparator --dry-run …                     # names and cost estimate, no upload
```

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

## Config

`~/.config/stemseparator/config.toml`, every key optional:

```toml
library = "~/Music/stems"                       # where stem sets go
models = ["scnet_xl_ihf", "bs_roformer_ep17"]   # averaged; also known: "htdemucs_ft"
parallel = 4                                    # tracks on the GPU at once
```

## Privacy

The track is sent to Modal for the length of one call; this tool stores no audio there and logs
no file names. Modal routes payloads above 2 MiB — every track — through its own blob storage in
the US, and its retention for that storage is not stated on its data-residency page. Read Modal's
current statement on this before uploading material that must not leave your hands.

## Model weights

The weights are downloaded from their public release pages at run time and are never committed to
this repository. The models were trained on MUSDB18, which is licensed for research and
non-commercial use; check the terms of each model before using its stems commercially.

## Development

```
.venv/bin/python -m unittest discover -s tests -t . -v
```

No test touches the network, Modal, a GPU or torch. `modal_app.py` and
`src/stemseparator/remote/runner.py` run only inside the Modal image.

## Licence

GPL-3.0-or-later, see `LICENSES/`. REUSE-compliant.
