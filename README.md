# stemseparator

A local command-line tool that sends audio (FLAC, WAV or AIFF) to a rented cloud GPU, separates it into four stems and writes the result as a stem set for StemDeck (44.1 kHz, 24-bit FLAC or WAV).

The separation runs on [Modal](https://modal.com) with models loaded through
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
(MSST, pinned to one commit). The model set is chosen with `--models`:

- **`space4`** (default): four stems by spatial job, `1 - low`, `2 - tops`, `3 - music`,
  `4 - vocals`. Drums and bass come from SCNet XL IHF, vocals from Kimberley Jensen's Mel-Band
  RoFormer. The MDX23C DrumSep model (aufr33 + jarredou) splits the drums stem, and only its kick
  is used: LOW = bass + kick, TOPS = drums − kick (hats, cymbals, snare, percussion), so the drums
  stay whole across the two. MUSIC = the mix minus the other three, so the four sum back to the
  mix.
- **`specialist`**: vocals from the Mel-Band RoFormer, drums and bass from SCNet XL IHF, other =
  the mix minus those three.
- **`ensemble4`**: SCNet Masked XL IHF and BS-RoFormer, all four stems averaged.
- **`sw4`**, **`space4sw`**: shootout candidates with BS-RoFormer SW (6 stems). `sw4` takes its
  drums, bass and vocals, and other is the rest (guitar, piano, other). `space4sw` is `space4`
  with SW's drums and bass in place of SCNet's.
- **`space4polar`**, **`space4deux`**: shootout candidates, `space4` with only the vocal model
  swapped: ZFTurbo's BS PolarFormer, or becruily's Mel-Band RoFormer "deux". MUSIC stays the
  exact remainder.

`htdemucs_ft` (Demucs v4, one fine-tuned checkpoint per stem) is there as a reference.

Published SDR in dB. "MSST" is MVSep's Multisong set as MSST's `docs/pretrained_models.md`
reports it. MVSep's own algorithms page uses its own Multisong scale, so its figures cannot be set
against MSST's:

| Model | drums | bass | other | vocals | source |
|---|---|---|---|---|---|
| SCNet XL IHF (space4, specialist: drums, bass) | 11.58 | 11.94 | 6.49 | 9.69 | MSST |
| Mel-Band RoFormer, Kimberley Jensen (vocals) | | | | 10.98 | MSST |
| SCNet Masked XL IHF (ensemble4) | 11.38 | 11.57 | 6.30 | 9.43 | MSST |
| BS-RoFormer ep 17 (ensemble4) | 11.29 | 11.08 | 5.96 | 9.19 | MSST |
| BS-RoFormer SW (sw4, space4sw) | 14.11 | 14.62 | 8.71 | 11.30 | MVSep |
| BS PolarFormer (space4polar: vocals) | | | | 11.00 | MSST |
| Mel-Band RoFormer deux, becruily (space4deux: vocals) | | | | 11.37 | model card |

The DrumSep model is measured on its own test set (MSST): kick 14.54, snare 9.79, toms 10.63,
hh 3.19, ride + crash 6.08. No figures are published for the ensemble or for any residual stem.

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
stemseparator ~/Music/new/                    # every FLAC/WAV/AIFF in the folder and below
stemseparator track.flac --artist X --album Y
stemseparator --models ensemble4 track.flac   # another model set than the config's
stemseparator --dry-run …                     # names and cost estimate, no upload
```

A folder is walked recursively; hidden entries and the library itself (if it lies inside) are left
out. The same file given twice (or a folder plus a file inside it) is separated once. Ctrl-C stops starting new tracks and
waits for the running ones, whose GPU time is already paid; a second Ctrl-C abandons them, and
they write nothing.

Further flags: `--format flac|wav` (overrides the config), `--title` (single track only),
`--library` (overrides the config), `--long` (allow
tracks over 20 minutes). Before the first upload the tool checks that a token exists and that the
app is deployed, and stops with a plain message if not. A failed network or Modal call is retried
twice; a track that still fails leaves nothing in the library.

- **Input:** FLAC, WAV or AIFF only, mono or stereo, any sample rate.
- **Output:** always 44.1 kHz, 24-bit, as FLAC by default or WAV with `--format wav` (the models
  run at 44.1 kHz; a 48 kHz track comes back at 44.1 kHz). StemDeck reads both: FLAC is decoded
  into memory on load, so seeking is as instant as with WAV. Per track:
  `<library>/<Artist>/<Album>/<Title> - 1 - low.flac` … `4 - vocals.flac` (`1 - drums` …
  `4 - vocals` for the other sets), plus the untouched source under `originals/`.
- GPU seconds and cost per track are approximate: wall time on the GPU times Modal's list price;
  cold starts and idle time are not in it.
- The `--dry-run` cost is a rough guess: it assumes 0.3 GPU seconds per second of audio and
  checkpoint run (three for `space4` and `space4sw`, two for `specialist` and `ensemble4`, one
  for `sw4`, four for `htdemucs_ft`), a placeholder
  until the first real runs have been measured.

## Converting your folders

```
stemseparator status ~/Music     # per track: done, pending, changed, unsupported; totals and cost
stemseparator ~/Music            # converts what is pending, skips what is done
bin/stems-convert [folder…]      # both in a row (default ~/Music); extra flags go to both
```

`status` uploads nothing and writes nothing; it ends with the totals and the estimated cost of
the pending tracks (the same rough estimate as `--dry-run`, which shows this plan for a normal
run). Only FLAC, WAV and AIFF are taken; other files are skipped silently, MP3/OGG and the like
are listed as unsupported.

A track is **done** when either of these holds, checked before anything is uploaded:

- the library holds its set under the name its tags give (artist, album, title; `--artist`,
  `--album`, `--title` count), all four stems of the model set present, in FLAC or WAV, and the
  stored original is the same file (a different file under the same name is *changed*, not done);
- `<library>/.stemseparator-manifest.json` knows its content: source path, size, mtime, a sha256
  of the first and last MB, model set, date, output folder. A renamed or moved file is therefore
  still done; a file whose hash differs is *changed* and counts as new.

Done tracks are skipped, never written a second time as `Title (2)`. `--redo` converts them again
and replaces their set. Two files with the same content in one run are converted once. Each
finished track enters the manifest at once, so an interrupted run (Ctrl-C, power cut) continues
with the rest when started again; staging guarantees no half set is ever in the library. A
manifest that cannot be read stops the run without changing anything.

The stem creator will move into A³ Canvas later; this is the stand-in until then.

## Shootout: comparing model sets

```
stemseparator --shootout track.wav --sets specialist,space4,space4polar,space4deux,sw4,space4sw
stemseparator --dry-run --shootout track.wav --sets …    # cost estimate per set
stemseparator compare <library>/shootout                  # measure again, new excerpts
```

`--shootout` separates one track with every set in `--sets` (up to `parallel` at once) into
`<library>/shootout/<set>/<Title> - N - <stem>.flac` (or `.wav` with `--format wav`). `runs.json` there keeps each set's GPU
seconds, cost, residual and stem names; a set run again replaces its folder. A shootout folder
holds one track only: for another track, move it away or give another `--library`. The track may
lie inside the library, not inside `shootout/`.

After the run, `compare` writes `report.md` next to the sets: per set the residual (mix minus
the sum of the stems), per stem its level and the share of each frequency band of the mix it
carries (a vocals stem with sub-bass is bleed), and per stem against the first set the difference
and the correlation (`music` is compared with `other`). `excerpts/` holds one WAV per stem: the same
12 s of every set back to back, 1 s apart, in the order of the report; by default the loudest 12 s
of the mix, `--at SECONDS` picks another start. No measure here is an SDR: a real track has no
reference stems, so the ear decides.

## Config

`~/.config/stemseparator/config.toml`, every key optional:

```toml
library = "~/Music/stems"                       # where stem sets go
models = "space4"                               # model set: space4, space4polar, space4deux, specialist, ensemble4, sw4, space4sw, htdemucs_ft
parallel = 4                                    # tracks on the GPU at once
format = "flac"                                 # stem files: flac (default) or wav
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
| MDX23C DrumSep 6-stem (aufr33 + jarredou) | Hugging Face mirror `Sucial/MSST-WebUI`, revision-pinned; config from `SUC-DriverOld/MSST-WebUI` at a pinned commit | cc-by-nc-sa-4.0 on the mirror (non-commercial); the original release stated none and is offline |
| BS-RoFormer SW (jarredou) | Hugging Face re-upload `FractalSurfer/BS-RoFormer-SW-6stem`, revision-pinned | unclear: the author published none, re-uploads say MIT or unknown |
| BS PolarFormer | MSST GitHub release v1.0.20 | none stated for the weights (MSST's code is MIT) |
| Mel-Band RoFormer deux (becruily) | Hugging Face `becruily/mel-band-roformer-deux`, revision-pinned | cc-by-nc-4.0 (non-commercial) |
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
