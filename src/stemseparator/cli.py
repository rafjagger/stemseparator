# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stemseparator command line: arguments, dry-run, preflight, reporting, exit codes."""

import argparse
import sys
from pathlib import Path
from typing import Callable, TextIO

from stemseparator import cost, pipeline
from stemseparator.audio_io import UndecodableError, read_track
from stemseparator.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from stemseparator.layout import plan_track
from stemseparator.pipeline import (
    SUM_WARN_DB, BatchSummary, Naming, TrackResult, collect_inputs, resolve_names, run_batch)
from stemseparator.remote.registry import MODELS
from stemseparator.separator import Separator, SeparatorUnavailable

EXIT_OK = 0
EXIT_FAILED_TRACKS = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def format_duration(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def format_cost(usd: float | None) -> str:
    return "cost unknown" if usd is None else f"~${usd:.2f}"


class _Parser(argparse.ArgumentParser):
    """Usage errors go to the injected stream and exit with code 2."""

    def __init__(self, err: TextIO, **kwargs):
        super().__init__(**kwargs)
        self._err = err

    def error(self, message):
        self.print_usage(self._err)
        print(f"{self.prog}: error: {message}", file=self._err)
        raise SystemExit(EXIT_USAGE)


def _build_parser(err: TextIO) -> argparse.ArgumentParser:
    parser = _Parser(err, prog="stemseparator",
                     description="Separate tracks into four-stem sets for StemDeck.")
    parser.add_argument("paths", nargs="+", type=Path, help="audio files or folders")
    parser.add_argument("--artist")
    parser.add_argument("--album")
    parser.add_argument("--title", help="only valid with a single input track")
    parser.add_argument("--library", type=Path, help="overrides the config's library")
    parser.add_argument("--dry-run", action="store_true",
                        help="show names and cost estimate, write and upload nothing")
    parser.add_argument("--long", action="store_true", help="allow tracks longer than 20 min")
    return parser


def _load_settings(args, config_path: Path, known_models: frozenset[str] | None) -> Config:
    config = load_config(config_path)
    unknown = [name for name in config.models if known_models is not None
               and name not in known_models]
    if unknown:
        raise ConfigError(f"unknown model(s) {', '.join(unknown)} in {config_path} "
                          f"(known: {', '.join(sorted(known_models))})")
    if args.library is None:
        return config
    return Config(library=args.library.expanduser(), models=config.models,
                  parallel=config.parallel)


def _print_failed(failed: list[TrackResult], out: TextIO) -> None:
    if not failed:
        return
    print("failed:", file=out)
    for result in failed:
        print(f"  {result.source}: {result.reason}", file=out)


def _dry_run_line(source: Path, config: Config, naming: Naming,
                  allow_long: bool) -> tuple[str, float, str | None]:
    """Returns (line, estimated usd, failure reason or None)."""
    try:
        track = read_track(source)
    except UndecodableError as error:
        return f"{source} → failed: {error}", 0.0, str(error)
    if track.seconds > pipeline.MAX_SECONDS and not allow_long:
        reason = "longer than 20 min (use --long)"
        return f"{source} → failed: {reason}", 0.0, reason
    artist, album, title = resolve_names(track, naming)
    plan = plan_track(config.library, artist, album, title, source)
    first_stem = plan.stem_paths[0].relative_to(config.library)
    estimate = cost.estimate_usd(track.seconds, config.models)
    detail = f"{format_duration(track.seconds)}, {format_cost(estimate)}"
    return f"{source} → {first_stem} … ({detail})", estimate, None


def _dry_run(sources: list[Path], refused: list[TrackResult], config: Config, naming: Naming,
             allow_long: bool, out: TextIO) -> int:
    failed = list(refused)
    total = 0.0
    for source in sources:
        line, estimate, reason = _dry_run_line(source, config, naming, allow_long)
        print(line, file=out)
        total += estimate
        if reason is not None:
            failed.append(TrackResult(source=source, ok=False, reason=reason))
    print(f"{len(sources)} track(s), estimated {format_cost(total)} (approx.)", file=out)
    _print_failed(failed, out)
    return EXIT_FAILED_TRACKS if failed else EXIT_OK


def _result_cost(result: TrackResult) -> str:
    return format_cost(cost.cost_usd(result.gpu_seconds, result.gpu))


def _warnings(result: TrackResult) -> list[str]:
    lines = []
    if result.residual_db > SUM_WARN_DB:
        lines.append(f"warning: stems do not add up to the mix "
                     f"(residual {result.residual_db:.1f} dB)")
    if result.clipped > 0:
        lines.append(f"warning: {result.clipped} sample(s) clipped")
    return lines


def _make_reporter(total: int, out: TextIO) -> Callable[[TrackResult], None]:
    counter = {"finished": 0}

    def report(result: TrackResult) -> None:
        counter["finished"] += 1
        prefix = f"[{counter['finished']}/{total}]"
        if not result.ok:
            print(f"{prefix} {result.source.stem}: failed: {result.reason}", file=out)
            return
        print(f"{prefix} {result.plan.title}: done "
              f"(GPU {result.gpu_seconds:.0f} s, {_result_cost(result)})", file=out)
        for line in _warnings(result):
            print(f"  {line}", file=out)

    return report


def _print_summary(summary: BatchSummary, refused: list[TrackResult], out: TextIO) -> None:
    cost_text = format_cost(summary.cost_usd)
    cost_text += "" if summary.cost_usd is None else " (approx.)"
    print(f"{summary.done} done, {summary.failed + len(refused)} failed · "
          f"GPU {summary.gpu_seconds:.0f} s · {cost_text}", file=out)
    _print_failed([r for r in summary.results if not r.ok] + refused, out)


def _real_run(sources, refused, config, naming, allow_long, factory, out, err) -> int:
    if not sources:
        _print_failed(refused, out)
        return EXIT_FAILED_TRACKS
    try:
        separator = factory()
    except Exception as error:
        # A missing modal install or a broken client is a setup problem, not a crash.
        print(f"error: cannot set up the separator: {error}", file=err)
        return EXIT_USAGE
    try:
        separator.preflight()
    except SeparatorUnavailable as error:
        print(f"error: {error}", file=err)
        return EXIT_USAGE
    summary = run_batch(sources, parallel=config.parallel, report=_make_reporter(len(sources), out),
                        library=config.library, separator=separator, models=config.models,
                        naming=naming, allow_long=allow_long)
    _print_summary(summary, refused, out)
    return EXIT_FAILED_TRACKS if summary.failed or refused else EXIT_OK


def _default_separator_factory() -> Separator:
    # Imported here so `python -m stemseparator --help` and the tests never load modal.
    from stemseparator.modal_separator import ModalSeparator
    return ModalSeparator()


def main(argv: list[str], *, separator_factory: Callable[[], Separator] | None = None,
         config_path: Path = DEFAULT_CONFIG_PATH,
         known_models: frozenset[str] | None = frozenset(MODELS),
         out: TextIO = sys.stdout, err: TextIO = sys.stderr) -> int:
    parser = _build_parser(err)
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    try:
        config = _load_settings(args, config_path, known_models)
    except ConfigError as error:
        print(f"error: {error}", file=err)
        return EXIT_USAGE
    sources, refused = collect_inputs(args.paths)
    if args.title and len(sources) > 1:
        print("error: --title only works with a single input track", file=err)
        return EXIT_USAGE
    naming = Naming(args.artist, args.album, args.title)
    try:
        if args.dry_run:
            return _dry_run(sources, refused, config, naming, args.long, out)
        return _real_run(sources, refused, config, naming, args.long,
                         separator_factory or _default_separator_factory, out, err)
    except KeyboardInterrupt:
        print("interrupted, nothing half-written", file=err)
        return EXIT_INTERRUPTED


def main_entry() -> None:
    sys.exit(main(sys.argv[1:]))
