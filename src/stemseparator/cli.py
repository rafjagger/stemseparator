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
from stemseparator.naming import STEM_NAMES
from stemseparator.pipeline import (
    SUM_WARN_DB, BatchSummary, Naming, TrackResult, collect_inputs, resolve_names, run_batch)
from stemseparator.remote.registry import MODEL_SETS
from stemseparator.separator import Separator, SeparatorUnavailable
from stemseparator.staging import StagedSet, StagingError

EXIT_OK = 0
EXIT_FAILED_TRACKS = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
INTERRUPTED_LINE = "interrupted, nothing half-written (sets reported as done stay in the library)"


def format_duration(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


format_cost = cost.format_usd


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
    parser.add_argument("--models", metavar="SET",
                        help=f"model set, overrides the config ({', '.join(MODEL_SETS)})")
    return parser


def _check_model_set(name: str, origin: str, known_sets: frozenset[str] | None) -> None:
    if known_sets is None or name in known_sets:
        return
    raise ConfigError(f"unknown model set '{name}' {origin} "
                      f"(known: {', '.join(sorted(known_sets))})")


def _load_settings(args, config_path: Path, known_sets: frozenset[str] | None) -> Config:
    config = load_config(config_path)
    if args.models is not None:
        _check_model_set(args.models, "given with --models", known_sets)
    else:
        _check_model_set(config.models, f"in {config_path}", known_sets)
    library = config.library if args.library is None else args.library.expanduser()
    models = config.models if args.models is None else args.models
    return Config(library=library, models=models, parallel=config.parallel)


def _print_failed(failed: list[TrackResult], out: TextIO) -> None:
    if not failed:
        return
    print("failed:", file=out)
    for result in failed:
        print(f"  {result.source}: {result.reason}", file=out)


def _stem_names(model_set: str) -> tuple[str, ...]:
    known = MODEL_SETS.get(model_set)
    return known.stem_names if known is not None else STEM_NAMES


def _dry_run_line(source: Path, config: Config, naming: Naming,
                  allow_long: bool) -> tuple[str, float, str | None]:
    """Returns (line, estimated usd, failure reason or None)."""
    refusal = pipeline.refusal_before_decode(source, config.library, allow_long)
    if refusal is not None:
        return f"{source} → failed: {refusal}", 0.0, refusal
    try:
        track = read_track(source)
    except UndecodableError as error:
        return f"{source} → failed: {error}", 0.0, str(error)
    artist, album, title = resolve_names(track, naming)
    plan = plan_track(config.library, artist, album, title, source,
                      stem_names=_stem_names(config.models))
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
    unfinished = f", {summary.unfinished} not finished" if summary.unfinished else ""
    print(f"{summary.done} done, {summary.failed + len(refused)} failed{unfinished} · "
          f"GPU {summary.gpu_seconds:.0f} s · {cost_text}", file=out)
    _print_failed([r for r in summary.results if not r.ok] + refused, out)


def _stopping_notice(err: TextIO) -> Callable[[int], None]:
    def notice(running: int) -> None:
        if running == 0:
            print("stopping", file=err)
            return
        print(f"stopping: waiting for {running} running track(s) — their GPU time is already "
              "paid; press Ctrl-C again to abandon them", file=err)
    return notice


def _abandoning_notice(err: TextIO) -> Callable[[int], None]:
    def notice(running: int) -> None:
        print(f"abandoning {running} track(s): they write nothing; the program ends when their "
              "GPU calls return", file=err)
    return notice


def _library_problem(library: Path) -> str | None:
    """Staging once up front: a library that cannot take a set fails before any GPU time."""
    try:
        with StagedSet(library):
            pass
    except StagingError as error:
        return str(error)
    except OSError as error:
        return f"cannot write to the library ({library}): {error.strerror or error}"
    return None


def _real_run(sources, refused, config, naming, allow_long, factory, out, err) -> int:
    if not sources:
        _print_failed(refused, out)
        return EXIT_FAILED_TRACKS
    problem = _library_problem(config.library)
    if problem is not None:
        print(f"error: {problem}", file=err)
        return EXIT_USAGE
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
    at_once = min(config.parallel, len(sources))
    print(f"separating {len(sources)} track(s) with {config.models}, up to {at_once} at a time…",
          file=out, flush=True)
    summary = run_batch(sources, parallel=config.parallel, report=_make_reporter(len(sources), out),
                        on_stopping=_stopping_notice(err), on_abandoning=_abandoning_notice(err),
                        library=config.library, separator=separator, model_set=config.models,
                        naming=naming, allow_long=allow_long)
    _print_summary(summary, refused, out)
    if summary.interrupted:
        print(INTERRUPTED_LINE, file=err)
        return EXIT_INTERRUPTED
    return EXIT_FAILED_TRACKS if summary.failed or refused else EXIT_OK


def _default_separator_factory() -> Separator:
    # Imported here so `python -m stemseparator --help` and the tests never load modal.
    from stemseparator.modal_separator import ModalSeparator
    return ModalSeparator()


def main(argv: list[str], *, separator_factory: Callable[[], Separator] | None = None,
         config_path: Path = DEFAULT_CONFIG_PATH,
         known_sets: frozenset[str] | None = frozenset(MODEL_SETS),
         out: TextIO = sys.stdout, err: TextIO = sys.stderr) -> int:
    parser = _build_parser(err)
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    try:
        config = _load_settings(args, config_path, known_sets)
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
        print(INTERRUPTED_LINE, file=err)
        return EXIT_INTERRUPTED


def main_entry() -> None:
    sys.exit(main(sys.argv[1:]))
