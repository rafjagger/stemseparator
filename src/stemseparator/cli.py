# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stemseparator command line: arguments, dry-run, preflight, reporting, exit codes."""

import argparse
import sys
from pathlib import Path
from typing import Callable, TextIO

from stemseparator import batch_plan, compare, cost, pipeline, shootout
from stemseparator.audio_io import UndecodableError, read_track
from stemseparator.batch_plan import CHANGED, DONE, DUPLICATE, PENDING, UNSUPPORTED, Item
from stemseparator.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from stemseparator.layout import plan_track
from stemseparator.manifest import Manifest, ManifestError
from stemseparator.naming import OUTPUT_FORMATS, STEM_NAMES
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
    parser.add_argument("paths", nargs="*", type=Path, help="audio files or folders")
    parser.add_argument("--artist")
    parser.add_argument("--album")
    parser.add_argument("--title", help="only valid with a single input track")
    parser.add_argument("--library", type=Path, help="overrides the config's library")
    parser.add_argument("--dry-run", action="store_true",
                        help="show names and cost estimate, write and upload nothing")
    parser.add_argument("--long", action="store_true", help="allow tracks longer than 20 min")
    parser.add_argument("--redo", action="store_true",
                        help="separate tracks the library already holds again, replacing their sets")
    parser.add_argument("--models", metavar="SET",
                        help=f"model set, overrides the config ({', '.join(MODEL_SETS)})")
    parser.add_argument("--format", choices=OUTPUT_FORMATS,
                        help="stem files as 24-bit FLAC or WAV, overrides the config (default flac)")
    parser.add_argument("--shootout", metavar="TRACK", type=Path,
                        help="separate one track with every set in --sets, into "
                             "<library>/shootout/<set>/, and compare them")
    parser.add_argument("--sets", metavar="A,B,…", help="the model sets of a --shootout")
    return parser


def _build_compare_parser(err: TextIO) -> argparse.ArgumentParser:
    parser = _Parser(err, prog="stemseparator compare",
                     description="Measure the stem sets in a shootout folder against the mix and "
                                 "cut listening excerpts; writes report.md and excerpts/.")
    parser.add_argument("folder", type=Path, help="a shootout folder (<library>/shootout)")
    parser.add_argument("--mix", type=Path, help="the track (default: from runs.json)")
    parser.add_argument("--at", type=float, metavar="SECONDS",
                        help="start of the excerpts (default: the loudest 12 s of the mix)")
    return parser


def _build_status_parser(err: TextIO) -> argparse.ArgumentParser:
    parser = _Parser(err, prog="stemseparator status",
                     description="List which tracks in the folders are converted, pending, "
                                 "changed or unsupported. Uploads nothing.")
    parser.add_argument("paths", nargs="+", type=Path, help="audio files or folders")
    parser.add_argument("--artist")
    parser.add_argument("--album")
    parser.add_argument("--library", type=Path, help="overrides the config's library")
    parser.add_argument("--models", metavar="SET", help="model set, overrides the config")
    parser.add_argument("--format", choices=OUTPUT_FORMATS, help=argparse.SUPPRESS)
    parser.add_argument("--long", action="store_true", help="allow tracks longer than 20 min")
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
    output_format = config.format if args.format is None else args.format
    return Config(library=library, models=models, parallel=config.parallel, format=output_format)


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
                      stem_names=_stem_names(config.models), extension=config.format)
    first_stem = plan.stem_paths[0].relative_to(config.library)
    estimate = cost.estimate_usd(track.seconds, config.models)
    detail = f"{format_duration(track.seconds)}, {format_cost(estimate)}"
    return f"{source} → {first_stem} … ({detail})", estimate, None


def _skip_line(item: Item) -> str:
    if item.state == DUPLICATE:
        return f"{item.source} → skipped: same content as {item.location}"
    return f"{item.source} → skipped: already done ({item.location})"


def _runnable(items: list[Item], redo: bool) -> list[Item]:
    wanted = (PENDING, CHANGED, DONE) if redo else (PENDING, CHANGED)
    return [item for item in items if item.state in wanted]


def _dry_run(items: list[Item], refused: list[TrackResult], config: Config, naming: Naming,
             allow_long: bool, redo: bool, out: TextIO) -> int:
    failed = list(refused)
    runnable = _runnable(items, redo)
    total = 0.0
    for item in items:
        if item.state == UNSUPPORTED:
            print(f"{item.source} → failed: {item.reason}", file=out)
            failed.append(TrackResult(source=item.source, ok=False, reason=item.reason))
        elif item not in runnable:
            print(_skip_line(item), file=out)
    for item in runnable:
        line, estimate, reason = _dry_run_line(item.source, config, naming, allow_long)
        print(line + (" [changed since done]" if item.state == CHANGED else ""), file=out)
        total += estimate
        if reason is not None:
            failed.append(TrackResult(source=item.source, ok=False, reason=reason))
    skipped = sum(1 for item in items if item.state in (DONE, DUPLICATE)) - sum(
        1 for item in runnable if item.state == DONE)
    note = f", {skipped} already done (skipped)" if skipped else ""
    print(f"{len(runnable)} track(s), estimated {format_cost(total)} (approx.){note}", file=out)
    _print_failed(failed, out)
    return EXIT_FAILED_TRACKS if failed else EXIT_OK


def _status_line(item: Item) -> str:
    if item.state == DONE:
        return f"done        {item.source}  → {item.location}"
    if item.state == DUPLICATE:
        return f"duplicate   {item.source}  same content as {item.location}"
    if item.state == UNSUPPORTED:
        return f"unsupported {item.source}  {item.reason}"
    detail = f"{format_duration(item.seconds)}, {{estimate}}"
    suffix = "  differs from the converted file" if item.state == CHANGED else ""
    return f"{item.state:<11} {item.source}  ({detail}){suffix}"


def _status(items: list[Item], refused: list[TrackResult], model_set: str, out: TextIO) -> int:
    todo, estimate = [], 0.0
    for item in items:
        line = _status_line(item)
        if item.state in (PENDING, CHANGED):
            usd = cost.estimate_usd(item.seconds, model_set)
            estimate += usd
            todo.append(item)
            line = line.format(estimate=format_cost(usd))
        print(line, file=out)
    for result in refused:
        print(f"unsupported {result.source}  {result.reason}", file=out)
    counts = {state: sum(1 for item in items if item.state == state)
              for state in (DONE, PENDING, CHANGED, DUPLICATE, UNSUPPORTED)}
    counts[UNSUPPORTED] += len(refused)
    print(" · ".join([", ".join(f"{count} {state}" for state, count in counts.items() if count)
                      or "no tracks",
                      f"to do: {len(todo)} track(s), estimated {format_cost(estimate)} (approx.)"]),
          file=out)
    return EXIT_OK


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


def _make_reporter(total: int, out: TextIO,
                   after_done: Callable[[TrackResult], None] = lambda result: None
                   ) -> Callable[[TrackResult], None]:
    counter = {"finished": 0}

    def report(result: TrackResult) -> None:
        counter["finished"] += 1
        if result.ok:
            after_done(result)
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


def _remember(manifest: Manifest, items: list[Item], config: Config) -> Callable[[TrackResult], None]:
    """Each finished track goes into the manifest at once, so an interrupted run resumes."""
    fingerprints = {item.source: item.fingerprint for item in items}

    def record(result: TrackResult) -> None:
        manifest.record(result.source, fingerprints[result.source], config.models,
                        str(result.plan.album_dir.relative_to(config.library)), result.plan.title)
    return record


def _adopt_untracked(manifest: Manifest, items: list[Item], config: Config) -> None:
    """A set found by its name gets a manifest entry, so a later rename still counts as done."""
    for item in items:
        if item.state != DONE or manifest.with_content(item.fingerprint, config.models):
            continue
        location = Path(item.location)
        manifest.record(item.source, item.fingerprint, config.models, str(location.parent),
                        location.name)


def _real_run(items, refused, manifest, config, naming, redo, allow_long, factory, out, err) -> int:
    runnable = _runnable(items, redo)
    skipped = len([item for item in items if item.state in (DONE, DUPLICATE)]) - len(
        [item for item in runnable if item.state == DONE])
    if skipped:
        print(f"skipping {skipped} track(s) already done (--redo converts them again)", file=out)
    problem = _library_problem(config.library)
    if problem is not None:
        print(f"error: {problem}", file=err)
        return EXIT_USAGE
    _adopt_untracked(manifest, items, config)
    unsupported = [TrackResult(source=item.source, ok=False, reason=item.reason)
                   for item in items if item.state == UNSUPPORTED]
    refused = [*refused, *unsupported]
    if not runnable:
        _print_failed(refused, out)
        return EXIT_FAILED_TRACKS if refused else EXIT_OK
    sources = [item.source for item in runnable]
    separator = _build_separator(factory, err)
    if separator is None:
        return EXIT_USAGE
    at_once = min(config.parallel, len(sources))
    print(f"separating {len(sources)} track(s) with {config.models}, up to {at_once} at a time…",
          file=out, flush=True)
    report = _make_reporter(len(sources), out, _remember(manifest, items, config))
    summary = run_batch(sources, parallel=config.parallel, report=report,
                        on_stopping=_stopping_notice(err), on_abandoning=_abandoning_notice(err),
                        overwrite=frozenset(item.source for item in runnable if item.state == DONE),
                        library=config.library, separator=separator, model_set=config.models,
                        naming=naming, allow_long=allow_long, output_format=config.format)
    _print_summary(summary, refused, out)
    if summary.interrupted:
        print(INTERRUPTED_LINE, file=err)
        return EXIT_INTERRUPTED
    return EXIT_FAILED_TRACKS if summary.failed or refused else EXIT_OK


def _usage(message: str, err: TextIO) -> int:
    print(f"error: {message}", file=err)
    return EXIT_USAGE


def _build_separator(factory: Callable[[], Separator], err: TextIO) -> Separator | None:
    try:
        separator = factory()
    except Exception as error:
        # A missing modal install or a broken client is a setup problem, not a crash.
        print(f"error: cannot set up the separator: {error}", file=err)
        return None
    try:
        separator.preflight()
    except SeparatorUnavailable as error:
        print(f"error: {error}", file=err)
        return None
    return separator


def _shootout_sets(args, known_sets: frozenset[str] | None) -> list[str]:
    if args.paths:
        raise ConfigError("--shootout takes one track; give no other paths")
    if not args.sets:
        raise ConfigError("--shootout needs --sets, e.g. --sets specialist,space4")
    names = list(dict.fromkeys(name.strip() for name in args.sets.split(",") if name.strip()))
    for name in names:
        _check_model_set(name, "given with --sets", known_sets)
    return names


def _shootout_dry_run(track: Path, sets: list[str], out: TextIO) -> int:
    try:
        seconds = read_track(track).seconds
    except UndecodableError as error:
        print(f"{track}: {error}", file=out)
        return EXIT_FAILED_TRACKS
    total = 0.0
    for name in sets:
        estimate = cost.estimate_usd(seconds, name)
        total += estimate
        print(f"{name}: {format_cost(estimate)}", file=out)
    print(f"{len(sets)} set(s) on {track.name} ({format_duration(seconds)}), "
          f"estimated {format_cost(total)} (approx.)", file=out)
    return EXIT_OK


def _report_set(out: TextIO) -> Callable[[shootout.SetRun], None]:
    def report(run: shootout.SetRun) -> None:
        if not run.ok:
            print(f"{run.set_name}: failed: {run.reason}", file=out, flush=True)
            return
        usd = cost.cost_usd(run.gpu_seconds, run.gpu)
        print(f"{run.set_name}: done (GPU {run.gpu_seconds:.0f} s, {format_cost(usd)})",
              file=out, flush=True)
    return report


def _shootout_run(track: Path, sets: list[str], config: Config, naming: Naming,
                  allow_long: bool, factory, out: TextIO, err: TextIO) -> int:
    problem = _library_problem(config.library)
    if problem is not None:
        return _usage(problem, err)
    separator = _build_separator(factory, err)
    if separator is None:
        return EXIT_USAGE
    print(f"shootout: {track.name} with {', '.join(sets)}…", file=out, flush=True)
    try:
        runs = shootout.run_shootout(track, sets, library=config.library, separator=separator,
                                     naming=naming, parallel=config.parallel,
                                     allow_long=allow_long, report=_report_set(out),
                                     output_format=config.format)
    except shootout.ShootoutError as error:
        return _usage(str(error), err)
    gpu_seconds = sum(run.gpu_seconds for run in runs)
    costs = [cost.cost_usd(run.gpu_seconds, run.gpu) for run in runs if run.gpu_seconds > 0]
    total = None if any(c is None for c in costs) else sum(costs)
    failed = [run for run in runs if not run.ok]
    print(f"{len(runs) - len(failed)} set(s) done, {len(failed)} failed · GPU {gpu_seconds:.0f} s"
          f" · {format_cost(total)}", file=out)
    if len(failed) < len(runs):
        try:
            compare.run_compare(shootout.shootout_dir(config.library), out=out)
        except compare.CompareError as error:
            print(f"warning: no comparison: {error}", file=err)
    return EXIT_FAILED_TRACKS if failed else EXIT_OK


def compare_main(argv: list[str], out: TextIO, err: TextIO) -> int:
    try:
        args = _build_compare_parser(err).parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    if not args.folder.is_dir():
        return _usage(f"{args.folder} is not a folder", err)
    try:
        compare.run_compare(args.folder, mix=args.mix, at_seconds=args.at, out=out)
    except compare.CompareError as error:
        return _usage(str(error), err)
    return EXIT_OK


def _default_separator_factory() -> Separator:
    # Imported here so `python -m stemseparator --help` and the tests never load modal.
    from stemseparator.modal_separator import ModalSeparator
    return ModalSeparator()


def main(argv: list[str], *, separator_factory: Callable[[], Separator] | None = None,
         config_path: Path = DEFAULT_CONFIG_PATH,
         known_sets: frozenset[str] | None = frozenset(MODEL_SETS),
         out: TextIO = sys.stdout, err: TextIO = sys.stderr) -> int:
    if argv[:1] == ["compare"]:
        return compare_main(argv[1:], out, err)
    if argv[:1] == ["status"]:
        return status_main(argv[1:], config_path, known_sets, out, err)
    parser = _build_parser(err)
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    try:
        config = _load_settings(args, config_path, known_sets)
        sets = _shootout_sets(args, known_sets) if args.shootout else []
    except ConfigError as error:
        return _usage(str(error), err)
    naming = Naming(args.artist, args.album, args.title)
    if args.shootout:
        if args.dry_run:
            return _shootout_dry_run(args.shootout, sets, out)
        try:
            return _shootout_run(args.shootout, sets, config, naming, args.long,
                                 separator_factory or _default_separator_factory, out, err)
        except KeyboardInterrupt:
            print("interrupted; set folders already written stay", file=err)
            return EXIT_INTERRUPTED
    if not args.paths:
        return _usage("give audio files or folders (or --shootout TRACK --sets …)", err)
    sources, refused = collect_inputs(args.paths, skip=config.library)
    if args.title and len(sources) > 1:
        print("error: --title only works with a single input track", file=err)
        return EXIT_USAGE
    try:
        manifest = Manifest(config.library)
    except ManifestError as error:
        return _usage(str(error), err)
    try:
        items = _triage(sources, config, naming, manifest, args.long)
        if args.dry_run:
            return _dry_run(items, refused, config, naming, args.long, args.redo, out)
        return _real_run(items, refused, manifest, config, naming, args.redo, args.long,
                         separator_factory or _default_separator_factory, out, err)
    except KeyboardInterrupt:
        print(INTERRUPTED_LINE, file=err)
        return EXIT_INTERRUPTED


def _triage(sources: list[Path], config: Config, naming: Naming, manifest: Manifest,
            allow_long: bool) -> list[Item]:
    return batch_plan.classify(sources, library=config.library, model_set=config.models,
                               stem_names=_stem_names(config.models), naming=naming,
                               manifest=manifest, allow_long=allow_long)


def status_main(argv: list[str], config_path: Path, known_sets: frozenset[str] | None,
                out: TextIO, err: TextIO) -> int:
    try:
        args = _build_status_parser(err).parse_args(argv)
        config = _load_settings(args, config_path, known_sets)
    except SystemExit as stop:
        return int(stop.code or 0)
    except ConfigError as error:
        return _usage(str(error), err)
    sources, refused = collect_inputs(args.paths, skip=config.library)
    try:
        manifest = Manifest(config.library)
    except ManifestError as error:
        return _usage(str(error), err)
    items = _triage(sources, config, Naming(args.artist, args.album), manifest, args.long)
    return _status(items, refused, config.models, out)


def main_entry() -> None:
    sys.exit(main(sys.argv[1:]))
