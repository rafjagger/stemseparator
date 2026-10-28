# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    library: Path = Path("~/Music/stems").expanduser()
    models: tuple[str, ...] = ("scnet_xl_ihf", "bs_roformer_ep17")
    parallel: int = 4


class ConfigError(Exception):
    pass


DEFAULT_CONFIG_PATH = Path("~/.config/stemseparator/config.toml").expanduser()
_KNOWN_KEYS = ("library", "models", "parallel")


def _read_table(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return {}
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error


def _check_keys(table: dict, path: Path) -> None:
    for key in table:
        if key not in _KNOWN_KEYS:
            raise ConfigError(f"unknown key '{key}' in {path} "
                              f"(known: {', '.join(_KNOWN_KEYS)})")


def _library(value, path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"'library' in {path} must be a non-empty string")
    return Path(value).expanduser()


def _models(value, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not value \
            or not all(isinstance(name, str) for name in value):
        raise ConfigError(f"'models' in {path} must be a non-empty list of strings")
    return tuple(value)


def _parallel(value, path: Path) -> int:
    # bool is an int subclass; `parallel = true` must not pass as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"'parallel' in {path} must be an integer >= 1")
    return value


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    table = _read_table(path)
    _check_keys(table, path)
    defaults = Config()
    return Config(
        library=_library(table["library"], path) if "library" in table else defaults.library,
        models=_models(table["models"], path) if "models" in table else defaults.models,
        parallel=_parallel(table["parallel"], path) if "parallel" in table else defaults.parallel,
    )
