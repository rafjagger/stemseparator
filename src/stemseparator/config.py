# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
import tomllib
from dataclasses import dataclass
from pathlib import Path

from stemseparator.remote.registry import DEFAULT_MODEL_SET


@dataclass(frozen=True)
class Config:
    library: Path = Path("~/Music/stems").expanduser()
    models: str = DEFAULT_MODEL_SET  # the name of a model set in the registry
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


def _models(value, path: Path) -> str:
    if isinstance(value, list):
        # The earlier form listed models to average; a set name replaced it.
        raise ConfigError(f"'models' in {path} names one model set now, "
                          f'e.g. models = "{DEFAULT_MODEL_SET}" or models = "ensemble4"')
    if not isinstance(value, str) or not value:
        raise ConfigError(f"'models' in {path} must be a model set name")
    return value


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
