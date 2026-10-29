# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Separator that calls the functions deployed from modal_app.py.

modal is imported only when a lookup actually happens, so the CLI and the tests never load it.
Errors are matched by class name, which lets tests use stand-in classes.
"""
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from stemseparator import transport
from stemseparator.separator import Separation, SeparatorError, SeparatorUnavailable

APP_NAME = "stemseparator"

_TRANSIENT_ERRORS = frozenset({"ConnectionError", "TimeoutError", "InternalFailure", "ServiceError"})
_BUDGET_WORDS = ("budget", "billing", "credit")


def _modal_lookup(app_name: str, function_name: str) -> Any:
    import modal
    return modal.Function.from_name(app_name, function_name)


def _modal_config_path() -> Path:
    return Path(os.environ.get("MODAL_CONFIG_PATH", "~/.modal.toml")).expanduser()


def _modal_token_present() -> bool:
    return "MODAL_TOKEN_ID" in os.environ or _modal_config_path().exists()


def classify(exc: BaseException) -> SeparatorError:
    name = type(exc).__name__
    return SeparatorError(f"Modal error ({name}): {exc}", transient=name in _TRANSIENT_ERRORS)


def _is_budget_error(exc: BaseException) -> bool:
    if type(exc).__name__ == "ResourceExhaustedError":
        return True
    text = str(exc).lower()
    return any(word in text for word in _BUDGET_WORDS)


def _unavailable(exc: BaseException) -> SeparatorUnavailable:
    name = type(exc).__name__
    if name == "NotFoundError":
        return SeparatorUnavailable(
            f"The Modal app '{APP_NAME}' is not deployed: run `modal deploy modal_app.py`.")
    if name == "AuthError":
        return SeparatorUnavailable("Modal rejected the token: run `modal token new`.")
    if _is_budget_error(exc):
        return SeparatorUnavailable(f"Modal budget or credit reached: {exc}")
    return SeparatorUnavailable(f"Modal is not reachable ({name}): {exc}")


class ModalSeparator:
    def __init__(self, lookup: Callable[[str, str], Any] | None = None,
                 token_present: Callable[[], bool] | None = None):
        self._lookup = lookup or _modal_lookup
        self._token_present = token_present or _modal_token_present

    def preflight(self) -> None:
        if not self._token_present():
            raise SeparatorUnavailable("No Modal token: run `modal token new` once.")
        try:
            self._lookup(APP_NAME, "ping").remote()
        except Exception as error:
            raise _unavailable(error) from error

    def separate(self, audio: np.ndarray, sample_rate: int,
                 model_set: str) -> Separation:
        try:
            upload = transport.encode_upload(audio, sample_rate)
        except Exception as error:
            raise SeparatorError(f"could not prepare the track for upload: {error}",
                                 transient=False) from error
        try:
            result = self._lookup(APP_NAME, "separate").remote(upload, model_set)
        except Exception as error:
            raise classify(error) from error
        return transport.decode_result(result)
