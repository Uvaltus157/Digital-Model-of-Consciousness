from __future__ import annotations

import builtins
import os
import re
import sys
import traceback
from typing import Any


RED = "\033[31m"
RESET = "\033[0m"

_ERROR_RE = re.compile(
    r"\b(error|failed|failure|exception|traceback|crashed|fatal)\b|ошиб",
    re.IGNORECASE,
)
_ORIGINAL_PRINT = builtins.print
_ORIGINAL_EXCEPTHOOK = sys.excepthook
_ORIGINAL_STDERR = sys.stderr
_INSTALLED = False


def _colors_enabled() -> bool:
    if os.environ.get("DISABLE_ERROR_COLORS"):
        return False
    return True


def red(text: str) -> str:
    if not _colors_enabled() or not text:
        return text
    if text.startswith(RED) and text.endswith(RESET):
        return text
    return f"{RED}{text}{RESET}"


def is_error_text(text: str) -> bool:
    return bool(_ERROR_RE.search(text or ""))


def print_error(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("file", sys.stderr)
    _ORIGINAL_PRINT(red(" ".join(str(arg) for arg in args)), **kwargs)


def _colored_print(*args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    if sep is None:
        sep = " "
    text = sep.join(str(arg) for arg in args)
    if is_error_text(text):
        args = (red(text),)
        kwargs["sep"] = ""
    _ORIGINAL_PRINT(*args, **kwargs)


def _colored_excepthook(exc_type, exc_value, exc_tb) -> None:
    if not _colors_enabled():
        _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_tb)
        return
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        _ORIGINAL_STDERR.write(red(text))
        if not text.endswith("\n"):
            _ORIGINAL_STDERR.write("\n")
    except Exception:
        _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_tb)


class _ColoringStderr:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        if _colors_enabled() and is_error_text(str(text)):
            text = red(str(text))
        return self._wrapped.write(text)

    def flush(self) -> None:
        return self._wrapped.flush()

    def isatty(self) -> bool:
        return self._wrapped.isatty()

    def fileno(self) -> int:
        return self._wrapped.fileno()

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


def install_colored_errors() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    builtins.print = _colored_print
    sys.excepthook = _colored_excepthook
    sys.stderr = _ColoringStderr(sys.stderr)
    _INSTALLED = True
