"""Sandboxed executor for LLM-generated pandas code.

Safety guarantees
-----------------
1. **Static pattern check** — code is scanned for a deny-list of dangerous
   patterns *before* any execution.  A match raises BlockedCodeError immediately.
2. **Restricted namespace** — only ``pd``, ``np``, ``datetime``, ``re``, and
   the target ``df`` are injected.  Built-ins are stripped to a minimal safe
   subset; dangerous names (``open``, ``eval``, ``exec``, ``__import__``, …)
   are absent.
3. **Timeout** — a ``SIGALRM`` signal fires after ``EXECUTOR_TIMEOUT_SECONDS``
   (default 30 s).  Works on macOS / Linux; not available on Windows.
4. **Atomic rollback** — the DataFrame is deep-copied before execution.  If
   anything fails the *original* copy is returned unchanged.

Public API
----------
execute(code, df)  →  ExecutionResult(df, error, timed_out, blocked)
"""

from __future__ import annotations

import builtins as _builtins_module
import re
import signal
import textwrap
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import BLOCKED_CODE_PATTERNS, EXECUTOR_TIMEOUT_SECONDS
from utils.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BlockedCodeError(Exception):
    """Raised when a dangerous pattern is found in the code string."""


class ExecutionTimeoutError(Exception):
    """Raised by the SIGALRM handler when execution exceeds the timeout."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Outcome of a single sandboxed execution attempt."""

    df: pd.DataFrame
    """The (possibly modified) DataFrame.  Always a valid DataFrame — if
    execution failed this is the original unchanged copy."""

    error: str | None = None
    """Full traceback string when execution raised, or ``None`` on success."""

    timed_out: bool = False
    """``True`` when execution was aborted by the timeout."""

    blocked: bool = False
    """``True`` when execution was refused due to a dangerous code pattern."""

    @property
    def success(self) -> bool:
        return self.error is None and not self.timed_out and not self.blocked


# ---------------------------------------------------------------------------
# Restricted built-ins
# ---------------------------------------------------------------------------

# Allow only the safe subset of Python built-ins needed for typical pandas code.
_SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(_builtins_module, name, None)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter",
        "float", "frozenset", "int", "isinstance", "issubclass", "len",
        "list", "map", "max", "min", "print", "range", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "type", "zip",
        # Exceptions that pandas internals may need to catch
        "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "StopIteration",
        "Exception", "BaseException",
        # Constants
        "True", "False", "None",
    )
}
# Remove any None values that arose from missing built-in names
_SAFE_BUILTINS = {k: v for k, v in _SAFE_BUILTINS.items() if v is not None}


def _make_namespace(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "datetime": datetime,
        "re": re,
        "df": df,
    }


# Matches import lines for libraries already injected into the namespace.
# LLMs often add these even though they're unnecessary in the sandbox.
_REDUNDANT_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+(?:numpy|pandas|re|datetime)"
    r"|from\s+(?:numpy|pandas|re|datetime)\s+import\s+\S+)"
    r"(?:\s+as\s+\w+)?\s*$",
    re.MULTILINE,
)


def _strip_redundant_imports(code: str) -> str:
    """Remove import lines for libraries already present in the exec namespace."""
    return _REDUNDANT_IMPORT_RE.sub("", code).strip()


# ---------------------------------------------------------------------------
# Pattern checker
# ---------------------------------------------------------------------------

def check_code(code: str) -> None:
    """Raise :class:`BlockedCodeError` if *code* contains a dangerous pattern.

    Checks are case-sensitive and match substrings (no word-boundary tricks
    needed — the patterns themselves are distinctive enough).
    """
    for pattern in BLOCKED_CODE_PATTERNS:
        if pattern in code:
            raise BlockedCodeError(
                f"Blocked: code contains forbidden pattern '{pattern}'. "
                f"LLM-generated code may not use this construct."
            )


# ---------------------------------------------------------------------------
# Timeout context manager (SIGALRM — Unix only)
# ---------------------------------------------------------------------------

def _in_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


class _Timeout:
    """Context manager that raises ExecutionTimeoutError after *seconds*.

    SIGALRM is only available on Unix and only from the main thread.
    When called from a non-main thread (e.g. Streamlit's page thread) the
    signal setup silently no-ops — execution proceeds without a hard timeout.
    """

    def __init__(self, seconds: int) -> None:
        self._seconds = seconds
        self._active = False

    def __enter__(self) -> "_Timeout":
        if _in_main_thread():
            try:
                signal.signal(signal.SIGALRM, self._handler)
                signal.alarm(self._seconds)
                self._active = True
            except (ValueError, OSError):
                # Signal setup unavailable in this context — skip timeout.
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._active:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
            except (ValueError, OSError):
                pass
        return False

    @staticmethod
    def _handler(signum, frame) -> None:
        raise ExecutionTimeoutError(
            f"Execution exceeded the {EXECUTOR_TIMEOUT_SECONDS}s timeout limit."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute(code: str, df: pd.DataFrame) -> ExecutionResult:
    """Run *code* against *df* in a restricted sandbox.

    Parameters
    ----------
    code:
        Python code string produced by the LLM.  Must assign back to ``df``
        (e.g. ``df = df.dropna()`` or mutate in-place) for changes to be
        visible in the result.
    df:
        The DataFrame to operate on.  Never mutated — a copy is made before
        any execution begins.

    Returns
    -------
    ExecutionResult
        Always returned (never raises).  Check ``.success`` or ``.error``.
    """
    # 1. Static safety check — before touching the DataFrame at all
    try:
        check_code(code)
    except BlockedCodeError as exc:
        _log.warning("execution_blocked", reason=str(exc))
        return ExecutionResult(df=df.copy(), error=str(exc), blocked=True)

    # 2. Snapshot for rollback
    original = df.copy(deep=True)

    # 3. Prepare namespace with a working copy
    working = df.copy(deep=True)
    namespace = _make_namespace(working)

    # 4. Execute with timeout
    clean_code = textwrap.dedent(_strip_redundant_imports(code))
    try:
        with _Timeout(EXECUTOR_TIMEOUT_SECONDS):
            exec(clean_code, namespace)  # noqa: S102  (intentional sandboxed exec)
    except ExecutionTimeoutError as exc:
        _log.warning("execution_timeout", timeout_seconds=EXECUTOR_TIMEOUT_SECONDS)
        return ExecutionResult(df=original, error=str(exc), timed_out=True)
    except Exception:
        tb = traceback.format_exc()
        _log.warning("execution_failed", error=tb[:500])
        return ExecutionResult(df=original, error=tb)

    # 5. Extract result — code may have rebound 'df' in the namespace
    result_df = namespace.get("df", working)

    # Validate it's still a DataFrame (guard against accidental overwrites)
    if not isinstance(result_df, pd.DataFrame):
        err = (
            f"Execution did not return a DataFrame. "
            f"'df' was overwritten with {type(result_df).__name__}. "
            f"Make sure your code assigns results back to 'df'."
        )
        _log.warning("execution_bad_return_type", got=type(result_df).__name__)
        return ExecutionResult(df=original, error=err)

    _log.info(
        "execution_success",
        rows_before=len(original),
        rows_after=len(result_df),
        cols_before=len(original.columns),
        cols_after=len(result_df.columns),
    )
    return ExecutionResult(df=result_df)
