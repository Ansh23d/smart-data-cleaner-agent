"""Tests for pipeline/executor.py.

All tests run against the real executor (no mocking of exec/signal) so the
sandbox guarantees are exercised end-to-end.

SIGALRM is used for the timeout — tests that exercise it are marked
``@pytest.mark.timeout_test`` and skipped automatically on Windows.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pandas as pd
import pytest

from pipeline.executor import (
    BlockedCodeError,
    ExecutionResult,
    ExecutionTimeoutError,
    check_code,
    execute,
)

# Skip timeout tests on Windows (no SIGALRM)
UNIX = platform.system() != "Windows"
timeout_only = pytest.mark.skipif(not UNIX, reason="SIGALRM not available on Windows")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Alice"],
        "revenue": [100.0, 200.0, None, 400.0],
        "category": ["A", "B", "A", "B"],
    })


# ---------------------------------------------------------------------------
# check_code — static pattern checker
# ---------------------------------------------------------------------------

class TestCheckCode:
    def test_clean_code_passes(self):
        check_code("df = df.dropna()")   # should not raise

    def test_import_os_blocked(self):
        with pytest.raises(BlockedCodeError, match="import os"):
            check_code("import os\nos.system('ls')")

    def test_import_sys_blocked(self):
        with pytest.raises(BlockedCodeError, match="import sys"):
            check_code("import sys")

    def test_subprocess_blocked(self):
        with pytest.raises(BlockedCodeError, match="subprocess"):
            check_code("import subprocess; subprocess.run(['ls'])")

    def test_open_blocked(self):
        with pytest.raises(BlockedCodeError, match=r"open\("):
            check_code("f = open('/etc/passwd')")

    def test_dunder_import_blocked(self):
        with pytest.raises(BlockedCodeError, match="__import__"):
            check_code("os = __import__('os')")

    def test_eval_blocked(self):
        with pytest.raises(BlockedCodeError, match=r"eval\("):
            check_code("eval('print(1)')")

    def test_exec_blocked(self):
        with pytest.raises(BlockedCodeError, match=r"exec\("):
            check_code("exec('x=1')")

    def test_import_socket_blocked(self):
        with pytest.raises(BlockedCodeError):
            check_code("import socket")

    def test_import_requests_blocked(self):
        with pytest.raises(BlockedCodeError):
            check_code("import requests")

    def test_import_urllib_blocked(self):
        with pytest.raises(BlockedCodeError):
            check_code("import urllib")

    def test_multiline_with_blocked_pattern(self):
        code = "df = df.dropna()\nimport os\ndf['x'] = 1"
        with pytest.raises(BlockedCodeError):
            check_code(code)


# ---------------------------------------------------------------------------
# execute — successful execution
# ---------------------------------------------------------------------------

class TestValidExecution:
    def test_drop_nulls(self, simple_df):
        result = execute("df = df.dropna()", simple_df)
        assert result.success
        assert len(result.df) == 3    # one NaN row removed
        assert result.error is None

    def test_fill_missing(self, simple_df):
        result = execute(
            "df['revenue'] = df['revenue'].fillna(df['revenue'].median())",
            simple_df,
        )
        assert result.success
        assert result.df["revenue"].isna().sum() == 0

    def test_drop_duplicates(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Alice"],
            "score": [10, 20, 10],
        })
        result = execute("df = df.drop_duplicates()", df)
        assert result.success
        assert len(result.df) == 2

    def test_add_column(self, simple_df):
        result = execute("df['doubled'] = df['revenue'] * 2", simple_df)
        assert result.success
        assert "doubled" in result.df.columns

    def test_rename_column(self, simple_df):
        result = execute("df = df.rename(columns={'revenue': 'rev'})", simple_df)
        assert result.success
        assert "rev" in result.df.columns
        assert "revenue" not in result.df.columns

    def test_filter_rows(self, simple_df):
        result = execute("df = df[df['category'] == 'A']", simple_df)
        assert result.success
        assert len(result.df) == 2

    def test_multiline_code(self, simple_df):
        code = """
median_rev = df['revenue'].median()
df['revenue'] = df['revenue'].fillna(median_rev)
df = df.drop_duplicates()
"""
        result = execute(code, simple_df)
        assert result.success
        assert result.df["revenue"].isna().sum() == 0

    def test_numpy_available(self, simple_df):
        result = execute("df['log_rev'] = np.log(df['revenue'].fillna(1))", simple_df)
        assert result.success
        assert "log_rev" in result.df.columns

    def test_datetime_available(self):
        df = pd.DataFrame({"date_str": ["2023-01-01", "2023-02-01"]})
        result = execute(
            "df['parsed'] = pd.to_datetime(df['date_str'])",
            df,
        )
        assert result.success
        assert pd.api.types.is_datetime64_any_dtype(result.df["parsed"])

    def test_re_available(self):
        """re is pre-injected in the namespace; code uses it directly (no import needed)."""
        df = pd.DataFrame({"text": ["hello world", "foo bar"]})
        result = execute(
            "df['has_hello'] = df['text'].apply(lambda s: bool(re.search(r'hello', s)))",
            df,
        )
        assert result.success
        assert list(result.df["has_hello"]) == [True, False]

    def test_result_success_flag(self, simple_df):
        result = execute("df = df.dropna()", simple_df)
        assert result.success is True
        assert result.timed_out is False
        assert result.blocked is False


# ---------------------------------------------------------------------------
# execute — rollback on failure
# ---------------------------------------------------------------------------

class TestRollback:
    def test_original_returned_on_runtime_error(self, simple_df):
        code = "df = df['nonexistent_column']"  # KeyError
        original_cols = list(simple_df.columns)
        result = execute(code, simple_df)
        assert not result.success
        assert list(result.df.columns) == original_cols
        assert len(result.df) == len(simple_df)

    def test_original_returned_on_syntax_error(self, simple_df):
        code = "df = df.dropna("          # SyntaxError — unclosed paren
        result = execute(code, simple_df)
        assert not result.success
        assert len(result.df) == len(simple_df)

    def test_original_not_mutated_by_working_copy(self, simple_df):
        """Ensure the original df object is never touched, even on success."""
        original_revenue_col = simple_df["revenue"].copy()
        execute("df['revenue'] = df['revenue'].fillna(0)", simple_df)
        pd.testing.assert_series_equal(simple_df["revenue"], original_revenue_col)

    def test_error_string_contains_traceback(self, simple_df):
        result = execute("x = df['nonexistent_col']", simple_df)
        assert result.error is not None
        assert "KeyError" in result.error or "nonexistent_col" in result.error

    def test_syntax_error_captured(self, simple_df):
        result = execute("df = df[[[", simple_df)
        assert not result.success
        assert result.error is not None


# ---------------------------------------------------------------------------
# execute — blocked patterns
# ---------------------------------------------------------------------------

class TestBlockedPatterns:
    def test_import_os_returns_blocked_result(self, simple_df):
        result = execute("import os\ndf['x'] = 1", simple_df)
        assert result.blocked is True
        assert result.success is False
        assert result.error is not None

    def test_blocked_does_not_modify_df(self, simple_df):
        original_cols = list(simple_df.columns)
        result = execute("import os\ndf['injected'] = 1", simple_df)
        assert result.blocked
        assert list(result.df.columns) == original_cols

    def test_open_returns_blocked(self, simple_df):
        result = execute("f = open('/etc/passwd')\ndf['x'] = 1", simple_df)
        assert result.blocked

    def test_eval_returns_blocked(self, simple_df):
        result = execute("eval('1+1')", simple_df)
        assert result.blocked

    def test_exec_returns_blocked(self, simple_df):
        result = execute("exec('x=1')", simple_df)
        assert result.blocked

    def test_dunder_import_returns_blocked(self, simple_df):
        result = execute("os = __import__('os')", simple_df)
        assert result.blocked

    def test_subprocess_returns_blocked(self, simple_df):
        result = execute("import subprocess", simple_df)
        assert result.blocked


# ---------------------------------------------------------------------------
# execute — restricted namespace
# ---------------------------------------------------------------------------

class TestRestrictedNamespace:
    def test_cannot_access_os_via_builtins(self, simple_df):
        """Even without 'import os', the namespace should not expose it."""
        result = execute("x = os.getcwd()", simple_df)
        assert not result.success      # NameError: 'os' not defined

    def test_open_not_in_builtins(self, simple_df):
        """'open' is stripped from __builtins__ — blocked before this but belt+suspenders."""
        # This won't even hit the exec because 'open(' is blocked by check_code.
        result = execute("open('/etc/passwd')", simple_df)
        assert not result.success

    def test_df_not_dataframe_result(self, simple_df):
        """If code overwrites df with a non-DataFrame, rollback triggers."""
        result = execute("df = 42", simple_df)
        assert not result.success
        assert isinstance(result.df, pd.DataFrame)
        assert len(result.df) == len(simple_df)


# ---------------------------------------------------------------------------
# execute — timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    @timeout_only
    def test_infinite_loop_times_out(self, simple_df):
        result = execute("while True: pass", simple_df)
        assert result.timed_out is True
        assert result.success is False

    @timeout_only
    def test_timeout_returns_original_df(self, simple_df):
        original_len = len(simple_df)
        result = execute("while True: pass", simple_df)
        assert result.timed_out
        assert len(result.df) == original_len

    @timeout_only
    def test_timeout_error_message(self, simple_df):
        result = execute("while True: pass", simple_df)
        assert result.error is not None
        assert "timeout" in result.error.lower() or "exceeded" in result.error.lower()

    @timeout_only
    def test_fast_code_does_not_timeout(self, simple_df):
        result = execute("df = df.dropna()", simple_df)
        assert not result.timed_out
        assert result.success


# ---------------------------------------------------------------------------
# execute — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_code_string(self, simple_df):
        result = execute("", simple_df)
        # Empty code is valid Python — df is returned unchanged
        assert result.success
        pd.testing.assert_frame_equal(result.df, simple_df)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = execute("df = df.dropna()", df)
        assert result.success
        assert len(result.df) == 0

    def test_large_dataframe_modification(self):
        df = pd.DataFrame({"x": range(10_000), "y": range(10_000)})
        result = execute("df['z'] = df['x'] + df['y']", df)
        assert result.success
        assert "z" in result.df.columns
        assert len(result.df) == 10_000

    def test_indented_code_handled(self, simple_df):
        """textwrap.dedent should strip common leading whitespace."""
        code = """
            df = df.dropna()
        """
        result = execute(code, simple_df)
        assert result.success

    def test_comment_only_code(self, simple_df):
        result = execute("# this is a comment", simple_df)
        assert result.success

    def test_execution_result_dataclass(self, simple_df):
        result = execute("df = df.dropna()", simple_df)
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.df, pd.DataFrame)
