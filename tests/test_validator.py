"""Tests for pipeline/validator.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.cleaning_plan import CleaningOperation, CleaningOperationType, CleaningPlan
from pipeline.validator import (
    CheckStatus,
    ValidationReport,
    validate,
    _check_distribution_shift,
    _check_duplicates,
    _check_dtypes,
    _check_null_expectations,
    _check_row_count,
    _check_schema_preserved,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_op(op_id: str, op_type: str, columns: list[str]) -> CleaningOperation:
    return CleaningOperation(
        id=op_id,
        type=CleaningOperationType(op_type),
        description="d",
        impact="i",
        confidence="high",
        reversible=False,
        code="pass",
        column_names=columns,
        alternatives=[],
    )


def _plan(*ops: CleaningOperation) -> CleaningPlan:
    return CleaningPlan(operations=list(ops))


# ---------------------------------------------------------------------------
# ValidationReport helpers
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_all_pass_status(self):
        from pipeline.validator import CheckResult
        report = ValidationReport(checks=[
            CheckResult("a", CheckStatus.PASS, "ok"),
            CheckResult("b", CheckStatus.PASS, "ok"),
        ])
        assert report.status == CheckStatus.PASS

    def test_any_warning_gives_warning(self):
        from pipeline.validator import CheckResult
        report = ValidationReport(checks=[
            CheckResult("a", CheckStatus.PASS, "ok"),
            CheckResult("b", CheckStatus.WARNING, "warn"),
        ])
        assert report.status == CheckStatus.WARNING

    def test_any_fail_gives_fail(self):
        from pipeline.validator import CheckResult
        report = ValidationReport(checks=[
            CheckResult("a", CheckStatus.WARNING, "warn"),
            CheckResult("b", CheckStatus.FAIL, "fail"),
        ])
        assert report.status == CheckStatus.FAIL

    def test_fail_beats_warning(self):
        from pipeline.validator import CheckResult
        report = ValidationReport(checks=[
            CheckResult("a", CheckStatus.WARNING, "w"),
            CheckResult("b", CheckStatus.FAIL, "f"),
            CheckResult("c", CheckStatus.PASS, "p"),
        ])
        assert report.status == CheckStatus.FAIL

    def test_summary_contains_status(self):
        from pipeline.validator import CheckResult
        report = ValidationReport(checks=[
            CheckResult("a", CheckStatus.PASS, "ok"),
        ])
        assert "PASS" in report.summary

    def test_summary_counts_checks(self):
        from pipeline.validator import CheckResult
        report = ValidationReport(checks=[
            CheckResult("a", CheckStatus.PASS, "ok"),
            CheckResult("b", CheckStatus.WARNING, "w"),
        ])
        assert "1 WARNING" in report.summary
        assert "1 PASS" in report.summary


# ---------------------------------------------------------------------------
# _check_row_count
# ---------------------------------------------------------------------------

class TestCheckRowCount:
    def _df(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({"x": range(n)})

    def test_no_rows_dropped_passes(self):
        before = self._df(100)
        r = _check_row_count(before, before.copy())
        assert r.status == CheckStatus.PASS

    def test_small_drop_passes(self):
        r = _check_row_count(self._df(100), self._df(90))
        assert r.status == CheckStatus.PASS

    def test_over_20pct_warns(self):
        r = _check_row_count(self._df(100), self._df(79))
        assert r.status == CheckStatus.WARNING

    def test_exactly_20pct_passes(self):
        r = _check_row_count(self._df(100), self._df(80))
        assert r.status == CheckStatus.PASS

    def test_over_50pct_fails(self):
        r = _check_row_count(self._df(100), self._df(49))
        assert r.status == CheckStatus.FAIL

    def test_exactly_50pct_warns(self):
        # exactly 50% dropped: not CRITICAL (>50%), but IS WARNING (>20%)
        r = _check_row_count(self._df(100), self._df(50))
        assert r.status == CheckStatus.WARNING

    def test_empty_before_passes(self):
        r = _check_row_count(self._df(0), self._df(0))
        assert r.status == CheckStatus.PASS

    def test_rows_increased_warns(self):
        r = _check_row_count(self._df(100), self._df(110))
        assert r.status == CheckStatus.WARNING

    def test_message_contains_counts(self):
        r = _check_row_count(self._df(100), self._df(90))
        assert "100" in r.message
        assert "90" in r.message

    def test_critical_message_mentions_threshold(self):
        r = _check_row_count(self._df(100), self._df(10))
        assert "CRITICAL" in r.message or "50%" in r.message


# ---------------------------------------------------------------------------
# _check_schema_preserved
# ---------------------------------------------------------------------------

class TestCheckSchemaPreserved:
    def _df(self, cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame({c: [1, 2] for c in cols})

    def test_identical_schema_passes(self):
        df = self._df(["a", "b"])
        r = _check_schema_preserved(df, df.copy(), plan=None)
        assert r.status == CheckStatus.PASS

    def test_extra_column_added_passes(self):
        before = self._df(["a", "b"])
        after = self._df(["a", "b", "c"])
        r = _check_schema_preserved(before, after, plan=None)
        assert r.status == CheckStatus.PASS

    def test_column_unexpectedly_removed_fails(self):
        before = self._df(["a", "b"])
        after = self._df(["a"])
        r = _check_schema_preserved(before, after, plan=None)
        assert r.status == CheckStatus.FAIL
        assert "b" in r.message

    def test_drop_column_op_allows_removal(self):
        before = self._df(["a", "b"])
        after = self._df(["a"])
        plan = _plan(_make_op("drop_b", "drop_column", ["b"]))
        r = _check_schema_preserved(before, after, plan=plan)
        assert r.status == CheckStatus.PASS

    def test_rename_columns_op_allows_old_col_removal(self):
        before = self._df(["old_name", "b"])
        after = self._df(["new_name", "b"])
        plan = _plan(_make_op("rename", "rename_columns", ["old_name"]))
        r = _check_schema_preserved(before, after, plan=plan)
        assert r.status == CheckStatus.PASS

    def test_multiple_removed_all_listed(self):
        before = self._df(["a", "b", "c"])
        after = self._df(["a"])
        r = _check_schema_preserved(before, after, plan=None)
        assert r.status == CheckStatus.FAIL
        assert "b" in r.message or "c" in r.message


# ---------------------------------------------------------------------------
# _check_null_expectations
# ---------------------------------------------------------------------------

class TestCheckNullExpectations:
    def test_no_plan_returns_empty(self):
        df = pd.DataFrame({"x": [1, None]})
        assert _check_null_expectations(df, plan=None) == []

    def test_fill_missing_op_column_has_no_nulls_passes(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0, 3.0]})
        plan = _plan(_make_op("fill_rev", "fill_missing", ["revenue"]))
        results = _check_null_expectations(df, plan=plan)
        assert len(results) == 1
        assert results[0].status == CheckStatus.PASS

    def test_fill_missing_op_column_still_has_nulls_warns(self):
        df = pd.DataFrame({"revenue": [1.0, None, 3.0]})
        plan = _plan(_make_op("fill_rev", "fill_missing", ["revenue"]))
        results = _check_null_expectations(df, plan=plan)
        assert len(results) == 1
        assert results[0].status == CheckStatus.WARNING
        assert "revenue" in results[0].message

    def test_drop_missing_op_checked_too(self):
        df = pd.DataFrame({"col": [1.0, None]})
        plan = _plan(_make_op("drop_col", "drop_missing", ["col"]))
        results = _check_null_expectations(df, plan=plan)
        assert results[0].status == CheckStatus.WARNING

    def test_non_fill_ops_not_checked(self):
        df = pd.DataFrame({"col": [1.0, None]})
        plan = _plan(_make_op("ws", "fix_whitespace", ["col"]))
        assert _check_null_expectations(df, plan=plan) == []

    def test_missing_column_in_after_skipped(self):
        df = pd.DataFrame({"other": [1, 2]})
        plan = _plan(_make_op("fill_rev", "fill_missing", ["revenue"]))
        results = _check_null_expectations(df, plan=plan)
        assert results == []


# ---------------------------------------------------------------------------
# _check_distribution_shift
# ---------------------------------------------------------------------------

class TestCheckDistributionShift:
    def test_no_shift_passes(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        results = _check_distribution_shift(df, df.copy())
        assert all(r.status == CheckStatus.PASS for r in results)

    def test_large_shift_warns(self):
        before = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # Shift mean from ~3 to ~100 — far more than 2 std
        after = pd.DataFrame({"x": [98.0, 99.0, 100.0, 101.0, 102.0]})
        results = _check_distribution_shift(before, after)
        assert any(r.status == CheckStatus.WARNING for r in results)

    def test_message_contains_column_name(self):
        before = pd.DataFrame({"revenue": [1.0, 2.0, 3.0, 4.0, 5.0]})
        after = pd.DataFrame({"revenue": [100.0, 101.0, 102.0, 103.0, 104.0]})
        results = _check_distribution_shift(before, after)
        assert "revenue" in results[0].message

    def test_non_numeric_column_skipped(self):
        before = pd.DataFrame({"name": ["Alice", "Bob"]})
        after = pd.DataFrame({"name": ["Charlie", "Dave"]})
        results = _check_distribution_shift(before, after)
        assert results == []

    def test_column_missing_in_after_skipped(self):
        before = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        after = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        results = _check_distribution_shift(before, after)
        assert all(r.name != "distribution_y" for r in results)

    def test_constant_column_not_checked(self):
        before = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        after = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        # std == 0 — division skipped, no result generated
        results = _check_distribution_shift(before, after)
        assert results == []

    def test_multiple_numeric_columns_all_checked(self):
        before = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        after = before.copy()
        results = _check_distribution_shift(before, after)
        names = {r.name for r in results}
        assert "distribution_a" in names
        assert "distribution_b" in names


# ---------------------------------------------------------------------------
# _check_duplicates
# ---------------------------------------------------------------------------

class TestCheckDuplicates:
    def test_no_duplicates_passes(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        r = _check_duplicates(df, plan=None)
        assert r.status == CheckStatus.PASS

    def test_duplicates_without_dedup_op_passes(self):
        """Duplicates are fine if no drop_duplicates was in the plan."""
        df = pd.DataFrame({"x": [1, 1, 2]})
        r = _check_duplicates(df, plan=None)
        assert r.status == CheckStatus.PASS

    def test_duplicates_after_dedup_op_warns(self):
        df = pd.DataFrame({"x": [1, 1, 2]})
        plan = _plan(_make_op("dedup", "drop_duplicates", []))
        r = _check_duplicates(df, plan=plan)
        assert r.status == CheckStatus.WARNING

    def test_no_duplicates_after_dedup_op_passes(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        plan = _plan(_make_op("dedup", "drop_duplicates", []))
        r = _check_duplicates(df, plan=plan)
        assert r.status == CheckStatus.PASS


# ---------------------------------------------------------------------------
# _check_dtypes
# ---------------------------------------------------------------------------

class TestCheckDtypes:
    def test_unchanged_dtypes_pass(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "name": ["a", "b"]})
        results = _check_dtypes(df, df.copy(), plan=None)
        assert all(r.status == CheckStatus.PASS for r in results)

    def test_numeric_to_object_warns(self):
        before = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        after = pd.DataFrame({"x": ["one", "two", "three"]})
        results = _check_dtypes(before, after, plan=None)
        assert any(r.status == CheckStatus.WARNING for r in results)
        assert any("x" in r.message for r in results)

    def test_fix_dtype_op_column_not_checked(self):
        before = pd.DataFrame({"x": [1.0, 2.0]})
        after = pd.DataFrame({"x": ["a", "b"]})
        plan = _plan(_make_op("fix_x", "fix_dtype", ["x"]))
        results = _check_dtypes(before, after, plan=plan)
        # x is in fix_dtype_cols — should be skipped entirely
        assert not any("x" in r.name for r in results)

    def test_column_removed_from_after_skipped(self):
        before = pd.DataFrame({"x": [1.0], "y": [2.0]})
        after = pd.DataFrame({"x": [1.0]})
        results = _check_dtypes(before, after, plan=None)
        # y is not in after — only dtype_x should appear; use exact name match
        assert not any(r.name == "dtype_y" for r in results)
        assert len(results) == 1
        assert results[0].name == "dtype_x"

    def test_object_stays_object_passes(self):
        before = pd.DataFrame({"name": ["Alice", "Bob"]})
        after = pd.DataFrame({"name": ["Alice", "Bob"]})
        results = _check_dtypes(before, after, plan=None)
        assert all(r.status == CheckStatus.PASS for r in results)


# ---------------------------------------------------------------------------
# validate — integration
# ---------------------------------------------------------------------------

class TestValidateIntegration:
    def _before(self) -> pd.DataFrame:
        return pd.DataFrame({
            "name": ["Alice ", "Bob", "Charlie", "Alice "],
            "revenue": [100.0, 200.0, None, 400.0],
        })

    def test_clean_validation_passes(self):
        before = self._before()
        after = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "revenue": [100.0, 200.0, 300.0],
        })
        plan = _plan(
            _make_op("ws", "fix_whitespace", ["name"]),
            _make_op("fill", "fill_missing", ["revenue"]),
            _make_op("dedup", "drop_duplicates", []),
        )
        report = validate(before, after, plan)
        assert report.status in (CheckStatus.PASS, CheckStatus.WARNING)

    def test_excessive_row_drop_fails(self):
        before = pd.DataFrame({"x": range(100)})
        after = pd.DataFrame({"x": range(10)})
        report = validate(before, after, plan=None)
        assert report.status == CheckStatus.FAIL

    def test_schema_mismatch_fails(self):
        before = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        after = pd.DataFrame({"a": [1, 2]})
        report = validate(before, after, plan=None)
        assert report.status == CheckStatus.FAIL

    def test_distribution_shift_detected(self):
        before = pd.DataFrame({"revenue": [1.0, 2.0, 3.0, 4.0, 5.0]})
        after = pd.DataFrame({"revenue": [100.0, 101.0, 102.0, 103.0, 104.0]})
        report = validate(before, after, plan=None)
        assert any(
            "distribution_revenue" in c.name and c.status == CheckStatus.WARNING
            for c in report.checks
        )

    def test_returns_validation_report(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = validate(df, df.copy(), plan=None)
        assert isinstance(report, ValidationReport)

    def test_no_plan_still_runs(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = validate(df, df.copy(), plan=None)
        assert report.status == CheckStatus.PASS

    def test_summary_is_string(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = validate(df, df.copy(), plan=None)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_all_check_names_unique(self):
        before = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        report = validate(before, before.copy(), plan=None)
        names = [c.name for c in report.checks]
        assert len(names) == len(set(names))
