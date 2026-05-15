"""Post-cleaning validator.

Compares a before-DataFrame and after-DataFrame and returns a structured
ValidationReport describing what changed and whether anything looks wrong.

Public API
----------
validate(before, after, plan) → ValidationReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from config import (
    MEAN_SHIFT_STD_WARNING,
    ROW_DROP_CRITICAL_PCT,
    ROW_DROP_WARNING_PCT,
)
from models.cleaning_plan import CleaningOperation, CleaningPlan


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class CheckStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str


@dataclass
class ValidationReport:
    """Outcome of validate(). Inspect .status for the overall verdict."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def status(self) -> CheckStatus:
        """Worst status across all checks: FAIL > WARNING > PASS."""
        if any(c.status == CheckStatus.FAIL for c in self.checks):
            return CheckStatus.FAIL
        if any(c.status == CheckStatus.WARNING for c in self.checks):
            return CheckStatus.WARNING
        return CheckStatus.PASS

    @property
    def summary(self) -> str:
        """Human-readable one-line summary."""
        counts = {s: 0 for s in CheckStatus}
        for c in self.checks:
            counts[c.status] += 1
        parts = []
        if counts[CheckStatus.FAIL]:
            parts.append(f"{counts[CheckStatus.FAIL]} FAIL")
        if counts[CheckStatus.WARNING]:
            parts.append(f"{counts[CheckStatus.WARNING]} WARNING")
        parts.append(f"{counts[CheckStatus.PASS]} PASS")
        return f"Validation {self.status.value}: {', '.join(parts)} across {len(self.checks)} checks."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(
    before: pd.DataFrame,
    after: pd.DataFrame,
    plan: CleaningPlan | None = None,
) -> ValidationReport:
    """Run all validation checks and return a ValidationReport.

    Parameters
    ----------
    before:
        DataFrame snapshot taken before any cleaning was applied.
    after:
        DataFrame produced by the executor after applying the cleaning plan.
    plan:
        The CleaningPlan that was applied.  Optional — some checks are
        only meaningful when the plan is provided.

    Returns
    -------
    ValidationReport
        Always returned (never raises).
    """
    report = ValidationReport()

    report.checks.append(_check_row_count(before, after))
    report.checks.append(_check_schema_preserved(before, after, plan))
    report.checks.extend(_check_null_expectations(after, plan))
    report.checks.extend(_check_distribution_shift(before, after))
    report.checks.append(_check_duplicates(after, plan))
    report.checks.extend(_check_dtypes(before, after, plan))

    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_row_count(before: pd.DataFrame, after: pd.DataFrame) -> CheckResult:
    n_before = len(before)
    n_after = len(after)

    if n_before == 0:
        return CheckResult(
            name="row_count",
            status=CheckStatus.PASS,
            message="Before DataFrame was empty — nothing to check.",
        )

    dropped_pct = (n_before - n_after) / n_before

    if dropped_pct > ROW_DROP_CRITICAL_PCT:
        return CheckResult(
            name="row_count",
            status=CheckStatus.FAIL,
            message=(
                f"CRITICAL: {n_before - n_after} rows dropped "
                f"({dropped_pct:.1%} of {n_before}). "
                f"Threshold is {ROW_DROP_CRITICAL_PCT:.0%}."
            ),
        )
    if dropped_pct > ROW_DROP_WARNING_PCT:
        return CheckResult(
            name="row_count",
            status=CheckStatus.WARNING,
            message=(
                f"WARNING: {n_before - n_after} rows dropped "
                f"({dropped_pct:.1%} of {n_before}). "
                f"Threshold is {ROW_DROP_WARNING_PCT:.0%}."
            ),
        )
    if n_after > n_before:
        return CheckResult(
            name="row_count",
            status=CheckStatus.WARNING,
            message=(
                f"WARNING: row count increased from {n_before} to {n_after}. "
                "Cleaning should not add rows."
            ),
        )

    return CheckResult(
        name="row_count",
        status=CheckStatus.PASS,
        message=f"Row count {n_before} → {n_after} ({n_before - n_after} dropped).",
    )


def _check_schema_preserved(
    before: pd.DataFrame,
    after: pd.DataFrame,
    plan: CleaningPlan | None,
) -> CheckResult:
    """All columns in before must still exist in after (unless a drop_column op removed them)."""
    dropped_by_plan: set[str] = set()
    renamed_by_plan: dict[str, str] = {}

    if plan:
        for op in plan.operations:
            if op.type.value == "drop_column":
                dropped_by_plan.update(op.column_names)
            if op.type.value == "rename_columns":
                # column_names lists the original names being renamed
                dropped_by_plan.update(op.column_names)

    expected_missing = dropped_by_plan
    actually_missing = set(before.columns) - set(after.columns) - expected_missing
    unexpected_new = set(after.columns) - set(before.columns)

    issues: list[str] = []
    if actually_missing:
        issues.append(f"columns unexpectedly removed: {sorted(actually_missing)}")
    if unexpected_new:
        # New columns added by the plan are fine — just note them
        pass

    if issues:
        return CheckResult(
            name="schema",
            status=CheckStatus.FAIL,
            message="Schema mismatch — " + "; ".join(issues),
        )

    added_note = f" ({len(unexpected_new)} new columns added)" if unexpected_new else ""
    return CheckResult(
        name="schema",
        status=CheckStatus.PASS,
        message=f"Schema preserved.{added_note}",
    )


def _check_null_expectations(
    after: pd.DataFrame,
    plan: CleaningPlan | None,
) -> list[CheckResult]:
    """For fill_missing / drop_missing ops, verify the targeted columns have no nulls."""
    if not plan:
        return []

    results: list[CheckResult] = []
    for op in plan.operations:
        if op.type.value not in ("fill_missing", "drop_missing"):
            continue
        for col in op.column_names:
            if col not in after.columns:
                continue
            remaining_nulls = int(after[col].isna().sum())
            if remaining_nulls > 0:
                results.append(CheckResult(
                    name=f"nulls_{col}",
                    status=CheckStatus.WARNING,
                    message=(
                        f"Column '{col}' still has {remaining_nulls} null(s) "
                        f"after {op.type.value} operation '{op.id}'."
                    ),
                ))
            else:
                results.append(CheckResult(
                    name=f"nulls_{col}",
                    status=CheckStatus.PASS,
                    message=f"Column '{col}' has 0 nulls after '{op.id}'.",
                ))

    return results


def _check_distribution_shift(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> list[CheckResult]:
    """Flag numeric columns where mean shifted by >MEAN_SHIFT_STD_WARNING std deviations."""
    results: list[CheckResult] = []
    numeric_cols = before.select_dtypes(include="number").columns

    for col in numeric_cols:
        if col not in after.columns:
            continue
        before_vals = before[col].dropna()
        after_vals = after[col].dropna()

        if len(before_vals) < 2 or len(after_vals) < 2:
            continue

        before_mean = float(before_vals.mean())
        before_std = float(before_vals.std())

        if before_std == 0:
            continue

        after_mean = float(after_vals.mean())
        shift = abs(after_mean - before_mean) / before_std

        if shift > MEAN_SHIFT_STD_WARNING:
            results.append(CheckResult(
                name=f"distribution_{col}",
                status=CheckStatus.WARNING,
                message=(
                    f"Column '{col}' mean shifted by {shift:.2f} std deviations "
                    f"({before_mean:.4g} → {after_mean:.4g}). "
                    f"Threshold is {MEAN_SHIFT_STD_WARNING} std."
                ),
            ))
        else:
            results.append(CheckResult(
                name=f"distribution_{col}",
                status=CheckStatus.PASS,
                message=(
                    f"Column '{col}' mean shift {shift:.2f} std deviations "
                    f"({before_mean:.4g} → {after_mean:.4g})."
                ),
            ))

    return results


def _check_duplicates(
    after: pd.DataFrame,
    plan: CleaningPlan | None,
) -> CheckResult:
    """Warn if duplicates remain and no drop_duplicates op was in the plan."""
    dup_count = int(after.duplicated().sum())

    if dup_count == 0:
        return CheckResult(
            name="duplicates",
            status=CheckStatus.PASS,
            message="No duplicate rows in cleaned DataFrame.",
        )

    # If plan explicitly includes drop_duplicates, remaining dupes are unexpected
    has_dedup_op = plan and any(
        op.type.value == "drop_duplicates" for op in plan.operations
    )

    if has_dedup_op:
        return CheckResult(
            name="duplicates",
            status=CheckStatus.WARNING,
            message=(
                f"{dup_count} duplicate row(s) remain after drop_duplicates operation."
            ),
        )

    return CheckResult(
        name="duplicates",
        status=CheckStatus.PASS,
        message=f"{dup_count} duplicate row(s) remain (no drop_duplicates in plan).",
    )


def _check_dtypes(
    before: pd.DataFrame,
    after: pd.DataFrame,
    plan: CleaningPlan | None,
) -> list[CheckResult]:
    """Flag columns whose dtype regressed (e.g. numeric became object)."""
    results: list[CheckResult] = []

    # Columns targeted by fix_dtype ops are expected to change — skip them
    fix_dtype_cols: set[str] = set()
    if plan:
        for op in plan.operations:
            if op.type.value == "fix_dtype":
                fix_dtype_cols.update(op.column_names)

    numeric_kinds = {"i", "u", "f", "c"}  # int, uint, float, complex

    for col in before.columns:
        if col not in after.columns:
            continue
        if col in fix_dtype_cols:
            continue

        before_kind = before[col].dtype.kind
        after_kind = after[col].dtype.kind

        # Flag: column was numeric before, is object now
        if before_kind in numeric_kinds and after_kind == "O":
            results.append(CheckResult(
                name=f"dtype_{col}",
                status=CheckStatus.WARNING,
                message=(
                    f"Column '{col}' dtype regressed: "
                    f"{before[col].dtype} → {after[col].dtype}."
                ),
            ))
        else:
            results.append(CheckResult(
                name=f"dtype_{col}",
                status=CheckStatus.PASS,
                message=f"Column '{col}' dtype: {before[col].dtype} → {after[col].dtype}.",
            ))

    return results
