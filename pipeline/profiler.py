"""Data profiler — takes a pandas DataFrame and returns a DataProfile Pydantic model.

Performs no I/O and makes no LLM calls. All computation is deterministic.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CONSTANT_COLUMN_UNIQUE,
    HIGH_CARDINALITY_RATIO,
    NULL_FLAG_THRESHOLD_PCT,
    SAMPLE_ROWS_COUNT,
    TOP_CATEGORIES_COUNT,
)
from models.profile import (
    CategoricalStats,
    ColumnSchema,
    DataProfile,
    DatetimeStats,
    NullReport,
    NumericStats,
    QualityFlag,
    QualityFlagType,
    SampleRows,
    SemanticType,
)

# ---------------------------------------------------------------------------
# Module-level thresholds (not user-configurable; internal profiler logic)
# ---------------------------------------------------------------------------
_BOOL_WORD_VALUES = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f"})
_DATETIME_INFER_THRESHOLD = 0.50   # fraction of non-null that must parse as date
_LOOKS_DATETIME_FLAG_THRESHOLD = 0.70   # to raise LOOKS_DATETIME quality flag
_LOOKS_NUMERIC_THRESHOLD = 0.80    # fraction numeric → LOOKS_NUMERIC flag
_MIXED_TYPES_LOWER = 0.10          # lower bound for MIXED_TYPES flag
_LONG_STRING_AVG_LEN = 50          # avg char len > this → TEXT semantic type
_DISCRETE_MAX_UNIQUE_ABS = 20      # absolute cap for NUMERIC_DISCRETE
_DISCRETE_MAX_UNIQUE_FRAC = 0.05   # fraction cap: unique/n ≤ this → discrete candidate
_DATETIME_SAMPLE_SIZE = 200        # max rows sampled for date inference on object cols
_HIGH_CARDINALITY_FLAG_LOWER = 0.50  # cardinality between this and HIGH_CARDINALITY_RATIO → HIGH_CARDINALITY flag
# Column-name patterns that strongly suggest an identifier column
_ID_COL_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|_key$|^key$)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile(df: pd.DataFrame) -> DataProfile:
    """Profile *df* and return a fully-populated DataProfile.

    Parameters
    ----------
    df:
        Any pandas DataFrame, including empty (0 rows) DataFrames.
    """
    n = len(df)

    schema_report = [_infer_schema(df[col], col, n) for col in df.columns]
    sem_map: dict[str, SemanticType] = {s.name: s.semantic_type for s in schema_report}

    null_analysis = [_null_report(df[col], col, n) for col in df.columns]
    duplicate_count = int(df.duplicated().sum())

    statistics: dict[str, NumericStats | CategoricalStats | DatetimeStats] = {
        col: _compute_stats(df[col], sem_map[col], n)
        for col in df.columns
    }

    quality_flags = _build_quality_flags(df, null_analysis, n)
    sample_rows = _build_sample_rows(df, n)

    return DataProfile(
        row_count=n,
        column_count=len(df.columns),
        schema_report=schema_report,
        null_analysis=null_analysis,
        duplicate_count=duplicate_count,
        statistics=statistics,
        quality_flags=quality_flags,
        sample_rows=sample_rows,
    )


# ---------------------------------------------------------------------------
# Schema — semantic type inference
# ---------------------------------------------------------------------------

def _infer_schema(series: pd.Series, col: str, n: int) -> ColumnSchema:
    sem = _infer_semantic_type(series, col, n)
    return ColumnSchema(name=col, dtype=str(series.dtype), semantic_type=sem)


def _infer_semantic_type(series: pd.Series, col: str, n: int) -> SemanticType:
    dtype = series.dtype

    # Native bool
    if dtype == bool or str(dtype) == "bool":
        return SemanticType.BOOLEAN

    # Native datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return SemanticType.DATETIME

    # Numeric dtypes
    if pd.api.types.is_numeric_dtype(dtype):
        return _numeric_semantic_type(series, n)

    # Pandas categorical dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return SemanticType.CATEGORICAL

    # Object / string dtype — infer from values
    return _object_semantic_type(series, col, n)


def _numeric_semantic_type(series: pd.Series, n: int) -> SemanticType:
    non_null = series.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(non_null) == 0:
        return SemanticType.NUMERIC_CONTINUOUS

    unique_count = int(non_null.nunique())

    # Only native integer dtype columns (int8/16/32/64/uint*) can be identifiers.
    # Float columns, even with integer-valued data, represent measurements not IDs.
    is_native_int = pd.api.types.is_integer_dtype(series.dtype)
    if is_native_int and n > 0 and unique_count / n > HIGH_CARDINALITY_RATIO:
        return SemanticType.IDENTIFIER

    # Integer-valued check — used for NUMERIC_DISCRETE classification
    try:
        is_int_valued = bool(np.all(non_null == np.floor(non_null)))
    except (TypeError, ValueError):
        is_int_valued = False

    discrete_unique_cap = max(_DISCRETE_MAX_UNIQUE_ABS, int(n * _DISCRETE_MAX_UNIQUE_FRAC))
    if is_int_valued and unique_count <= discrete_unique_cap:
        return SemanticType.NUMERIC_DISCRETE

    return SemanticType.NUMERIC_CONTINUOUS


def _object_semantic_type(series: pd.Series, col: str, n: int) -> SemanticType:
    non_null = series.dropna()
    if len(non_null) == 0:
        return SemanticType.UNKNOWN

    str_values = non_null.astype(str)
    unique_count = int(non_null.nunique())
    total_non_null = len(non_null)

    # Boolean-like word values (True/False/Yes/No/Y/N …)
    lower_unique = set(str_values.str.lower().unique())
    if len(lower_unique) <= 2 and lower_unique.issubset(_BOOL_WORD_VALUES):
        return SemanticType.BOOLEAN

    # Datetime-like — sample for speed
    sample = non_null.head(_DATETIME_SAMPLE_SIZE)
    try:
        dt_parsed = pd.to_datetime(sample.astype(str), errors="coerce", format="mixed")
        dt_ratio = dt_parsed.notna().sum() / len(sample)
    except Exception:
        dt_ratio = 0.0
    if dt_ratio > _DATETIME_INFER_THRESHOLD:
        return SemanticType.DATETIME

    cardinality_ratio = unique_count / total_non_null if total_non_null > 0 else 0.0

    # Name-pattern suggests identifier (e.g. "transaction_id", "uuid") — use a lower
    # cardinality threshold since duplicate rows can push it just below HIGH_CARDINALITY_RATIO
    if _ID_COL_PATTERN.search(col) and cardinality_ratio > 0.5:
        return SemanticType.IDENTIFIER

    # Pure cardinality — even without a suggestive name
    if cardinality_ratio > HIGH_CARDINALITY_RATIO:
        return SemanticType.IDENTIFIER

    # Long average string → free text
    avg_len = float(str_values.str.len().mean())
    if avg_len > _LONG_STRING_AVG_LEN:
        return SemanticType.TEXT

    return SemanticType.CATEGORICAL


# ---------------------------------------------------------------------------
# Null analysis
# ---------------------------------------------------------------------------

def _null_report(series: pd.Series, col: str, n: int) -> NullReport:
    null_count = int(series.isna().sum())
    null_pct = null_count / n if n > 0 else 0.0
    flagged = null_pct > NULL_FLAG_THRESHOLD_PCT
    return NullReport(column=col, null_count=null_count, null_pct=null_pct, flagged=flagged)


# ---------------------------------------------------------------------------
# Statistics dispatch
# ---------------------------------------------------------------------------

def _compute_stats(
    series: pd.Series,
    sem_type: SemanticType,
    n: int,
) -> NumericStats | CategoricalStats | DatetimeStats:
    if sem_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE):
        return _numeric_stats(series)
    if sem_type == SemanticType.DATETIME:
        return _datetime_stats(series)
    # CATEGORICAL, BOOLEAN, TEXT, IDENTIFIER, UNKNOWN
    return _categorical_stats(series, n)


def _numeric_stats(series: pd.Series) -> NumericStats:
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(cleaned) == 0:
        return NumericStats(
            min=None, max=None, mean=None, median=None,
            std=None, q1=None, q3=None, skewness=None,
        )
    return NumericStats(
        min=_f(cleaned.min()),
        max=_f(cleaned.max()),
        mean=_f(cleaned.mean()),
        median=_f(cleaned.median()),
        std=_f(cleaned.std()),
        q1=_f(cleaned.quantile(0.25)),
        q3=_f(cleaned.quantile(0.75)),
        skewness=_f(cleaned.skew()) if len(cleaned) > 2 else None,
    )


def _categorical_stats(series: pd.Series, n: int) -> CategoricalStats:
    non_null = series.dropna()
    total_non_null = len(non_null)
    if total_non_null == 0:
        return CategoricalStats(unique_count=0, cardinality_ratio=0.0, top_values={})

    unique_count = int(non_null.nunique())
    cardinality_ratio = unique_count / total_non_null

    top = (
        non_null.astype(str)
        .value_counts()
        .head(TOP_CATEGORIES_COUNT)
        .to_dict()
    )
    top_values: dict[str, int] = {str(k): int(v) for k, v in top.items()}

    return CategoricalStats(
        unique_count=unique_count,
        cardinality_ratio=cardinality_ratio,
        top_values=top_values,
    )


def _datetime_stats(series: pd.Series) -> DatetimeStats:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = series
    else:
        dt = pd.to_datetime(series.astype(str), errors="coerce")

    non_null = dt.dropna()
    if len(non_null) == 0:
        return DatetimeStats(min_date=None, max_date=None, inferred_frequency=None)

    min_date = non_null.min().isoformat()
    max_date = non_null.max().isoformat()

    freq: str | None = None
    if len(non_null) >= 4:
        try:
            freq = pd.infer_freq(non_null.sort_values())
        except Exception:
            freq = None

    return DatetimeStats(min_date=min_date, max_date=max_date, inferred_frequency=freq)


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

def _build_quality_flags(
    df: pd.DataFrame,
    null_analysis: list[NullReport],
    n: int,
) -> list[QualityFlag]:
    flags: list[QualityFlag] = []

    # HIGH_NULLS — mirror flagged NullReports as QualityFlags
    for nr in null_analysis:
        if nr.flagged:
            flags.append(QualityFlag(
                column=nr.column,
                flag_type=QualityFlagType.HIGH_NULLS,
                description=(
                    f"{nr.null_pct:.0%} of values are null "
                    f"({nr.null_count}/{n}). Consider dropping this column."
                ),
            ))

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        total_non_null = len(non_null)

        # CONSTANT_COLUMN — applies to all dtypes
        if total_non_null > 0 and int(non_null.nunique()) <= CONSTANT_COLUMN_UNIQUE:
            flags.append(QualityFlag(
                column=col,
                flag_type=QualityFlagType.CONSTANT_COLUMN,
                description=(
                    f"Column has only {non_null.nunique()} unique non-null value(s). "
                    f"Likely carries no information."
                ),
            ))
            continue  # constant → skip further checks on this column

        # POTENTIAL_ID / HIGH_CARDINALITY — mirrors the semantic-type IDENTIFIER logic
        if total_non_null > 0 and n > 0:
            unique_ratio = float(non_null.nunique()) / total_non_null
            is_id_by_name = bool(_ID_COL_PATTERN.search(col))
            is_id = (is_id_by_name and unique_ratio > 0.5) or (unique_ratio > HIGH_CARDINALITY_RATIO)
            if is_id:
                flags.append(QualityFlag(
                    column=col,
                    flag_type=QualityFlagType.POTENTIAL_ID,
                    description=(
                        f"Cardinality ratio is {unique_ratio:.2f} — nearly all values are unique. "
                        f"This may be an identifier column that can be dropped."
                    ),
                ))
            elif unique_ratio > _HIGH_CARDINALITY_FLAG_LOWER:
                flags.append(QualityFlag(
                    column=col,
                    flag_type=QualityFlagType.HIGH_CARDINALITY,
                    description=(
                        f"Cardinality ratio is {unique_ratio:.2f}. "
                        f"May be too granular for direct use as a categorical feature."
                    ),
                ))

        # Object-column-specific checks
        if not pd.api.types.is_object_dtype(series):
            continue

        if total_non_null == 0:
            continue

        str_series = non_null.astype(str)

        # WHITESPACE_ISSUES
        stripped = str_series.str.strip()
        if (str_series != stripped).any():
            flags.append(QualityFlag(
                column=col,
                flag_type=QualityFlagType.WHITESPACE_ISSUES,
                description=(
                    f"Some values have leading or trailing whitespace. "
                    f"Strip before analysis."
                ),
            ))

        # LOOKS_NUMERIC / MIXED_TYPES
        numeric_coerced = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = float(numeric_coerced.notna().sum() / total_non_null)

        if numeric_ratio >= _LOOKS_NUMERIC_THRESHOLD:
            flags.append(QualityFlag(
                column=col,
                flag_type=QualityFlagType.LOOKS_NUMERIC,
                description=(
                    f"{numeric_ratio:.0%} of non-null values parse as numeric. "
                    f"Consider casting to a numeric dtype (fix_dtype)."
                ),
            ))
        elif numeric_ratio >= _MIXED_TYPES_LOWER:
            flags.append(QualityFlag(
                column=col,
                flag_type=QualityFlagType.MIXED_TYPES,
                description=(
                    f"{numeric_ratio:.0%} of non-null values parse as numeric, "
                    f"the rest do not. Column has genuinely mixed types."
                ),
            ))

        # LOOKS_DATETIME — only if not already flagged as LOOKS_NUMERIC
        if numeric_ratio < _MIXED_TYPES_LOWER:
            sample = non_null.head(_DATETIME_SAMPLE_SIZE)
            try:
                dt_parsed = pd.to_datetime(sample.astype(str), errors="coerce", format="mixed")
                dt_ratio = float(dt_parsed.notna().sum() / len(sample))
            except Exception:
                dt_ratio = 0.0
            if dt_ratio >= _LOOKS_DATETIME_FLAG_THRESHOLD:
                flags.append(QualityFlag(
                    column=col,
                    flag_type=QualityFlagType.LOOKS_DATETIME,
                    description=(
                        f"{dt_ratio:.0%} of non-null values parse as a date/time. "
                        f"Consider casting to datetime dtype (fix_dtype)."
                    ),
                ))

    return flags


# ---------------------------------------------------------------------------
# Sample rows
# ---------------------------------------------------------------------------

def _build_sample_rows(df: pd.DataFrame, n: int) -> SampleRows:
    k = SAMPLE_ROWS_COUNT

    def to_records(sub: pd.DataFrame) -> list[dict[str, Any]]:
        return [_clean_record(r) for r in sub.to_dict(orient="records")]

    first = to_records(df.head(k))
    last = to_records(df.tail(k))

    if n > k:
        indices = random.sample(range(n), min(k, n))
        rand_rows = to_records(df.iloc[indices])
    else:
        rand_rows = to_records(df.head(k))

    return SampleRows(first=first, last=last, random=rand_rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _f(val: Any) -> float | None:
    """Convert a value to float, returning None for NaN/inf/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Replace NaN / NaT / inf with None for JSON-safe output."""
    cleaned: dict[str, Any] = {}
    for k, v in record.items():
        if v is None or v is pd.NaT:
            cleaned[k] = None
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[k] = None
        elif hasattr(v, "item"):          # numpy scalar → Python native
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned
