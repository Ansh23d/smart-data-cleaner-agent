"""Pydantic models for the DataProfile produced by pipeline/profiler.py."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SemanticType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERIC_CONTINUOUS = "numeric-continuous"
    NUMERIC_DISCRETE = "numeric-discrete"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class QualityFlagType(str, Enum):
    MIXED_TYPES = "mixed_types"
    LOOKS_NUMERIC = "looks_numeric"
    LOOKS_DATETIME = "looks_datetime"
    POTENTIAL_ID = "potential_id"
    CONSTANT_COLUMN = "constant_column"
    WHITESPACE_ISSUES = "whitespace_issues"
    HIGH_NULLS = "high_nulls"
    HIGH_CARDINALITY = "high_cardinality"


class ColumnSchema(BaseModel):
    name: str
    dtype: str                      # pandas dtype string, e.g. "object", "float64"
    semantic_type: SemanticType


class NullReport(BaseModel):
    column: str
    null_count: int
    null_pct: float = Field(ge=0.0, le=1.0)
    flagged: bool = False           # True when null_pct > NULL_FLAG_THRESHOLD_PCT


class NumericStats(BaseModel):
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    q1: float | None
    q3: float | None
    skewness: float | None


class CategoricalStats(BaseModel):
    unique_count: int
    cardinality_ratio: float        # unique / total rows
    top_values: dict[str, int]      # {value: count} for top N


class DatetimeStats(BaseModel):
    min_date: str | None            # ISO-8601 string
    max_date: str | None
    inferred_frequency: str | None  # e.g. "D", "M", "Y", or None


# Union type stored per-column — key = column name
ColumnStats = NumericStats | CategoricalStats | DatetimeStats


class QualityFlag(BaseModel):
    column: str
    flag_type: QualityFlagType
    description: str


class SampleRows(BaseModel):
    first: list[dict[str, Any]]
    last: list[dict[str, Any]]
    random: list[dict[str, Any]]


class DataProfile(BaseModel):
    """Complete profile of a DataFrame, produced by pipeline/profiler.py."""

    row_count: int
    column_count: int
    schema_report: list[ColumnSchema]
    null_analysis: list[NullReport]
    duplicate_count: int
    statistics: dict[str, NumericStats | CategoricalStats | DatetimeStats]
    quality_flags: list[QualityFlag]
    sample_rows: SampleRows
