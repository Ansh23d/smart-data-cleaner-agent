"""Tests for pipeline/profiler.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.ingestion import ingest
from pipeline.profiler import profile
from models.profile import (
    CategoricalStats,
    DataProfile,
    DatetimeStats,
    NumericStats,
    QualityFlagType,
    SemanticType,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag_types_for(p: DataProfile, col: str) -> set[QualityFlagType]:
    return {f.flag_type for f in p.quality_flags if f.column == col}


def _schema_type(p: DataProfile, col: str) -> SemanticType:
    for s in p.schema_report:
        if s.name == col:
            return s.semantic_type
    raise KeyError(col)


def _null_report_for(p: DataProfile, col: str):
    for nr in p.null_analysis:
        if nr.column == col:
            return nr
    raise KeyError(col)


# ---------------------------------------------------------------------------
# Fixture-based: messy_credit_card.csv
# 14 data rows; rows 1 and 4 (TXN001) are exact duplicates.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cc_df() -> pd.DataFrame:
    return ingest(FIXTURES / "messy_credit_card.csv")


@pytest.fixture(scope="module")
def cc_profile(cc_df) -> DataProfile:
    return profile(cc_df)


class TestNullAnalysis:
    def test_notes_null_count(self, cc_profile):
        """notes column has only 1 non-null value → 13 nulls out of 14 rows."""
        nr = _null_report_for(cc_profile, "notes")
        assert nr.null_count == 13
        assert pytest.approx(nr.null_pct, abs=0.01) == 13 / 14

    def test_notes_flagged(self, cc_profile):
        nr = _null_report_for(cc_profile, "notes")
        assert nr.flagged is True

    def test_transaction_id_no_nulls(self, cc_profile):
        nr = _null_report_for(cc_profile, "transaction_id")
        assert nr.null_count == 0
        assert nr.flagged is False

    def test_null_pct_bounds(self, cc_profile):
        for nr in cc_profile.null_analysis:
            assert 0.0 <= nr.null_pct <= 1.0

    def test_null_analysis_covers_all_columns(self, cc_df, cc_profile):
        assert {nr.column for nr in cc_profile.null_analysis} == set(cc_df.columns)


class TestDuplicateCount:
    def test_one_exact_duplicate(self, cc_profile):
        """Rows 1 and 4 (TXN001) are identical → duplicate_count == 1."""
        assert cc_profile.duplicate_count == 1

    def test_no_duplicates_on_clean(self):
        df = ingest(FIXTURES / "clean_sample.csv")
        p = profile(df)
        assert p.duplicate_count == 0

    def test_healthcare_duplicate(self):
        """P001 appears twice in messy_healthcare.csv."""
        df = ingest(FIXTURES / "messy_healthcare.csv")
        p = profile(df)
        assert p.duplicate_count == 1


class TestQualityFlags:
    def test_amount_looks_numeric(self, cc_profile):
        """amount is object dtype; ~92% of non-null values are numeric → LOOKS_NUMERIC."""
        flags = _flag_types_for(cc_profile, "amount")
        assert QualityFlagType.LOOKS_NUMERIC in flags

    def test_transaction_date_looks_datetime(self, cc_profile):
        """transaction_date is stored as strings; most values parse as dates → LOOKS_DATETIME."""
        flags = _flag_types_for(cc_profile, "transaction_date")
        assert QualityFlagType.LOOKS_DATETIME in flags

    def test_always_approved_constant(self, cc_profile):
        """always_approved has a single value 'Approved' for every row → CONSTANT_COLUMN."""
        flags = _flag_types_for(cc_profile, "always_approved")
        assert QualityFlagType.CONSTANT_COLUMN in flags

    def test_notes_high_nulls_flag(self, cc_profile):
        """notes > 90% null → HIGH_NULLS quality flag."""
        flags = _flag_types_for(cc_profile, "notes")
        assert QualityFlagType.HIGH_NULLS in flags

    def test_amount_whitespace_flag(self, cc_profile):
        """amount row 7 has ' 200.00 ' → WHITESPACE_ISSUES."""
        flags = _flag_types_for(cc_profile, "amount")
        assert QualityFlagType.WHITESPACE_ISSUES in flags

    def test_merchant_whitespace_flag(self, cc_profile):
        """merchant row 7 has ' Amazon ' → WHITESPACE_ISSUES."""
        flags = _flag_types_for(cc_profile, "merchant")
        assert QualityFlagType.WHITESPACE_ISSUES in flags

    def test_transaction_id_potential_id(self, cc_profile):
        """transaction_id is unique per row → POTENTIAL_ID flag."""
        flags = _flag_types_for(cc_profile, "transaction_id")
        assert QualityFlagType.POTENTIAL_ID in flags

    def test_no_false_constant_on_varied_column(self, cc_profile):
        flags = _flag_types_for(cc_profile, "amount")
        assert QualityFlagType.CONSTANT_COLUMN not in flags

    def test_mixed_types_flag(self):
        """Column with ~50% numeric, ~50% string → MIXED_TYPES or LOOKS_NUMERIC."""
        df = pd.DataFrame({
            "col": ["1", "2", "three", "4", "five", "6", "seven", "8", "nine", "10"]
        })
        p = profile(df)
        flags = _flag_types_for(p, "col")
        assert QualityFlagType.MIXED_TYPES in flags or QualityFlagType.LOOKS_NUMERIC in flags

    def test_no_flags_on_clean_critical(self):
        """Clean sample should have no HIGH_NULLS or MIXED_TYPES flags."""
        df = ingest(FIXTURES / "clean_sample.csv")
        p = profile(df)
        bad = [
            f for f in p.quality_flags
            if f.flag_type in (QualityFlagType.HIGH_NULLS, QualityFlagType.MIXED_TYPES)
        ]
        assert len(bad) == 0


class TestSemanticTypes:
    def test_transaction_id_is_identifier(self, cc_profile):
        assert _schema_type(cc_profile, "transaction_id") == SemanticType.IDENTIFIER

    def test_status_is_categorical(self, cc_profile):
        assert _schema_type(cc_profile, "status") == SemanticType.CATEGORICAL

    def test_transaction_date_is_datetime(self, cc_profile):
        """Most values parse as dates → DATETIME semantic type (even though stored as str)."""
        assert _schema_type(cc_profile, "transaction_date") == SemanticType.DATETIME

    def test_boolean_true_false(self):
        df = pd.DataFrame({"active": ["True", "False", "True", "True", "False"]})
        p = profile(df)
        assert _schema_type(p, "active") == SemanticType.BOOLEAN

    def test_boolean_yes_no(self):
        df = pd.DataFrame({"flag": ["yes", "no", "yes", "no", "yes"]})
        p = profile(df)
        assert _schema_type(p, "flag") == SemanticType.BOOLEAN

    def test_native_bool_dtype(self):
        df = pd.DataFrame({"ok": [True, False, True, True, False]})
        p = profile(df)
        assert _schema_type(p, "ok") == SemanticType.BOOLEAN

    def test_numeric_continuous(self):
        df = pd.DataFrame({"price": [1.1, 2.7, 3.14, 4.8, 5.23, 6.0, 7.3, 8.9, 9.01, 10.1]})
        p = profile(df)
        assert _schema_type(p, "price") == SemanticType.NUMERIC_CONTINUOUS

    def test_numeric_discrete(self):
        df = pd.DataFrame({"rating": [1, 2, 3, 4, 5, 3, 2, 1, 5, 4] * 5})
        p = profile(df)
        assert _schema_type(p, "rating") == SemanticType.NUMERIC_DISCRETE

    def test_native_datetime_dtype(self):
        df = pd.DataFrame({"ts": pd.date_range("2023-01-01", periods=5, freq="D")})
        p = profile(df)
        assert _schema_type(p, "ts") == SemanticType.DATETIME

    def test_high_cardinality_numeric_is_identifier(self):
        """Integer column where every value is unique → IDENTIFIER."""
        df = pd.DataFrame({"id": range(1000)})
        p = profile(df)
        assert _schema_type(p, "id") == SemanticType.IDENTIFIER

    def test_clean_sample_schema(self):
        df = ingest(FIXTURES / "clean_sample.csv")
        p = profile(df)
        # product_id is unique integer → IDENTIFIER
        assert _schema_type(p, "product_id") == SemanticType.IDENTIFIER
        # price is a float → NUMERIC_CONTINUOUS
        assert _schema_type(p, "price") == SemanticType.NUMERIC_CONTINUOUS
        # category has few values → CATEGORICAL
        assert _schema_type(p, "category") == SemanticType.CATEGORICAL


class TestStatistics:
    def test_numeric_stats_fields(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        p = profile(df)
        stats = p.statistics["x"]
        assert isinstance(stats, NumericStats)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(8.0)
        assert stats.mean == pytest.approx(4.5)
        assert stats.median == pytest.approx(4.5)
        assert stats.q1 is not None
        assert stats.q3 is not None
        assert stats.skewness is not None

    def test_numeric_stats_skips_nan(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
        p = profile(df)
        stats = p.statistics["x"]
        assert isinstance(stats, NumericStats)
        assert stats.mean == pytest.approx(3.0)

    def test_numeric_stats_skips_inf(self):
        import numpy as np
        df = pd.DataFrame({"x": [1.0, 2.0, np.inf, 3.0, -np.inf]})
        p = profile(df)
        stats = p.statistics["x"]
        assert isinstance(stats, NumericStats)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(3.0)

    def test_categorical_stats_fields(self):
        df = pd.DataFrame({"cat": ["A", "B", "A", "C", "B", "A"]})
        p = profile(df)
        stats = p.statistics["cat"]
        assert isinstance(stats, CategoricalStats)
        assert stats.unique_count == 3
        assert stats.top_values["A"] == 3
        assert stats.cardinality_ratio == pytest.approx(3 / 6)

    def test_categorical_top_values_capped(self):
        from app.config import TOP_CATEGORIES_COUNT
        vals = list("ABCDEFGHIJ")
        df = pd.DataFrame({"cat": vals * 10})
        p = profile(df)
        assert len(p.statistics["cat"].top_values) <= TOP_CATEGORIES_COUNT  # type: ignore[union-attr]

    def test_datetime_stats_fields(self):
        df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=5, freq="D")})
        p = profile(df)
        stats = p.statistics["dt"]
        assert isinstance(stats, DatetimeStats)
        assert "2023-01-01" in stats.min_date  # type: ignore[operator]
        assert "2023-01-05" in stats.max_date  # type: ignore[operator]

    def test_all_null_numeric_stats_are_none(self):
        df = pd.DataFrame({"x": pd.array([None, None, None], dtype="Float64")})
        p = profile(df)
        stats = p.statistics["x"]
        assert isinstance(stats, NumericStats)
        assert stats.min is None
        assert stats.mean is None

    def test_credit_limit_has_numeric_stats(self, cc_profile):
        stats = cc_profile.statistics["credit_limit"]
        assert isinstance(stats, NumericStats)
        assert stats.min == pytest.approx(1000.0)
        assert stats.max == pytest.approx(10000.0)


class TestSampleRows:
    def test_sample_row_count(self):
        from app.config import SAMPLE_ROWS_COUNT
        df = pd.DataFrame({"a": range(20), "b": range(20, 40)})
        p = profile(df)
        assert len(p.sample_rows.first) == SAMPLE_ROWS_COUNT
        assert len(p.sample_rows.last) == SAMPLE_ROWS_COUNT
        assert len(p.sample_rows.random) == SAMPLE_ROWS_COUNT

    def test_no_nan_in_sample_rows(self):
        import math
        df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0, 5.0, 6.0]})
        p = profile(df)
        for row in p.sample_rows.first + p.sample_rows.last + p.sample_rows.random:
            for v in row.values():
                if isinstance(v, float):
                    assert not math.isnan(v), f"NaN found in sample rows: {row}"

    def test_sample_rows_have_all_columns(self, cc_df, cc_profile):
        for row in cc_profile.sample_rows.first:
            assert set(row.keys()) == set(cc_df.columns)

    def test_sample_rows_when_few_rows(self):
        """Fewer rows than SAMPLE_ROWS_COUNT — should not raise."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        p = profile(df)
        assert len(p.sample_rows.first) == 3

    def test_first_rows_are_first(self):
        df = pd.DataFrame({"val": range(10)})
        p = profile(df)
        assert p.sample_rows.first[0]["val"] == 0

    def test_last_rows_are_last(self):
        df = pd.DataFrame({"val": range(10)})
        p = profile(df)
        assert p.sample_rows.last[-1]["val"] == 9


class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame({
            "a": pd.Series([], dtype="object"),
            "b": pd.Series([], dtype="float64"),
        })
        p = profile(df)
        assert p.row_count == 0
        assert p.duplicate_count == 0
        assert len(p.null_analysis) == 2
        assert all(nr.null_count == 0 for nr in p.null_analysis)

    def test_header_only_csv(self):
        df = ingest(FIXTURES / "edge_empty.csv")
        p = profile(df)
        assert p.row_count == 0
        assert p.column_count == 3

    def test_single_row(self):
        df = pd.DataFrame({"x": [42], "y": ["hello"]})
        p = profile(df)
        assert p.row_count == 1
        assert p.duplicate_count == 0

    def test_all_null_column_flagged(self):
        df = pd.DataFrame({"ok": [1, 2, 3], "empty": [None, None, None]})
        p = profile(df)
        nr = _null_report_for(p, "empty")
        assert nr.flagged is True
        assert QualityFlagType.HIGH_NULLS in _flag_types_for(p, "empty")

    def test_constant_numeric_column(self):
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
        p = profile(df)
        assert QualityFlagType.CONSTANT_COLUMN in _flag_types_for(p, "x")

    def test_messy_ecommerce_loads_and_profiles(self):
        df = ingest(FIXTURES / "messy_ecommerce.csv")
        p = profile(df)
        assert p.row_count == 12
        assert len(p.quality_flags) > 0


class TestSerialization:
    def test_profile_serializes_to_json(self, cc_profile):
        import json
        json_str = cc_profile.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["row_count"] == cc_profile.row_count
        assert "schema_report" in parsed
        assert "quality_flags" in parsed
        assert "statistics" in parsed

    def test_profile_round_trip(self, cc_profile):
        data = cc_profile.model_dump()
        rebuilt = DataProfile.model_validate(data)
        assert rebuilt.row_count == cc_profile.row_count
        assert rebuilt.duplicate_count == cc_profile.duplicate_count
        assert len(rebuilt.schema_report) == len(cc_profile.schema_report)
        assert len(rebuilt.quality_flags) == len(cc_profile.quality_flags)
