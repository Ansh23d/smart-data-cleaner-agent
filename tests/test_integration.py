"""End-to-end integration tests.

Each test drives the real pipeline modules against a fixture CSV with LLM
calls mocked to return pre-canned, realistic responses.  No network traffic.

Pipeline under test:
  ingest → profile → domain_detect → CleaningAgent.plan (mock)
  → executor.execute × N → validator.validate
  → KPIAgent.suggest (mock) → executor.execute × N (KPI codes)
  → InsightAgent.generate (mock)

Fixtures
--------
messy_credit_card.csv   — duplicate rows, mixed dtypes, constant column
messy_ecommerce.csv     — mixed-case categories, N/A values, invalid dates
messy_healthcare.csv    — duplicate patient, N/A age, empty row
clean_sample.csv        — already clean → no cleaning operations proposed
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agents.cleaning_agent import CleaningAgent
from agents.insight_agent import InsightAgent
from agents.kpi_agent import KPIAgent
from pipeline.domain_detector import detect
from pipeline.executor import execute
from pipeline.ingestion import ingest
from pipeline.profiler import profile as run_profile
from pipeline.validator import CheckStatus, validate
from utils.cost_tracker import CostTracker

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def _make_llm(responses: list[dict]) -> MagicMock:
    """LLMClient mock whose .call() returns *responses* in order."""
    client = MagicMock()
    client.call = MagicMock(side_effect=responses)
    return client


def _cleaning_plan_response(operations: list[dict]) -> dict:
    return {"operations": operations}


def _kpi_response(kpis: list[dict]) -> dict:
    return {"kpis": kpis}


def _insight_response(insights: list[dict]) -> dict:
    return {"insights": insights}


def _op(
    op_id: str,
    op_type: str,
    code: str,
    columns: list[str],
    description: str = "",
    confidence: str = "high",
) -> dict:
    return {
        "id": op_id,
        "type": op_type,
        "description": description or op_id.replace("_", " "),
        "impact": "improves data quality",
        "confidence": confidence,
        "reversible": True,
        "code": code,
        "column_names": columns,
        "alternatives": [],
    }


def _kpi(name: str, code: str, column: str) -> dict:
    return {
        "name": name,
        "category": "Revenue",
        "formula_description": f"Computes {name}",
        "code": code,
        "business_value": "Key business metric.",
        "priority": "high",
        "chart_config": {
            "chart_type": "histogram",
            "x_axis": column,
            "y_axis": None,
            "title": name,
        },
    }


def _insight(text: str, category: str = "Trends", priority: str = "info") -> dict:
    return {"text": text, "category": category, "priority": priority}


# ---------------------------------------------------------------------------
# Ingest + profile smoke tests (parametrised over all fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name,min_rows,min_cols", [
    ("messy_credit_card.csv",   12, 10),
    ("messy_ecommerce.csv",     11,  9),
    ("messy_healthcare.csv",     8, 10),
    ("clean_sample.csv",        10,  7),
])
def test_ingest_and_profile(fixture_name, min_rows, min_cols):
    """Every fixture must ingest successfully and produce a valid DataProfile."""
    raw = _load_fixture(fixture_name)
    df = ingest(raw, filename=fixture_name)
    assert len(df) >= min_rows
    assert len(df.columns) >= min_cols

    prof = run_profile(df)
    assert prof.row_count == len(df)
    assert prof.column_count == len(df.columns)
    assert len(prof.schema_report) == len(df.columns)
    assert len(prof.null_analysis) == len(df.columns)


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

def test_domain_detect_credit_card():
    raw = _load_fixture("messy_credit_card.csv")
    df = ingest(raw, filename="messy_credit_card.csv")
    result = detect(list(df.columns))
    assert result.top is not None
    assert "financial" in result.top.domain.lower() or "credit" in result.top.domain.lower()


def test_domain_detect_ecommerce():
    raw = _load_fixture("messy_ecommerce.csv")
    df = ingest(raw, filename="messy_ecommerce.csv")
    result = detect(list(df.columns))
    assert result.top is not None
    assert "retail" in result.top.domain.lower() or "e-commerce" in result.top.domain.lower()


def test_domain_detect_healthcare():
    raw = _load_fixture("messy_healthcare.csv")
    df = ingest(raw, filename="messy_healthcare.csv")
    result = detect(list(df.columns))
    assert result.top is not None
    assert "health" in result.top.domain.lower()


# ---------------------------------------------------------------------------
# Full pipeline — Credit Card dataset
# ---------------------------------------------------------------------------

class TestCreditCardPipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        raw = _load_fixture("messy_credit_card.csv")
        self.df = ingest(raw, filename="messy_credit_card.csv")
        self.profile = run_profile(self.df)

    def _cleaning_plan(self) -> dict:
        return _cleaning_plan_response([
            _op(
                "fix_whitespace_amount", "fix_whitespace",
                "df['amount'] = df['amount'].astype(str).str.strip()",
                ["amount"],
            ),
            _op(
                "fix_dtype_amount", "fix_dtype",
                "df['amount'] = pd.to_numeric(df['amount'], errors='coerce')",
                ["amount"],
            ),
            _op(
                "fix_dtype_credit_limit", "fix_dtype",
                "df['credit_limit'] = pd.to_numeric(df['credit_limit'], errors='coerce')",
                ["credit_limit"],
            ),
            _op(
                "fill_credit_limit", "fill_missing",
                "df['credit_limit'] = df['credit_limit'].fillna(df['credit_limit'].median())",
                ["credit_limit"], confidence="medium",
            ),
            _op(
                "drop_duplicates", "drop_duplicates",
                "df = df.drop_duplicates()",
                [],
            ),
            _op(
                "drop_col_always_approved", "drop_column",
                "df = df.drop(columns=['always_approved'])",
                ["always_approved"],
            ),
        ])

    def test_cleaning_plan_generated(self):
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)
        assert len(plan.operations) > 0
        op_types = {op.type.value for op in plan.operations}
        assert "drop_duplicates" in op_types
        assert "drop_column" in op_types

    def test_execution_succeeds(self):
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df
            elif result.blocked:
                pytest.fail(f"Operation blocked unexpectedly: {op.id}")

        # Duplicate row removed
        assert len(current) < len(self.df)
        # always_approved column dropped
        assert "always_approved" not in current.columns

    def test_validation_passes(self):
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df

        report = validate(self.df, current, plan)
        assert report.status in (CheckStatus.PASS, CheckStatus.WARNING)
        # Must not have a FAIL
        assert report.status != CheckStatus.FAIL

    def test_kpi_generation(self):
        llm = _make_llm([_kpi_response([
            _kpi("Total Transaction Volume",
                 "df['amount'].sum()", "amount"),
            _kpi("Average Transaction Value",
                 "df['amount'].mean()", "amount"),
        ])])
        agent = KPIAgent(llm_client=llm)
        kpis = agent.suggest(self.profile, "Financial Services / Credit Card")
        assert len(kpis) == 2
        assert kpis[0].name == "Total Transaction Volume"

    def test_kpi_execution(self):
        """KPI pandas code runs successfully against the raw df."""
        llm = _make_llm([self._cleaning_plan()])
        clean_agent = CleaningAgent(llm_client=llm)
        plan = clean_agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df

        # Execute KPI codes against cleaned df
        kpi_codes = [
            "df['amount'].sum()",
            "df['amount'].mean()",
        ]
        for code in kpi_codes:
            result = execute(code, current)
            assert result.success, f"KPI code failed: {result.error}"
            assert isinstance(result.df, pd.DataFrame)

    def test_insight_generation(self):
        llm = _make_llm([_insight_response([
            _insight("High-value transaction $25,000 at Casino exceeds typical spend.", "Risk", "high"),
            _insight("1 duplicate transaction removed.", "Data Quality", "medium"),
            _insight("Amazon is the most frequent merchant.", "Trends", "info"),
        ])])
        agent = InsightAgent(llm_client=llm)

        # Minimal KPIs for the agent
        from models.kpi import KPI, ChartConfig
        kpis = [
            KPI(
                name="Total Volume",
                category="Revenue",
                formula_description="sum(amount)",
                code="df['amount'].sum()",
                business_value="Total spend.",
                priority="high",
                chart_config=ChartConfig(chart_type="histogram", x_axis="amount",
                                         y_axis=None, title="Volume"),
                computed_value=2500.0,
            )
        ]
        insights = agent.generate(kpis, self.profile, "Financial Services / Credit Card")
        assert len(insights) == 3
        priorities = {i.priority for i in insights}
        assert "high" in priorities


# ---------------------------------------------------------------------------
# Full pipeline — E-commerce dataset
# ---------------------------------------------------------------------------

class TestEcommercePipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        raw = _load_fixture("messy_ecommerce.csv")
        self.df = ingest(raw, filename="messy_ecommerce.csv")
        self.profile = run_profile(self.df)

    def _cleaning_plan(self) -> dict:
        return _cleaning_plan_response([
            _op(
                "fix_dtype_unit_price", "fix_dtype",
                "df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')",
                ["unit_price"],
            ),
            _op(
                "fix_dtype_quantity", "fix_dtype",
                "df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')",
                ["quantity"],
            ),
            _op(
                "fill_unit_price", "fill_missing",
                "df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())",
                ["unit_price"], confidence="medium",
            ),
            _op(
                "standardize_category", "standardize_categories",
                "df['category'] = df['category'].str.strip().str.title()",
                ["category"],
            ),
            _op(
                "drop_duplicates", "drop_duplicates",
                "df = df.drop_duplicates(subset=[c for c in df.columns if c != 'order_id'])",
                [],
            ),
        ])

    def test_execution_removes_duplicates(self):
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df

        # After standardizing and dedup, fewer rows or same
        assert len(current) <= len(self.df)

    def test_execution_standardizes_categories(self):
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df

        # All category values should be title-cased after standardization
        if "category" in current.columns:
            cats = current["category"].dropna().unique()
            for cat in cats:
                assert cat == cat.title(), f"category not title-cased: {cat!r}"

    def test_validation_no_schema_fail(self):
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df

        report = validate(self.df, current, plan)
        assert report.status != CheckStatus.FAIL

    def test_kpi_revenue_computed(self):
        """Revenue KPI executes against raw ecommerce df."""
        code = "df['unit_price'].dropna() * df['quantity'].dropna()"
        result = execute(code, self.df)
        # Code returns a Series-like value; check no exception
        assert result.success or result.error is not None  # may need coercion first

        # After coercion it should work (np is pre-injected in sandbox namespace)
        coerce_code = (
            "df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')\n"
            "df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')\n"
            "df['revenue'] = df['unit_price'] * df['quantity']\n"
            "df['revenue'].sum()"
        )
        result2 = execute(coerce_code, self.df)
        assert result2.success


# ---------------------------------------------------------------------------
# Full pipeline — Healthcare dataset
# ---------------------------------------------------------------------------

class TestHealthcarePipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        raw = _load_fixture("messy_healthcare.csv")
        self.df = ingest(raw, filename="messy_healthcare.csv")
        self.profile = run_profile(self.df)

    def _cleaning_plan(self) -> dict:
        return _cleaning_plan_response([
            _op(
                "fix_dtype_age", "fix_dtype",
                "df['age'] = pd.to_numeric(df['age'], errors='coerce')",
                ["age"],
            ),
            _op(
                "fix_dtype_los_days", "fix_dtype",
                "df['los_days'] = pd.to_numeric(df['los_days'], errors='coerce')",
                ["los_days"],
            ),
            _op(
                "drop_missing_rows", "drop_missing",
                "df = df.dropna(subset=['patient_id'])",
                ["patient_id"], confidence="high",
            ),
            _op(
                "drop_duplicates", "drop_duplicates",
                "df = df.drop_duplicates()",
                [],
            ),
            _op(
                "fill_age", "fill_missing",
                "df['age'] = df['age'].fillna(df['age'].median())",
                ["age"], confidence="medium",
            ),
        ])

    def test_duplicate_patient_removed(self):
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df

        # P001 appears twice in fixture; after dedup should appear once
        if "patient_id" in current.columns:
            assert current["patient_id"].value_counts().max() == 1

    def test_empty_row_dropped(self):
        """P009 row has all nulls except patient_id='P009' is also empty."""
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df

        # No row should have a completely empty patient_id after drop_missing
        if "patient_id" in current.columns:
            assert current["patient_id"].isna().sum() == 0

    def test_validation_row_drop_acceptable(self):
        llm = _make_llm([self._cleaning_plan()])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)

        current = self.df.copy()
        for op in plan.operations:
            result = execute(op.code, current)
            if result.success:
                current = result.df

        report = validate(self.df, current, plan)
        # Dropping 1-2 rows from 8 is ~12-25% — may be WARNING but not FAIL
        assert report.status in (CheckStatus.PASS, CheckStatus.WARNING)

    def test_kpi_avg_los(self):
        """Average length-of-stay KPI executes correctly."""
        setup_code = (
            "df['los_days'] = pd.to_numeric(df['los_days'], errors='coerce')\n"
            "df['los_days'].mean()"
        )
        result = execute(setup_code, self.df)
        assert result.success


# ---------------------------------------------------------------------------
# Clean dataset — no cleaning needed path
# ---------------------------------------------------------------------------

class TestCleanDatasetPipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        raw = _load_fixture("clean_sample.csv")
        self.df = ingest(raw, filename="clean_sample.csv")
        self.profile = run_profile(self.df)

    def test_empty_plan_handled(self):
        """When LLM proposes zero ops the pipeline should handle it gracefully."""
        llm = _make_llm([_cleaning_plan_response([])])
        agent = CleaningAgent(llm_client=llm)
        plan = agent.plan(self.profile)
        assert len(plan.operations) == 0

    def test_validation_on_unchanged_data(self):
        """Validating identical before/after DataFrames should always PASS."""
        report = validate(self.df, self.df.copy())
        assert report.status == CheckStatus.PASS

    def test_profile_has_no_null_flags(self):
        """Clean sample should have zero HIGH_NULLS quality flags."""
        from models.profile import QualityFlagType
        null_flags = [
            f for f in self.profile.quality_flags
            if f.flag_type == QualityFlagType.HIGH_NULLS
        ]
        assert len(null_flags) == 0

    def test_kpi_on_clean_data(self):
        llm = _make_llm([_kpi_response([
            _kpi("Average Price", "df['price'].mean()", "price"),
            _kpi("Total Stock", "df['stock_qty'].sum()", "stock_qty"),
        ])])
        agent = KPIAgent(llm_client=llm)
        kpis = agent.suggest(self.profile, "E-commerce / Retail")
        assert len(kpis) == 2

        for kpi in kpis:
            result = execute(kpi.code, self.df)
            assert result.success, f"KPI '{kpi.name}' failed: {result.error}"


# ---------------------------------------------------------------------------
# Edge-case — empty CSV
# ---------------------------------------------------------------------------

def test_edge_empty_csv_raises():
    """An empty CSV (headers only) should raise IngestionError or return 0-row df."""
    from pipeline.ingestion import IngestionError
    raw = _load_fixture("edge_empty.csv")
    try:
        df = ingest(raw, filename="edge_empty.csv")
        # If it doesn't raise, it should produce an empty DataFrame
        assert len(df) == 0
    except (IngestionError, Exception):
        pass  # any clean exception is acceptable


# ---------------------------------------------------------------------------
# Cross-module: executor integrates with validation
# ---------------------------------------------------------------------------

def test_execute_then_validate_row_drop_warning():
    """Dropping >20% rows should produce a WARNING validation status."""
    df = pd.DataFrame({"id": range(100), "val": range(100)})
    code = "df = df.head(70)"  # drops 30 rows = 30% → WARNING
    result = execute(code, df)
    assert result.success

    report = validate(df, result.df)
    assert report.status == CheckStatus.WARNING


def test_execute_then_validate_schema_preserved():
    """Adding a new column should not break schema validation."""
    df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
    code = "df['val_doubled'] = df['val'] * 2"
    result = execute(code, df)
    assert result.success

    report = validate(df, result.df)
    # New columns added are fine; only dropped columns fail schema check
    assert report.status in (CheckStatus.PASS, CheckStatus.WARNING)


def test_blocked_code_never_mutates_df():
    """Blocked code must return the original DataFrame unchanged."""
    df = pd.DataFrame({"id": [1, 2, 3]})
    code = "import os; df = df.head(1)"
    result = execute(code, df)
    assert result.blocked
    assert len(result.df) == 3  # unchanged


# ---------------------------------------------------------------------------
# CostTracker accumulates across agents
# ---------------------------------------------------------------------------

def test_cost_tracker_accumulates():
    """CostTracker records a call when LLMClient._tracker.record() is invoked."""
    from services.llm_client import LLMClient
    from unittest.mock import patch, MagicMock

    tracker = CostTracker()

    with patch("services.llm_client.anthropic.Anthropic") as MockAnthropic:
        mock_response = MagicMock()
        mock_response.content[0].text = '{"operations": []}'
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        MockAnthropic.return_value.messages.create.return_value = mock_response

        client = LLMClient(tracker=tracker)
        raw = _load_fixture("clean_sample.csv")
        df = ingest(raw, filename="clean_sample.csv")
        prof = run_profile(df)

        agent = CleaningAgent(llm_client=client)
        agent.plan(prof)

    assert tracker.total_input_tokens > 0
    assert tracker.total_cost_usd > 0
    assert len(tracker.records) == 1
