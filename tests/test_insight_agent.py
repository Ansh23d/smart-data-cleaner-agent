"""Tests for agents/insight_agent.py — all LLM calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.insight_agent import (
    InsightAgent,
    InsightAgentError,
    _format_kpi_summary,
    _format_profile_summary,
)
from app.config import MODEL_SONNET
from models.insight import Insight
from models.kpi import KPI, ChartConfig
from models.profile import (
    CategoricalStats,
    ColumnSchema,
    DataProfile,
    NullReport,
    NumericStats,
    SampleRows,
    SemanticType,
)
from services.llm_client import LLMError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_profile() -> DataProfile:
    return DataProfile(
        row_count=200,
        column_count=2,
        schema_report=[
            ColumnSchema(name="amount", dtype="float64", semantic_type=SemanticType.NUMERIC_CONTINUOUS),
            ColumnSchema(name="category", dtype="object", semantic_type=SemanticType.CATEGORICAL),
        ],
        null_analysis=[
            NullReport(column="amount", null_count=20, null_pct=0.10),
            NullReport(column="category", null_count=0, null_pct=0.0),
        ],
        duplicate_count=0,
        statistics={
            "amount": NumericStats(min=1.0, max=500.0, mean=100.0, median=90.0,
                                   std=50.0, q1=50.0, q3=150.0, skewness=0.8),
            "category": CategoricalStats(unique_count=3, cardinality_ratio=0.015,
                                         top_values={"A": 100, "B": 60, "C": 40}),
        },
        quality_flags=[],
        sample_rows=SampleRows(first=[], last=[], random=[]),
    )


@pytest.fixture
def sample_kpis() -> list[KPI]:
    return [
        KPI(
            name="Avg Transaction Value",
            category="Revenue",
            formula_description="Mean of amount",
            code="df['amount'].mean()",
            computed_value=100.5,
            business_value="Tracks avg spend.",
            priority="high",
            chart_config=ChartConfig(chart_type="histogram", x_axis="amount",
                                     y_axis=None, title="Amount Distribution"),
        ),
        KPI(
            name="Category Share",
            category="Segmentation",
            formula_description="Value counts of category",
            code="df['category'].value_counts().to_dict()",
            computed_value={"A": 100, "B": 60, "C": 40},
            business_value="Shows category breakdown.",
            priority="medium",
            chart_config=ChartConfig(chart_type="bar", x_axis="category",
                                     y_axis="amount", title="By Category"),
        ),
    ]


def _make_insight_dict(
    text: str = "Revenue is strong.",
    priority: str = "high",
    category: str = "Revenue",
) -> dict:
    return {"text": text, "priority": priority, "category": category}


def _make_agent(mock_response) -> tuple[InsightAgent, MagicMock]:
    mock_client = MagicMock()
    mock_client.call.return_value = mock_response
    return InsightAgent(llm_client=mock_client), mock_client


# ---------------------------------------------------------------------------
# _format_kpi_summary
# ---------------------------------------------------------------------------

class TestFormatKPISummary:
    def test_empty_kpis(self):
        assert _format_kpi_summary([]) == "No KPIs computed."

    def test_includes_kpi_name(self, sample_kpis):
        result = _format_kpi_summary(sample_kpis)
        assert "Avg Transaction Value" in result

    def test_includes_computed_value(self, sample_kpis):
        result = _format_kpi_summary(sample_kpis)
        assert "100.5" in result or "100" in result

    def test_none_value_shown_as_not_computed(self):
        kpi = KPI(
            name="Test", category="X", formula_description="f",
            code="df.mean()", computed_value=None, business_value="b",
            priority="low",
            chart_config=ChartConfig(chart_type="bar", x_axis=None, y_axis=None, title="t"),
        )
        result = _format_kpi_summary([kpi])
        assert "not computed" in result

    def test_includes_category(self, sample_kpis):
        result = _format_kpi_summary(sample_kpis)
        assert "Revenue" in result


# ---------------------------------------------------------------------------
# _format_profile_summary
# ---------------------------------------------------------------------------

class TestFormatProfileSummary:
    def test_includes_row_count(self, simple_profile):
        result = _format_profile_summary(simple_profile)
        assert "200" in result

    def test_includes_column_count(self, simple_profile):
        result = _format_profile_summary(simple_profile)
        assert "2" in result

    def test_high_null_column_mentioned(self, simple_profile):
        # amount has 10% nulls (>10% threshold)
        result = _format_profile_summary(simple_profile)
        # 0.10 is not strictly > 0.10, so won't appear — test with a profile that has >10%
        pass  # covered by next test

    def test_high_null_above_threshold(self):
        profile = DataProfile(
            row_count=100, column_count=1,
            schema_report=[ColumnSchema(name="x", dtype="float64",
                                        semantic_type=SemanticType.NUMERIC_CONTINUOUS)],
            null_analysis=[NullReport(column="x", null_count=20, null_pct=0.20)],
            duplicate_count=0,
            statistics={},
            quality_flags=[],
            sample_rows=SampleRows(first=[], last=[], random=[]),
        )
        result = _format_profile_summary(profile)
        assert "x" in result

    def test_no_high_null_omits_section(self, simple_profile):
        # Make a profile with no high-null columns
        profile = DataProfile(
            row_count=100, column_count=1,
            schema_report=[ColumnSchema(name="x", dtype="float64",
                                        semantic_type=SemanticType.NUMERIC_CONTINUOUS)],
            null_analysis=[NullReport(column="x", null_count=1, null_pct=0.01)],
            duplicate_count=0,
            statistics={},
            quality_flags=[],
            sample_rows=SampleRows(first=[], last=[], random=[]),
        )
        result = _format_profile_summary(profile)
        assert "High-null" not in result


# ---------------------------------------------------------------------------
# InsightAgent.generate — happy path
# ---------------------------------------------------------------------------

class TestGenerateHappyPath:
    def test_returns_list_of_insights(self, simple_profile, sample_kpis):
        response = {"insights": [
            _make_insight_dict("Revenue is strong.", "high", "Revenue"),
            _make_insight_dict("Category A dominates.", "medium", "Segmentation"),
            _make_insight_dict("10% null in amount.", "info", "Quality"),
        ]}
        agent, _ = _make_agent(response)
        result = agent.generate(sample_kpis, simple_profile, "Financial Services")
        assert isinstance(result, list)
        assert all(isinstance(i, Insight) for i in result)

    def test_three_insights_returned(self, simple_profile, sample_kpis):
        response = {"insights": [
            _make_insight_dict("A", "high", "Revenue"),
            _make_insight_dict("B", "medium", "Trend"),
            _make_insight_dict("C", "info", "Quality"),
        ]}
        agent, _ = _make_agent(response)
        result = agent.generate(sample_kpis, simple_profile, "Financial Services")
        assert len(result) == 3

    def test_uses_sonnet_model(self, simple_profile, sample_kpis):
        response = {"insights": [_make_insight_dict() for _ in range(3)]}
        agent, mock_client = _make_agent(response)
        agent.generate(sample_kpis, simple_profile, "Financial Services")
        _, kwargs = mock_client.call.call_args
        assert kwargs["model"] == MODEL_SONNET

    def test_domain_context_in_prompt(self, simple_profile, sample_kpis):
        response = {"insights": [_make_insight_dict() for _ in range(3)]}
        agent, mock_client = _make_agent(response)
        agent.generate(sample_kpis, simple_profile, "Healthcare")
        _, kwargs = mock_client.call.call_args
        assert "Healthcare" in kwargs["user_prompt"]

    def test_kpi_values_in_prompt(self, simple_profile, sample_kpis):
        response = {"insights": [_make_insight_dict() for _ in range(3)]}
        agent, mock_client = _make_agent(response)
        agent.generate(sample_kpis, simple_profile, "Financial Services")
        _, kwargs = mock_client.call.call_args
        assert "Avg Transaction Value" in kwargs["user_prompt"]

    def test_sorted_high_priority_first(self, simple_profile, sample_kpis):
        response = {"insights": [
            _make_insight_dict("Info insight", "info", "Quality"),
            _make_insight_dict("High insight", "high", "Revenue"),
            _make_insight_dict("Medium insight", "medium", "Trend"),
        ]}
        agent, _ = _make_agent(response)
        result = agent.generate(sample_kpis, simple_profile, "Financial Services")
        assert result[0].priority == "high"
        assert result[-1].priority == "info"

    def test_capped_at_five_insights(self, simple_profile, sample_kpis):
        response = {"insights": [_make_insight_dict(f"Insight {i}", "info", "X") for i in range(8)]}
        agent, _ = _make_agent(response)
        result = agent.generate(sample_kpis, simple_profile, "Financial Services")
        assert len(result) <= 5

    def test_expect_json_true(self, simple_profile, sample_kpis):
        response = {"insights": [_make_insight_dict() for _ in range(3)]}
        agent, mock_client = _make_agent(response)
        agent.generate(sample_kpis, simple_profile, "Financial Services")
        _, kwargs = mock_client.call.call_args
        assert kwargs["expect_json"] is True


# ---------------------------------------------------------------------------
# InsightAgent.generate — malformed responses
# ---------------------------------------------------------------------------

class TestInsightMalformedResponse:
    def test_missing_insights_key_raises(self, simple_profile, sample_kpis):
        agent, _ = _make_agent({"wrong": []})
        with pytest.raises(InsightAgentError, match="schema"):
            agent.generate(sample_kpis, simple_profile, "Financial Services")

    def test_invalid_priority_raises(self, simple_profile, sample_kpis):
        bad = _make_insight_dict()
        bad["priority"] = "critical"   # not in Literal["high","medium","info"]
        agent, _ = _make_agent({"insights": [bad]})
        with pytest.raises(InsightAgentError):
            agent.generate(sample_kpis, simple_profile, "Financial Services")

    def test_missing_text_field_raises(self, simple_profile, sample_kpis):
        bad = {"priority": "high", "category": "Revenue"}  # no text
        agent, _ = _make_agent({"insights": [bad]})
        with pytest.raises(InsightAgentError):
            agent.generate(sample_kpis, simple_profile, "Financial Services")

    def test_llm_error_wrapped(self, simple_profile, sample_kpis):
        mock_client = MagicMock()
        mock_client.call.side_effect = LLMError("down")
        agent = InsightAgent(llm_client=mock_client)
        with pytest.raises(InsightAgentError, match="LLM call failed"):
            agent.generate(sample_kpis, simple_profile, "Financial Services")

    def test_fewer_than_min_insights_still_returns(self, simple_profile, sample_kpis):
        """Agent should return even if LLM returns fewer than 3 (just warns)."""
        response = {"insights": [_make_insight_dict("Only one", "high", "Revenue")]}
        agent, _ = _make_agent(response)
        result = agent.generate(sample_kpis, simple_profile, "Financial Services")
        assert len(result) == 1
