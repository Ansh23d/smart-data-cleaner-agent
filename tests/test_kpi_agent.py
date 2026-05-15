"""Tests for agents/kpi_agent.py — all LLM calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.kpi_agent import KPIAgent, KPIAgentError, _referenced_unknown_columns
from app.config import MODEL_HAIKU, MODEL_SONNET
from models.kpi import KPI, ChartConfig
from models.profile import (
    CategoricalStats,
    ColumnSchema,
    DataProfile,
    NullReport,
    NumericStats,
    QualityFlag,
    QualityFlagType,
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
        row_count=500,
        column_count=3,
        schema_report=[
            ColumnSchema(name="amount", dtype="float64", semantic_type=SemanticType.NUMERIC_CONTINUOUS),
            ColumnSchema(name="category", dtype="object", semantic_type=SemanticType.CATEGORICAL),
            ColumnSchema(name="date", dtype="object", semantic_type=SemanticType.DATETIME),
        ],
        null_analysis=[
            NullReport(column="amount", null_count=5, null_pct=0.01),
            NullReport(column="category", null_count=0, null_pct=0.0),
            NullReport(column="date", null_count=0, null_pct=0.0),
        ],
        duplicate_count=2,
        statistics={
            "amount": NumericStats(min=1.0, max=999.0, mean=150.0, median=120.0,
                                   std=80.0, q1=60.0, q3=220.0, skewness=1.1),
            "category": CategoricalStats(unique_count=5, cardinality_ratio=0.01,
                                         top_values={"A": 200, "B": 150}),
        },
        quality_flags=[],
        sample_rows=SampleRows(first=[], last=[], random=[]),
    )


def _make_kpi_response(*kpis: dict) -> dict:
    return {"kpis": list(kpis)}


def _sample_kpi_dict(name: str = "Avg Amount", col: str = "amount") -> dict:
    return {
        "name": name,
        "category": "Revenue",
        "formula_description": "Mean of amount column",
        "code": f"df['{col}'].mean()",
        "business_value": "Tracks average spend.",
        "priority": "high",
        "chart_config": {
            "chart_type": "histogram",
            "x_axis": col,
            "y_axis": None,
            "title": f"Distribution of {col}",
        },
    }


def _make_agent(mock_response) -> tuple[KPIAgent, MagicMock]:
    mock_client = MagicMock()
    mock_client.call.return_value = mock_response
    return KPIAgent(llm_client=mock_client), mock_client


# ---------------------------------------------------------------------------
# _referenced_unknown_columns
# ---------------------------------------------------------------------------

class TestReferencedUnknownColumns:
    def test_no_references(self):
        assert _referenced_unknown_columns("df.mean()", {"amount"}) == []

    def test_known_column_not_flagged(self):
        assert _referenced_unknown_columns("df['amount'].mean()", {"amount"}) == []

    def test_unknown_column_flagged(self):
        result = _referenced_unknown_columns("df['ghost'].mean()", {"amount"})
        assert "ghost" in result

    def test_double_quote_syntax(self):
        result = _referenced_unknown_columns('df["ghost"].mean()', {"amount"})
        assert "ghost" in result

    def test_multiple_columns(self):
        code = "df['a'].sum() / df['b'].sum()"
        result = _referenced_unknown_columns(code, {"a"})
        assert "b" in result
        assert "a" not in result

    def test_all_known(self):
        code = "df['a'].mean() + df['b'].std()"
        assert _referenced_unknown_columns(code, {"a", "b"}) == []


# ---------------------------------------------------------------------------
# KPIAgent.suggest — happy path
# ---------------------------------------------------------------------------

class TestSuggestHappyPath:
    def test_returns_list_of_kpis(self, simple_profile):
        response = _make_kpi_response(_sample_kpi_dict())
        agent, _ = _make_agent(response)
        result = agent.suggest(simple_profile, "Financial Services")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], KPI)

    def test_multiple_kpis(self, simple_profile):
        response = _make_kpi_response(
            _sample_kpi_dict("Avg Amount", "amount"),
            _sample_kpi_dict("Category Count", "category"),
        )
        agent, _ = _make_agent(response)
        result = agent.suggest(simple_profile, "Financial Services")
        assert len(result) == 2

    def test_empty_kpis_list(self, simple_profile):
        agent, _ = _make_agent({"kpis": []})
        result = agent.suggest(simple_profile, "Unknown")
        assert result == []

    def test_uses_sonnet_model(self, simple_profile):
        agent, mock_client = _make_agent(_make_kpi_response(_sample_kpi_dict()))
        agent.suggest(simple_profile, "Financial Services")
        _, kwargs = mock_client.call.call_args
        assert kwargs["model"] == MODEL_SONNET

    def test_expect_json_true(self, simple_profile):
        agent, mock_client = _make_agent(_make_kpi_response(_sample_kpi_dict()))
        agent.suggest(simple_profile, "Financial Services")
        _, kwargs = mock_client.call.call_args
        assert kwargs["expect_json"] is True

    def test_domain_context_in_prompt(self, simple_profile):
        agent, mock_client = _make_agent(_make_kpi_response(_sample_kpi_dict()))
        agent.suggest(simple_profile, "E-commerce / Retail")
        _, kwargs = mock_client.call.call_args
        assert "E-commerce / Retail" in kwargs["user_prompt"]

    def test_profile_columns_in_prompt(self, simple_profile):
        agent, mock_client = _make_agent(_make_kpi_response(_sample_kpi_dict()))
        agent.suggest(simple_profile, "Financial Services")
        _, kwargs = mock_client.call.call_args
        assert "amount" in kwargs["user_prompt"]

    def test_kpi_name_preserved(self, simple_profile):
        response = _make_kpi_response(_sample_kpi_dict("Total Revenue"))
        agent, _ = _make_agent(response)
        result = agent.suggest(simple_profile, "Financial Services")
        assert result[0].name == "Total Revenue"

    def test_chart_config_parsed(self, simple_profile):
        agent, _ = _make_agent(_make_kpi_response(_sample_kpi_dict()))
        result = agent.suggest(simple_profile, "Financial Services")
        assert isinstance(result[0].chart_config, ChartConfig)


# ---------------------------------------------------------------------------
# KPIAgent.suggest — column validation
# ---------------------------------------------------------------------------

class TestKPIColumnValidation:
    def test_unknown_column_drops_kpi(self, simple_profile):
        response = _make_kpi_response(_sample_kpi_dict("Bad KPI", "ghost_col"))
        agent, _ = _make_agent(response)
        result = agent.suggest(simple_profile, "Financial Services")
        assert result == []

    def test_known_column_kept(self, simple_profile):
        response = _make_kpi_response(_sample_kpi_dict("Good KPI", "amount"))
        agent, _ = _make_agent(response)
        result = agent.suggest(simple_profile, "Financial Services")
        assert len(result) == 1

    def test_mix_valid_and_invalid(self, simple_profile):
        response = _make_kpi_response(
            _sample_kpi_dict("Good", "amount"),
            _sample_kpi_dict("Bad", "nonexistent"),
        )
        agent, _ = _make_agent(response)
        result = agent.suggest(simple_profile, "Financial Services")
        assert len(result) == 1
        assert result[0].name == "Good"

    def test_code_without_bracket_access_passes(self, simple_profile):
        """Code like df.mean() doesn't reference specific columns."""
        kpi = _sample_kpi_dict()
        kpi["code"] = "df.describe()"
        agent, _ = _make_agent({"kpis": [kpi]})
        result = agent.suggest(simple_profile, "Financial Services")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# KPIAgent.suggest — malformed responses
# ---------------------------------------------------------------------------

class TestKPIMalformedResponse:
    def test_missing_kpis_key_raises(self, simple_profile):
        agent, _ = _make_agent({"wrong": []})
        with pytest.raises(KPIAgentError, match="schema"):
            agent.suggest(simple_profile, "Financial Services")

    def test_kpi_missing_required_field_raises(self, simple_profile):
        bad = {"name": "x", "category": "Rev"}  # many fields missing
        agent, _ = _make_agent({"kpis": [bad]})
        with pytest.raises(KPIAgentError):
            agent.suggest(simple_profile, "Financial Services")

    def test_invalid_priority_raises(self, simple_profile):
        kpi = _sample_kpi_dict()
        kpi["priority"] = "critical"  # not in Literal["high","medium","low"]
        agent, _ = _make_agent({"kpis": [kpi]})
        with pytest.raises(KPIAgentError):
            agent.suggest(simple_profile, "Financial Services")

    def test_invalid_chart_type_raises(self, simple_profile):
        kpi = _sample_kpi_dict()
        kpi["chart_config"]["chart_type"] = "donut"  # not in allowed types
        agent, _ = _make_agent({"kpis": [kpi]})
        with pytest.raises(KPIAgentError):
            agent.suggest(simple_profile, "Financial Services")

    def test_llm_error_wrapped(self, simple_profile):
        mock_client = MagicMock()
        mock_client.call.side_effect = LLMError("down")
        agent = KPIAgent(llm_client=mock_client)
        with pytest.raises(KPIAgentError, match="LLM call failed"):
            agent.suggest(simple_profile, "Financial Services")


# ---------------------------------------------------------------------------
# KPIAgent.fix_kpi
# ---------------------------------------------------------------------------

class TestFixKPI:
    def _sample_kpi(self) -> KPI:
        return KPI(
            name="Avg Amount",
            category="Revenue",
            formula_description="Mean of amount",
            code="df['amount'].mean()",
            business_value="Tracks avg spend.",
            priority="high",
            chart_config=ChartConfig(
                chart_type="histogram", x_axis="amount", y_axis=None,
                title="Distribution"
            ),
        )

    def _make_fix_agent(self, code: str) -> tuple[KPIAgent, MagicMock]:
        mock_client = MagicMock()
        mock_client.call.return_value = code
        return KPIAgent(llm_client=mock_client), mock_client

    def test_returns_fixed_code(self):
        agent, _ = self._make_fix_agent("df['amount'].fillna(0).mean()")
        result = agent.fix_kpi(self._sample_kpi(), "KeyError: 'amount'")
        assert result == "df['amount'].fillna(0).mean()"

    def test_strips_whitespace(self):
        agent, _ = self._make_fix_agent("  df['amount'].mean()  \n")
        result = agent.fix_kpi(self._sample_kpi(), "error")
        assert result == "df['amount'].mean()"

    def test_uses_haiku_model(self):
        agent, mock_client = self._make_fix_agent("df['amount'].mean()")
        agent.fix_kpi(self._sample_kpi(), "error")
        _, kwargs = mock_client.call.call_args
        assert kwargs["model"] == MODEL_HAIKU

    def test_expect_json_false(self):
        agent, mock_client = self._make_fix_agent("df['amount'].mean()")
        agent.fix_kpi(self._sample_kpi(), "error")
        _, kwargs = mock_client.call.call_args
        assert kwargs["expect_json"] is False

    def test_traceback_in_prompt(self):
        agent, mock_client = self._make_fix_agent("df['amount'].mean()")
        tb = "KeyError: 'amount'"
        agent.fix_kpi(self._sample_kpi(), tb)
        _, kwargs = mock_client.call.call_args
        assert tb in kwargs["user_prompt"]

    def test_llm_error_wrapped(self):
        mock_client = MagicMock()
        mock_client.call.side_effect = LLMError("down")
        agent = KPIAgent(llm_client=mock_client)
        with pytest.raises(KPIAgentError, match="Fix LLM call failed"):
            agent.fix_kpi(self._sample_kpi(), "error")
