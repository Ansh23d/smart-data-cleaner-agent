"""Tests for agents/cleaning_agent.py.

All LLM calls are mocked — no real network traffic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.cleaning_agent import CleaningAgent, CleaningAgentError
from app.config import CLEANING_OPERATION_ORDER, MODEL_HAIKU, MODEL_SONNET
from models.cleaning_plan import CleaningOperation, CleaningOperationType, CleaningPlan
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
    """Minimal DataProfile with two columns: 'name' (categorical) and 'revenue' (numeric)."""
    return DataProfile(
        row_count=10,
        column_count=2,
        schema_report=[
            ColumnSchema(name="name", dtype="object", semantic_type=SemanticType.CATEGORICAL),
            ColumnSchema(name="revenue", dtype="float64", semantic_type=SemanticType.NUMERIC_CONTINUOUS),
        ],
        null_analysis=[
            NullReport(column="name", null_count=0, null_pct=0.0),
            NullReport(column="revenue", null_count=2, null_pct=0.2),
        ],
        duplicate_count=0,
        statistics={
            "name": CategoricalStats(unique_count=8, cardinality_ratio=0.8, top_values={"Alice": 3}),
            "revenue": NumericStats(min=10.0, max=500.0, mean=200.0, median=180.0,
                                    std=90.0, q1=100.0, q3=300.0, skewness=0.5),
        },
        quality_flags=[
            QualityFlag(column="name", flag_type=QualityFlagType.WHITESPACE_ISSUES,
                        description="Leading/trailing whitespace detected"),
            QualityFlag(column="revenue", flag_type=QualityFlagType.HIGH_NULLS,
                        description="20% null"),
        ],
        sample_rows=SampleRows(first=[], last=[], random=[]),
    )


def _make_op(
    op_id: str = "fill_revenue_median",
    op_type: str = "fill_missing",
    columns: list[str] | None = None,
) -> dict:
    return {
        "id": op_id,
        "type": op_type,
        "description": "Fill nulls in revenue",
        "impact": "Fills 2 nulls",
        "confidence": "medium",
        "reversible": False,
        "code": "df['revenue'] = df['revenue'].fillna(df['revenue'].median())",
        "column_names": columns if columns is not None else ["revenue"],
        "alternatives": [],
    }


def _make_plan_response(*ops: dict) -> dict:
    return {"operations": list(ops)}


def _make_agent(mock_response) -> tuple[CleaningAgent, MagicMock]:
    """Return (CleaningAgent, mock_llm_client) primed with mock_response."""
    mock_client = MagicMock()
    mock_client.call.return_value = mock_response
    agent = CleaningAgent(llm_client=mock_client)
    return agent, mock_client


# ---------------------------------------------------------------------------
# CleaningAgent.plan — happy path
# ---------------------------------------------------------------------------

class TestPlanHappyPath:
    def test_returns_cleaning_plan(self, simple_profile):
        response = _make_plan_response(_make_op())
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert isinstance(result, CleaningPlan)

    def test_single_valid_operation(self, simple_profile):
        response = _make_plan_response(_make_op())
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert len(result.operations) == 1
        assert result.operations[0].id == "fill_revenue_median"

    def test_multiple_valid_operations(self, simple_profile):
        op1 = _make_op("strip_name", "fix_whitespace", ["name"])
        op2 = _make_op("fill_revenue_median", "fill_missing", ["revenue"])
        response = _make_plan_response(op1, op2)
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert len(result.operations) == 2

    def test_empty_operations_list(self, simple_profile):
        response = {"operations": []}
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert result.operations == []

    def test_uses_sonnet_model(self, simple_profile):
        response = _make_plan_response(_make_op())
        agent, mock_client = _make_agent(response)
        agent.plan(simple_profile)
        _, kwargs = mock_client.call.call_args
        assert kwargs["model"] == MODEL_SONNET

    def test_expect_json_true(self, simple_profile):
        response = _make_plan_response(_make_op())
        agent, mock_client = _make_agent(response)
        agent.plan(simple_profile)
        _, kwargs = mock_client.call.call_args
        assert kwargs["expect_json"] is True

    def test_profile_json_in_user_prompt(self, simple_profile):
        response = _make_plan_response(_make_op())
        agent, mock_client = _make_agent(response)
        agent.plan(simple_profile)
        _, kwargs = mock_client.call.call_args
        assert "revenue" in kwargs["user_prompt"]

    def test_allowed_types_in_user_prompt(self, simple_profile):
        response = _make_plan_response(_make_op())
        agent, mock_client = _make_agent(response)
        agent.plan(simple_profile)
        _, kwargs = mock_client.call.call_args
        for op_type in CLEANING_OPERATION_ORDER:
            assert op_type in kwargs["user_prompt"]

    def test_user_feedback_included_in_prompt(self, simple_profile):
        response = _make_plan_response(_make_op())
        agent, mock_client = _make_agent(response)
        agent.plan(simple_profile, user_feedback="Don't drop any rows")
        _, kwargs = mock_client.call.call_args
        assert "Don't drop any rows" in kwargs["user_prompt"]

    def test_no_user_feedback_omits_section(self, simple_profile):
        response = _make_plan_response(_make_op())
        agent, mock_client = _make_agent(response)
        agent.plan(simple_profile, user_feedback=None)
        _, kwargs = mock_client.call.call_args
        assert "USER INSTRUCTIONS" not in kwargs["user_prompt"]


# ---------------------------------------------------------------------------
# CleaningAgent.plan — column name validation
# ---------------------------------------------------------------------------

class TestColumnValidation:
    def test_unknown_column_is_dropped(self, simple_profile):
        op = _make_op("bad_op", "fill_missing", ["nonexistent_column"])
        response = _make_plan_response(op)
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert len(result.operations) == 0

    def test_mix_valid_and_invalid_columns(self, simple_profile):
        op_valid = _make_op("fill_revenue_median", "fill_missing", ["revenue"])
        op_bad = _make_op("bad_op", "fill_missing", ["ghost_column"])
        response = _make_plan_response(op_valid, op_bad)
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert len(result.operations) == 1
        assert result.operations[0].id == "fill_revenue_median"

    def test_empty_column_names_passes(self, simple_profile):
        """Operations with no column_names (e.g. drop_duplicates) should be kept."""
        op = _make_op("drop_dupes", "drop_duplicates", [])
        response = _make_plan_response(op)
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert len(result.operations) == 1

    def test_partial_bad_columns_drops_whole_op(self, simple_profile):
        """If ANY column in column_names is unknown, the whole op is dropped."""
        op = _make_op("mixed", "fill_missing", ["revenue", "ghost"])
        response = _make_plan_response(op)
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert len(result.operations) == 0

    def test_all_known_columns_kept(self, simple_profile):
        op = _make_op("multi_col", "fix_whitespace", ["name", "revenue"])
        response = _make_plan_response(op)
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert len(result.operations) == 1


# ---------------------------------------------------------------------------
# CleaningAgent.plan — operation ordering
# ---------------------------------------------------------------------------

class TestOperationOrdering:
    def test_operations_sorted_by_priority(self, simple_profile):
        """fill_missing (index 4) should come after fix_whitespace (index 1)."""
        op_fill = _make_op("fill_revenue_median", "fill_missing", ["revenue"])
        op_ws = _make_op("strip_name", "fix_whitespace", ["name"])
        # Pass fill first, whitespace second — expect whitespace first after sort
        response = _make_plan_response(op_fill, op_ws)
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert result.operations[0].type == CleaningOperationType.FIX_WHITESPACE
        assert result.operations[1].type == CleaningOperationType.FILL_MISSING

    def test_already_sorted_stays_sorted(self, simple_profile):
        op_ws = _make_op("strip_name", "fix_whitespace", ["name"])
        op_fill = _make_op("fill_revenue_median", "fill_missing", ["revenue"])
        response = _make_plan_response(op_ws, op_fill)
        agent, _ = _make_agent(response)
        result = agent.plan(simple_profile)
        assert result.operations[0].type == CleaningOperationType.FIX_WHITESPACE
        assert result.operations[1].type == CleaningOperationType.FILL_MISSING


# ---------------------------------------------------------------------------
# CleaningAgent.plan — malformed JSON / schema errors
# ---------------------------------------------------------------------------

class TestMalformedResponse:
    def test_missing_operations_key_raises(self, simple_profile):
        agent, _ = _make_agent({"wrong_key": []})
        with pytest.raises(CleaningAgentError, match="schema"):
            agent.plan(simple_profile)

    def test_operation_missing_required_field_raises(self, simple_profile):
        bad_op = {"id": "x", "type": "fill_missing"}  # missing required fields
        agent, _ = _make_agent({"operations": [bad_op]})
        with pytest.raises(CleaningAgentError):
            agent.plan(simple_profile)

    def test_invalid_operation_type_raises(self, simple_profile):
        op = _make_op()
        op["type"] = "not_a_real_type"
        agent, _ = _make_agent({"operations": [op]})
        with pytest.raises(CleaningAgentError):
            agent.plan(simple_profile)

    def test_invalid_confidence_raises(self, simple_profile):
        op = _make_op()
        op["confidence"] = "extreme"   # not in Literal["high", "medium", "low"]
        agent, _ = _make_agent({"operations": [op]})
        with pytest.raises(CleaningAgentError):
            agent.plan(simple_profile)

    def test_llm_error_wrapped_as_cleaning_agent_error(self, simple_profile):
        mock_client = MagicMock()
        mock_client.call.side_effect = LLMError("API down")
        agent = CleaningAgent(llm_client=mock_client)
        with pytest.raises(CleaningAgentError, match="LLM call failed"):
            agent.plan(simple_profile)


# ---------------------------------------------------------------------------
# CleaningAgent.fix_operation
# ---------------------------------------------------------------------------

class TestFixOperation:
    def _make_fix_agent(self, fixed_code: str) -> tuple[CleaningAgent, MagicMock]:
        mock_client = MagicMock()
        mock_client.call.return_value = fixed_code
        return CleaningAgent(llm_client=mock_client), mock_client

    def _sample_op(self) -> CleaningOperation:
        return CleaningOperation(
            id="fill_revenue_median",
            type=CleaningOperationType.FILL_MISSING,
            description="Fill nulls in revenue",
            impact="Fills 2 nulls",
            confidence="medium",
            reversible=False,
            code="df['revenue'] = df['revenue'].fillna(df['revenue'].median())",
            column_names=["revenue"],
            alternatives=[],
        )

    def test_returns_fixed_code_string(self):
        agent, _ = self._make_fix_agent("df['revenue'] = df['revenue'].fillna(0)")
        result = agent.fix_operation(self._sample_op(), "KeyError: 'revenue'")
        assert result == "df['revenue'] = df['revenue'].fillna(0)"

    def test_strips_whitespace_from_code(self):
        agent, _ = self._make_fix_agent("  df['x'] = 1  \n")
        result = agent.fix_operation(self._sample_op(), "some error")
        assert result == "df['x'] = 1"

    def test_uses_haiku_model(self):
        agent, mock_client = self._make_fix_agent("df['x'] = 1")
        agent.fix_operation(self._sample_op(), "error")
        _, kwargs = mock_client.call.call_args
        assert kwargs["model"] == MODEL_HAIKU

    def test_expect_json_false(self):
        agent, mock_client = self._make_fix_agent("df['x'] = 1")
        agent.fix_operation(self._sample_op(), "error")
        _, kwargs = mock_client.call.call_args
        assert kwargs["expect_json"] is False

    def test_traceback_in_user_prompt(self):
        agent, mock_client = self._make_fix_agent("df['x'] = 1")
        tb = "Traceback:\n  KeyError: 'revenue'"
        agent.fix_operation(self._sample_op(), tb)
        _, kwargs = mock_client.call.call_args
        assert tb in kwargs["user_prompt"]

    def test_original_code_in_user_prompt(self):
        agent, mock_client = self._make_fix_agent("df['x'] = 1")
        op = self._sample_op()
        agent.fix_operation(op, "some error")
        _, kwargs = mock_client.call.call_args
        assert op.code in kwargs["user_prompt"]

    def test_llm_error_wrapped(self):
        mock_client = MagicMock()
        mock_client.call.side_effect = LLMError("down")
        agent = CleaningAgent(llm_client=mock_client)
        with pytest.raises(CleaningAgentError, match="Fix LLM call failed"):
            agent.fix_operation(self._sample_op(), "error")


# ---------------------------------------------------------------------------
# CleaningPlan model helpers (plan-level validation)
# ---------------------------------------------------------------------------

class TestCleaningPlanHelpers:
    def _make_plan(self) -> CleaningPlan:
        ops = [
            CleaningOperation(
                id="op1",
                type=CleaningOperationType.FILL_MISSING,
                description="d",
                impact="i",
                confidence="high",
                reversible=False,
                code="df['x'] = 1",
                column_names=["x"],
                alternatives=[],
            ),
            CleaningOperation(
                id="op2",
                type=CleaningOperationType.DROP_DUPLICATES,
                description="d",
                impact="i",
                confidence="low",
                reversible=True,
                code="df = df.drop_duplicates()",
                column_names=[],
                alternatives=[],
            ),
        ]
        return CleaningPlan(operations=ops)

    def test_filter_by_ids(self):
        plan = self._make_plan()
        filtered = plan.filter_by_ids(["op1"])
        assert len(filtered.operations) == 1
        assert filtered.operations[0].id == "op1"

    def test_filter_by_ids_empty(self):
        plan = self._make_plan()
        filtered = plan.filter_by_ids([])
        assert filtered.operations == []

    def test_operations_by_type(self):
        plan = self._make_plan()
        fills = plan.operations_by_type(CleaningOperationType.FILL_MISSING)
        assert len(fills) == 1
        assert fills[0].id == "op1"

    def test_operations_by_type_no_match(self):
        plan = self._make_plan()
        result = plan.operations_by_type(CleaningOperationType.DROP_COLUMN)
        assert result == []
