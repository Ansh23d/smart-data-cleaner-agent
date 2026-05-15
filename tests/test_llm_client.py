"""Tests for services/llm_client.py and utils/cost_tracker.py.

All Anthropic API calls are mocked — no real network traffic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import httpx
import pytest
from anthropic import InternalServerError, RateLimitError

from app.config import (
    COST_PER_M_INPUT,
    COST_PER_M_OUTPUT,
    LLM_MAX_RETRIES,
    MODEL_HAIKU,
    MODEL_SONNET,
)
from services.llm_client import LLMClient, LLMError, _parse_json, _strip_fences
from utils.cost_tracker import CostTracker, _compute_cost


# ---------------------------------------------------------------------------
# Helpers — build fake Anthropic error / response objects
# ---------------------------------------------------------------------------

def _http_response(status: int) -> httpx.Response:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status, request=req)


def _rate_limit_error(msg: str = "rate limit") -> RateLimitError:
    return RateLimitError(msg, response=_http_response(429), body=None)


def _server_error(msg: str = "internal error") -> InternalServerError:
    return InternalServerError(msg, response=_http_response(500), body=None)


# ---------------------------------------------------------------------------
# Helpers — build fake Anthropic response objects
# ---------------------------------------------------------------------------

def _make_response(text: str, input_tokens: int = 100, output_tokens: int = 50, model: str = MODEL_SONNET):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    content = [SimpleNamespace(text=text)]
    return SimpleNamespace(usage=usage, content=content, model=model)


def _make_client(response_text: str, **kwargs) -> tuple[LLMClient, MagicMock]:
    """Return (LLMClient, mock_messages_create) primed to return one response."""
    tracker = CostTracker()
    client = LLMClient(tracker=tracker)
    mock_create = MagicMock(return_value=_make_response(response_text, **kwargs))
    client._client.messages.create = mock_create
    return client, mock_create


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_no_fence(self):
        assert _strip_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fence(self):
        assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_fence(self):
        assert _strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_uppercase_json_fence(self):
        assert _strip_fences('```JSON\n{"a": 1}\n```') == '{"a": 1}'

    def test_inline_backtick(self):
        assert _strip_fences('`{"a": 1}`') == '{"a": 1}'

    def test_fence_with_leading_whitespace(self):
        result = _strip_fences('  ```json\n{"x": 2}\n```  ')
        assert json.loads(result) == {"x": 2}

    def test_multiline_json_in_fence(self):
        raw = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        result = _strip_fences(raw)
        assert json.loads(result) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_clean_json(self):
        assert _parse_json('{"key": "value"}') == {"key": "value"}

    def test_json_in_fence(self):
        assert _parse_json('```json\n{"key": "value"}\n```') == {"key": "value"}

    def test_json_array(self):
        assert _parse_json('[1, 2, 3]') == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_json("not json at all")

    def test_partial_json_raises(self):
        with pytest.raises(ValueError):
            _parse_json('{"unclosed": ')

    def test_whitespace_around_json(self):
        assert _parse_json('  {"a": 1}  ') == {"a": 1}

    def test_nested_json(self):
        nested = '{"outer": {"inner": [1, 2, 3]}}'
        assert _parse_json(nested) == {"outer": {"inner": [1, 2, 3]}}


# ---------------------------------------------------------------------------
# LLMClient.call — happy path
# ---------------------------------------------------------------------------

class TestCallHappyPath:
    def test_returns_parsed_json(self):
        client, mock_create = _make_client('{"answer": 42}')
        result = client.call("sys", "user")
        assert result == {"answer": 42}

    def test_returns_json_from_fence(self):
        client, mock_create = _make_client('```json\n{"x": 1}\n```')
        result = client.call("sys", "user")
        assert result == {"x": 1}

    def test_expect_json_false_returns_text(self):
        client, mock_create = _make_client("plain text response")
        result = client.call("sys", "user", expect_json=False)
        assert result == "plain text response"

    def test_default_model_is_sonnet(self):
        client, mock_create = _make_client('{"ok": true}')
        client.call("sys", "user")
        _, kwargs = mock_create.call_args
        assert kwargs["model"] == MODEL_SONNET

    def test_explicit_model_override(self):
        client, mock_create = _make_client('{"ok": true}')
        client.call("sys", "user", model=MODEL_HAIKU)
        _, kwargs = mock_create.call_args
        assert kwargs["model"] == MODEL_HAIKU

    def test_max_tokens_passed_through(self):
        client, mock_create = _make_client('{"ok": true}')
        client.call("sys", "user", max_tokens=512)
        _, kwargs = mock_create.call_args
        assert kwargs["max_tokens"] == 512

    def test_system_and_user_passed(self):
        client, mock_create = _make_client('{"ok": true}')
        client.call("my system", "my user")
        _, kwargs = mock_create.call_args
        assert kwargs["system"] == "my system"
        assert kwargs["messages"][0]["content"] == "my user"


# ---------------------------------------------------------------------------
# LLMClient — retry on API errors
# ---------------------------------------------------------------------------

class TestRetryOnApiError:
    def _client_with_errors_then_success(self, errors: list) -> LLMClient:
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(
            side_effect=errors + [_make_response('{"ok": true}')]
        )
        return client

    @patch("services.llm_client.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep):
        client = self._client_with_errors_then_success([_rate_limit_error()])
        result = client.call("sys", "user")
        assert result == {"ok": True}
        mock_sleep.assert_called_once()

    @patch("services.llm_client.time.sleep")
    def test_retries_on_internal_server_error(self, mock_sleep):
        client = self._client_with_errors_then_success([_server_error()])
        result = client.call("sys", "user")
        assert result == {"ok": True}
        mock_sleep.assert_called_once()

    @patch("services.llm_client.time.sleep")
    def test_retry_uses_correct_delay(self, mock_sleep):
        from app.config import LLM_RETRY_DELAYS
        client = self._client_with_errors_then_success([_rate_limit_error()])
        client.call("sys", "user")
        mock_sleep.assert_called_once_with(LLM_RETRY_DELAYS[0])

    @patch("services.llm_client.time.sleep")
    def test_exhausted_retries_raise_llm_error(self, mock_sleep):
        from app.config import LLM_RETRY_DELAYS
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(
            side_effect=_rate_limit_error("always fails")
        )
        with pytest.raises(LLMError):
            client.call("sys", "user")
        # Each retry now sleeps 1s at a time for the full delay duration;
        # total calls = sum(LLM_RETRY_DELAYS) = 1 + 2 + 4 = 7
        assert mock_sleep.call_count == sum(LLM_RETRY_DELAYS)

    @patch("services.llm_client.time.sleep")
    def test_two_errors_then_success(self, mock_sleep):
        from app.config import LLM_RETRY_DELAYS
        client = self._client_with_errors_then_success(
            [_rate_limit_error(), _rate_limit_error()]
        )
        result = client.call("sys", "user")
        assert result == {"ok": True}
        # Two retries with delays 1s and 2s → 1 + 2 = 3 sleep(1) calls
        assert mock_sleep.call_count == sum(LLM_RETRY_DELAYS[:2])

    def test_unexpected_error_raises_immediately(self):
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(
            side_effect=ValueError("unexpected")
        )
        with pytest.raises(LLMError, match="Unexpected error"):
            client.call("sys", "user")


# ---------------------------------------------------------------------------
# LLMClient — retry on JSON parse failure
# ---------------------------------------------------------------------------

class TestRetryOnJsonFailure:
    @patch("services.llm_client.time.sleep")
    def test_bad_json_triggers_strict_retry(self, mock_sleep):
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(side_effect=[
            _make_response("not json"),        # first attempt — bad JSON
            _make_response('{"fixed": true}'), # second attempt — good JSON
        ])
        result = client.call("sys", "user")
        assert result == {"fixed": True}
        # No sleep for JSON retries
        mock_sleep.assert_not_called()

    @patch("services.llm_client.time.sleep")
    def test_strict_prompt_appended_on_retry(self, mock_sleep):
        from services.llm_client import _JSON_STRICT_SUFFIX
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(side_effect=[
            _make_response("bad"),
            _make_response('{"ok": 1}'),
        ])
        client.call("sys", "original user prompt")
        # Second call should have the strict suffix appended
        second_call_kwargs = client._client.messages.create.call_args_list[1][1]
        assert _JSON_STRICT_SUFFIX in second_call_kwargs["messages"][0]["content"]

    @patch("services.llm_client.time.sleep")
    def test_always_bad_json_raises(self, mock_sleep):
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(
            return_value=_make_response("never json !!!")
        )
        with pytest.raises(LLMError):
            client.call("sys", "user")

    @patch("services.llm_client.time.sleep")
    def test_fenced_json_does_not_retry(self, mock_sleep):
        """A fenced JSON response should parse on first try — no retry needed."""
        client, mock_create = _make_client('```json\n{"clean": true}\n```')
        result = client.call("sys", "user")
        assert result == {"clean": True}
        assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_tokens == 0
        assert len(tracker.records) == 0

    def test_record_adds_entry(self):
        tracker = CostTracker()
        tracker.record(MODEL_SONNET, 1000, 500)
        assert len(tracker.records) == 1

    def test_cost_calculation_sonnet(self):
        tracker = CostTracker()
        tracker.record(MODEL_SONNET, 1_000_000, 0)
        expected = COST_PER_M_INPUT[MODEL_SONNET]
        assert tracker.total_cost_usd == pytest.approx(expected)

    def test_cost_calculation_output(self):
        tracker = CostTracker()
        tracker.record(MODEL_SONNET, 0, 1_000_000)
        expected = COST_PER_M_OUTPUT[MODEL_SONNET]
        assert tracker.total_cost_usd == pytest.approx(expected)

    def test_cost_calculation_haiku(self):
        tracker = CostTracker()
        tracker.record(MODEL_HAIKU, 1_000_000, 1_000_000)
        expected = COST_PER_M_INPUT[MODEL_HAIKU] + COST_PER_M_OUTPUT[MODEL_HAIKU]
        assert tracker.total_cost_usd == pytest.approx(expected)

    def test_accumulates_across_multiple_calls(self):
        tracker = CostTracker()
        tracker.record(MODEL_SONNET, 500, 200)
        tracker.record(MODEL_SONNET, 500, 200)
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 400
        assert len(tracker.records) == 2

    def test_reset_clears_all(self):
        tracker = CostTracker()
        tracker.record(MODEL_SONNET, 100, 50)
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert len(tracker.records) == 0

    def test_summary_keys(self):
        tracker = CostTracker()
        tracker.record(MODEL_SONNET, 100, 50)
        summary = tracker.summary()
        assert set(summary.keys()) == {
            "calls", "input_tokens", "output_tokens",
            "total_tokens", "estimated_cost_usd",
        }

    def test_summary_values(self):
        tracker = CostTracker()
        tracker.record(MODEL_SONNET, 300, 100)
        s = tracker.summary()
        assert s["calls"] == 1
        assert s["input_tokens"] == 300
        assert s["output_tokens"] == 100
        assert s["total_tokens"] == 400

    def test_unknown_model_cost_is_zero(self):
        tracker = CostTracker()
        tracker.record("unknown-model-xyz", 1_000_000, 1_000_000)
        assert tracker.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Cost tracking via LLMClient
# ---------------------------------------------------------------------------

class TestLLMClientCostTracking:
    def test_single_call_records_cost(self):
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(
            return_value=_make_response('{"ok": true}', input_tokens=200, output_tokens=80)
        )
        client.call("sys", "user")
        assert tracker.total_input_tokens == 200
        assert tracker.total_output_tokens == 80
        assert tracker.total_cost_usd > 0

    def test_multiple_calls_accumulate(self):
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(
            return_value=_make_response('{"ok": true}', input_tokens=100, output_tokens=50)
        )
        client.call("sys", "user")
        client.call("sys", "user")
        assert tracker.total_input_tokens == 200
        assert len(tracker.records) == 2

    def test_failed_call_does_not_record_cost(self):
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(
            side_effect=_rate_limit_error("limit")
        )
        with patch("services.llm_client.time.sleep"):
            with pytest.raises(LLMError):
                client.call("sys", "user")
        assert tracker.total_tokens == 0

    def test_haiku_model_recorded_separately(self):
        tracker = CostTracker()
        client = LLMClient(tracker=tracker)
        client._client.messages.create = MagicMock(
            return_value=_make_response('{"ok": true}', input_tokens=100, output_tokens=50)
        )
        client.call("sys", "user", model=MODEL_HAIKU)
        assert tracker.records[0].model == MODEL_HAIKU

    def test_tracker_shared_across_calls(self):
        """Two LLMClient instances sharing one tracker accumulate correctly."""
        tracker = CostTracker()
        c1 = LLMClient(tracker=tracker)
        c2 = LLMClient(tracker=tracker)
        for c in (c1, c2):
            c._client.messages.create = MagicMock(
                return_value=_make_response('{"ok": true}', input_tokens=50, output_tokens=20)
            )
        c1.call("sys", "user")
        c2.call("sys", "user")
        assert len(tracker.records) == 2
        assert tracker.total_input_tokens == 100
