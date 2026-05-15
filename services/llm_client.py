"""Anthropic SDK wrapper with retry, JSON parsing, and cost tracking.

Usage
-----
    tracker = CostTracker()
    client = LLMClient(tracker=tracker)

    result: dict = client.call(
        system_prompt="You are a helpful assistant.",
        user_prompt="Return JSON: {\"answer\": 42}",
    )
    print(tracker.summary())
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

import anthropic
from anthropic import APIStatusError, APITimeoutError, InternalServerError, RateLimitError
from dotenv import load_dotenv

from config import (
    COST_PER_M_INPUT,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAYS,
    MAX_TOKENS_CLEANING_PLAN,
    MODEL_HAIKU,
    MODEL_SONNET,
)
from utils.cost_tracker import CostTracker
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Errors that warrant a retry
_RETRYABLE = (RateLimitError, InternalServerError, APITimeoutError)

# Prompt appended on a JSON-parse failure to demand strict output
_JSON_STRICT_SUFFIX = (
    "\n\nIMPORTANT: Your previous response could not be parsed as JSON. "
    "Return ONLY valid JSON — no markdown fences, no explanation, no preamble. "
    "Start your response with { or [ and end with } or ]."
)


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class LLMClient:
    """Thin wrapper around the Anthropic Messages API.

    Parameters
    ----------
    tracker:
        A :class:`~utils.cost_tracker.CostTracker` instance.
        All calls made by this client are recorded there.
    default_model:
        Model used when the caller does not specify one.
    """

    def __init__(
        self,
        tracker: CostTracker | None = None,
        default_model: str = MODEL_SONNET,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        self._tracker = tracker or CostTracker()
        self._default_model = default_model
        self._status_callback = status_callback

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = MAX_TOKENS_CLEANING_PLAN,
        expect_json: bool = True,
        temperature: float = 0.2,
    ) -> dict[str, Any] | str:
        """Send a prompt to Claude and return the parsed response.

        Parameters
        ----------
        system_prompt:
            The system turn content.
        user_prompt:
            The user turn content.
        model:
            Overrides ``default_model`` for this call.
            Pass ``MODEL_HAIKU`` for cheap code-fix retries.
        max_tokens:
            Hard cap on output tokens.
        expect_json:
            If ``True`` (default), parse the response as JSON and retry
            with a stricter prompt on failure.
            If ``False``, return the raw text string.

        Returns
        -------
        dict | str
            Parsed JSON dict (when ``expect_json=True``) or raw text.

        Raises
        ------
        LLMError
            When all retries are exhausted.
        """
        resolved_model = model or self._default_model
        return self._call_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=resolved_model,
            max_tokens=max_tokens,
            expect_json=expect_json,
            temperature=temperature,
        )

    # ------------------------------------------------------------------ #
    # Internal — retry loop
    # ------------------------------------------------------------------ #

    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        expect_json: bool,
        temperature: float,
    ) -> dict[str, Any] | str:
        last_exc: Exception | None = None
        current_user_prompt = user_prompt

        for attempt in range(LLM_MAX_RETRIES + 1):   # 0 … LLM_MAX_RETRIES
            try:
                raw_text = self._send(
                    system_prompt=system_prompt,
                    user_prompt=current_user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt < LLM_MAX_RETRIES:
                    delay = LLM_RETRY_DELAYS[min(attempt, len(LLM_RETRY_DELAYS) - 1)]
                    logger.warning(
                        "llm_rate_limit",
                        attempt=attempt + 1,
                        max_retries=LLM_MAX_RETRIES,
                        error=str(exc),
                        retry_in_seconds=delay,
                    )
                    for remaining in range(delay, 0, -1):
                        if self._status_callback:
                            self._status_callback(
                                f"Waiting for API… retrying in {remaining}s "
                                f"(attempt {attempt + 1}/{LLM_MAX_RETRIES})"
                            )
                        time.sleep(1)
                continue
            except Exception as exc:
                raise LLMError(f"Unexpected error calling LLM: {exc}") from exc

            if not expect_json:
                return raw_text

            # JSON parse — may append strict suffix and loop once more
            try:
                return _parse_json(raw_text)
            except ValueError as json_exc:
                last_exc = json_exc
                if attempt < LLM_MAX_RETRIES:
                    logger.warning(
                        "llm_json_parse_failed",
                        attempt=attempt + 1,
                        max_retries=LLM_MAX_RETRIES,
                        error=str(json_exc),
                    )
                    current_user_prompt = user_prompt + _JSON_STRICT_SUFFIX
                    # No sleep needed — parse failures are not rate-limit related
                    continue

        raise LLMError(
            f"LLM call failed after {LLM_MAX_RETRIES} retries. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------ #
    # Internal — single network call
    # ------------------------------------------------------------------ #

    def _send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Make one API call, record cost, and return the text content."""
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens
        self._tracker.record(model, input_tokens, output_tokens)

        logger.info(
            "llm_call",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(self._tracker.records[-1].cost_usd, 6),
        )

        # Extract text from the first content block
        return response.content[0].text


# ---------------------------------------------------------------------------
# JSON parsing utilities (module-level so tests can call them directly)
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict[str, Any]:
    """Strip markdown fences then parse JSON. Raises ValueError on failure."""
    cleaned = _strip_fences(text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from LLM: {exc}\nRaw text: {text!r}") from exc


def _strip_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` markdown code fences."""
    # Match optional language tag after the opening fence
    fenced = re.sub(
        r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    if fenced != text:
        return fenced
    # Single-line fence (rare): `{...}`
    return re.sub(r"^`(.+)`$", r"\1", text.strip())
