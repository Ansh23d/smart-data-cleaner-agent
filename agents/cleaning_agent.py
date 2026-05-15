"""Cleaning agent: takes a DataProfile, asks Claude for a cleaning plan,
validates the plan against the profile, and returns a CleaningPlan.

Public API
----------
CleaningAgent(llm_client).plan(profile, user_feedback=None) → CleaningPlan
"""

from __future__ import annotations

import logging

from config import (
    CLEANING_OPERATION_ORDER,
    MAX_TOKENS_CLEANING_PLAN,
    MAX_TOKENS_CODE_FIX,
    MODEL_HAIKU,
    MODEL_SONNET,
    PROMPT_CLEANING_FIX,
    PROMPT_CLEANING_PLAN,
    TEMPERATURE_CLEANING_PLAN,
    TEMPERATURE_CODE_FIX,
    PROMPT_CLEANING_SYSTEM,
)
from models.cleaning_plan import CleaningOperation, CleaningOperationType, CleaningPlan
from models.profile import DataProfile
from services.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)


class CleaningAgentError(Exception):
    """Raised when the cleaning agent cannot produce a valid plan."""


class CleaningAgent:
    """Generates and validates a CleaningPlan from a DataProfile.

    Parameters
    ----------
    llm_client:
        An :class:`~services.llm_client.LLMClient` instance.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client
        self._system_prompt = PROMPT_CLEANING_SYSTEM.read_text(encoding="utf-8")
        self._plan_template = PROMPT_CLEANING_PLAN.read_text(encoding="utf-8")
        self._fix_template = PROMPT_CLEANING_FIX.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def plan(
        self,
        profile: DataProfile,
        user_feedback: str | None = None,
    ) -> CleaningPlan:
        """Ask the LLM for a cleaning plan, validate it, and return it.

        Parameters
        ----------
        profile:
            The DataProfile of the DataFrame to clean.
        user_feedback:
            Optional free-text instructions from the user (e.g. "don't drop
            any rows, prefer filling over dropping").

        Returns
        -------
        CleaningPlan
            Validated plan with only operations referencing columns that exist
            in the profile and operation types from the allowed set.

        Raises
        ------
        CleaningAgentError
            When the LLM fails to return a valid plan after retries.
        """
        user_prompt = self._build_user_prompt(profile, user_feedback)

        try:
            raw: dict = self._client.call(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                model=MODEL_SONNET,
                max_tokens=MAX_TOKENS_CLEANING_PLAN,
                expect_json=True,
                temperature=TEMPERATURE_CLEANING_PLAN,
            )
        except LLMError as exc:
            raise CleaningAgentError(f"LLM call failed: {exc}") from exc

        plan = self._parse_and_validate(raw, profile)
        plan = self._sort_operations(plan)
        return plan

    def fix_operation(
        self,
        operation: CleaningOperation,
        traceback: str,
    ) -> str:
        """Ask the LLM to fix a failing operation and return corrected code.

        Parameters
        ----------
        operation:
            The operation whose code failed at runtime.
        traceback:
            The full traceback string from the executor.

        Returns
        -------
        str
            Corrected Python code string (not yet executed).

        Raises
        ------
        CleaningAgentError
            When the LLM fails to return a response.
        """
        user_prompt = self._fix_template.format(
            operation_id=operation.id,
            operation_type=operation.type.value,
            operation_description=operation.description,
            original_code=operation.code,
            traceback=traceback,
        )

        try:
            fixed_code: str = self._client.call(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                model=MODEL_HAIKU,
                max_tokens=MAX_TOKENS_CODE_FIX,
                expect_json=False,
                temperature=TEMPERATURE_CODE_FIX,
            )
        except LLMError as exc:
            raise CleaningAgentError(f"Fix LLM call failed: {exc}") from exc

        return fixed_code.strip()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_user_prompt(
        self,
        profile: DataProfile,
        user_feedback: str | None,
    ) -> str:
        allowed_types = "\n".join(
            f"  - {op}" for op in CLEANING_OPERATION_ORDER
        )
        profile_json = profile.model_dump_json(indent=2)

        if user_feedback:
            feedback_section = f"\nUSER INSTRUCTIONS:\n{user_feedback}\n"
        else:
            feedback_section = ""

        return self._plan_template.format(
            allowed_types=allowed_types,
            profile_json=profile_json,
            user_feedback_section=feedback_section,
        )

    def _parse_and_validate(
        self,
        raw: dict,
        profile: DataProfile,
    ) -> CleaningPlan:
        """Parse the raw LLM dict into a CleaningPlan and strip invalid ops."""
        known_columns = {col.name for col in profile.schema_report}
        allowed_types = {t.value for t in CleaningOperationType}

        try:
            plan = CleaningPlan.model_validate(raw)
        except Exception as exc:
            raise CleaningAgentError(
                f"LLM response did not match CleaningPlan schema: {exc}\n"
                f"Raw response: {raw!r}"
            ) from exc

        valid_ops: list[CleaningOperation] = []
        for op in plan.operations:
            # Drop operations whose type is not in the allowed set
            # (Pydantic already validates this via the enum, but guard anyway)
            if op.type.value not in allowed_types:
                logger.warning("Dropping op %r — unknown type %r", op.id, op.type)
                continue

            # Drop operations referencing non-existent columns
            bad_cols = [c for c in op.column_names if c not in known_columns]
            if bad_cols:
                logger.warning(
                    "Dropping op %r — references unknown columns: %s",
                    op.id,
                    bad_cols,
                )
                continue

            valid_ops.append(op)

        return CleaningPlan(operations=valid_ops)

    def _sort_operations(self, plan: CleaningPlan) -> CleaningPlan:
        """Re-order operations according to CLEANING_OPERATION_ORDER."""
        order_index = {name: i for i, name in enumerate(CLEANING_OPERATION_ORDER)}
        sorted_ops = sorted(
            plan.operations,
            key=lambda op: order_index.get(op.type.value, len(CLEANING_OPERATION_ORDER)),
        )
        return CleaningPlan(operations=sorted_ops)
