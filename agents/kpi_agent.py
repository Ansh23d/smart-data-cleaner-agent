"""KPI agent: takes a cleaned DataProfile + domain context, returns KPIs.

Public API
----------
KPIAgent(llm_client).suggest(profile, domain_context) → list[KPI]
KPIAgent(llm_client).fix_kpi(kpi, traceback)          → str  (corrected code)
"""

from __future__ import annotations

import logging

from config import (
    MAX_TOKENS_CODE_FIX,
    MAX_TOKENS_KPI,
    MODEL_HAIKU,
    MODEL_SONNET,
    PROMPT_KPI_FIX,
    PROMPT_KPI_SUGGEST,
    PROMPT_KPI_SYSTEM,
    TEMPERATURE_CODE_FIX,
    TEMPERATURE_KPI_SUGGEST,
)
from models.kpi import KPI, ChartConfig
from models.profile import DataProfile
from services.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)


class KPIAgentError(Exception):
    """Raised when the KPI agent cannot produce a valid response."""


class KPIAgent:
    """Generates KPI suggestions from a DataProfile.

    Parameters
    ----------
    llm_client:
        An :class:`~services.llm_client.LLMClient` instance.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client
        self._system_prompt = PROMPT_KPI_SYSTEM.read_text(encoding="utf-8")
        self._suggest_template = PROMPT_KPI_SUGGEST.read_text(encoding="utf-8")
        self._fix_template = PROMPT_KPI_FIX.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def suggest(
        self,
        profile: DataProfile,
        domain_context: str,
    ) -> list[KPI]:
        """Ask the LLM for KPI suggestions and return validated KPIs.

        Parameters
        ----------
        profile:
            DataProfile of the cleaned DataFrame.
        domain_context:
            Domain string from DomainDetector or entered by the user.

        Returns
        -------
        list[KPI]
            Validated KPIs referencing only columns in the profile.

        Raises
        ------
        KPIAgentError
            When the LLM fails or the response is invalid.
        """
        user_prompt = self._build_user_prompt(profile, domain_context)

        try:
            raw: dict = self._client.call(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                model=MODEL_SONNET,
                max_tokens=MAX_TOKENS_KPI,
                expect_json=True,
                temperature=TEMPERATURE_KPI_SUGGEST,
            )
        except LLMError as exc:
            raise KPIAgentError(f"LLM call failed: {exc}") from exc

        return self._parse_and_validate(raw, profile)

    def fix_kpi(self, kpi: KPI, traceback: str) -> str:
        """Ask the LLM to fix a failing KPI computation.

        Returns corrected Python code string (not yet executed).

        Raises
        ------
        KPIAgentError
            When the LLM fails.
        """
        user_prompt = self._fix_template.format(
            kpi_name=kpi.name,
            formula_description=kpi.formula_description,
            original_code=kpi.code,
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
            raise KPIAgentError(f"Fix LLM call failed: {exc}") from exc

        return fixed_code.strip()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_user_prompt(self, profile: DataProfile, domain_context: str) -> str:
        return self._suggest_template.format(
            domain_context=domain_context,
            profile_json=profile.model_dump_json(indent=2),
        )

    def _parse_and_validate(self, raw: dict, profile: DataProfile) -> list[KPI]:
        known_columns = {col.name for col in profile.schema_report}

        try:
            if "kpis" not in raw:
                raise ValueError("Response missing required 'kpis' key")
            kpis_raw = raw["kpis"]
            if not isinstance(kpis_raw, list):
                raise ValueError("'kpis' must be a list")
            kpis = [KPI.model_validate(k) for k in kpis_raw]
        except Exception as exc:
            raise KPIAgentError(
                f"LLM response did not match KPI schema: {exc}\nRaw: {raw!r}"
            ) from exc

        valid: list[KPI] = []
        for kpi in kpis:
            # Check code references only known columns
            bad_cols = _referenced_unknown_columns(kpi.code, known_columns)
            if bad_cols:
                logger.warning(
                    "Dropping KPI %r — code references unknown columns: %s",
                    kpi.name, bad_cols,
                )
                continue
            valid.append(kpi)

        return valid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _referenced_unknown_columns(code: str, known: set[str]) -> list[str]:
    """Return column names mentioned via df['col'] that are not in known."""
    import re
    # Match df['colname'] and df["colname"]
    pattern = re.compile(r"""df\[['"]([^'"]+)['"]\]""")
    referenced = set(pattern.findall(code))
    return sorted(referenced - known)
