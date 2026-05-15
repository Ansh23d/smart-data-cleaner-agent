"""Pydantic models for the CleaningPlan produced by agents/cleaning_agent.py."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CleaningOperationType(str, Enum):
    DROP_DUPLICATES = "drop_duplicates"
    RENAME_COLUMNS = "rename_columns"
    FILL_MISSING = "fill_missing"
    DROP_MISSING = "drop_missing"
    DROP_COLUMN = "drop_column"
    FIX_DTYPE = "fix_dtype"
    FIX_WHITESPACE = "fix_whitespace"
    FIX_MIXED_TYPES = "fix_mixed_types"
    CLIP_OUTLIERS = "clip_outliers"
    STANDARDIZE_CATEGORIES = "standardize_categories"


class CleaningOperation(BaseModel):
    """A single proposed cleaning step."""

    id: str = Field(description="Unique slug, e.g. 'fill_revenue_median'")
    type: CleaningOperationType
    description: str = Field(description="Human-readable explanation of what this does")
    impact: str = Field(description="Expected effect, e.g. 'Fills 45 nulls in revenue'")
    confidence: Literal["high", "medium", "low"]
    reversible: bool
    code: str = Field(description="Executable pandas code; 'df' is the DataFrame variable")
    column_names: list[str] = Field(
        default_factory=list,
        description="Columns this operation touches — validated against the DataProfile",
    )
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative code strings shown for medium/low-confidence operations",
    )


class CleaningPlan(BaseModel):
    """Ordered list of cleaning operations returned by the cleaning agent."""

    operations: list[CleaningOperation]

    def operations_by_type(self, op_type: CleaningOperationType) -> list[CleaningOperation]:
        return [op for op in self.operations if op.type == op_type]

    def filter_by_ids(self, ids: list[str]) -> "CleaningPlan":
        """Return a new CleaningPlan containing only operations with the given IDs."""
        kept = [op for op in self.operations if op.id in ids]
        return CleaningPlan(operations=kept)
