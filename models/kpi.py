"""Pydantic models for KPIs produced by agents/kpi_agent.py."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, field_validator


class ChartConfig(BaseModel):
    chart_type: Literal["line", "bar", "scatter", "pie", "histogram", "heatmap"]
    x_axis: str | None = None
    y_axis: str | None = None
    title: str


def _to_python(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars to native Python types."""
    # Import lazily so the model module has no hard numpy dependency at import time
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
    except ImportError:
        pass
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    return value


class KPI(BaseModel):
    """A single business KPI with its formula and chart configuration."""

    name: str = Field(description="Display name, e.g. 'Average Transaction Value'")
    category: str = Field(description="Grouping, e.g. 'Revenue', 'Retention', 'Engagement'")
    formula_description: str = Field(description="Human-readable formula")
    code: str = Field(description="Executable pandas code; 'df' is the cleaned DataFrame")
    unit: str = Field(
        default="",
        description="Display unit: '$' for money, '%' for rates/percentages, 'days' for durations, '' for counts",
    )
    computed_value: Any = Field(
        default=None,
        description="Filled in after executor runs the code — scalar, list, or dict",
    )
    business_value: str = Field(description="Why this KPI matters (1-2 sentences)")
    priority: Literal["high", "medium", "low"]
    chart_config: ChartConfig

    @field_validator("computed_value", mode="before")
    @classmethod
    def coerce_numpy(cls, v: Any) -> Any:
        """Convert numpy/pandas types on assignment (direct construction)."""
        return _to_python(v)

    @field_serializer("computed_value")
    def serialize_computed_value(self, v: Any) -> Any:
        """Convert numpy/pandas types at serialisation time (catches model_copy paths)."""
        return _to_python(v)
