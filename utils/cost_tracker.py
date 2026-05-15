"""Session-level token and cost accumulator.

One CostTracker instance lives for the duration of a pipeline session.
LLMClient receives a tracker at construction time and records every call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import COST_PER_M_INPUT, COST_PER_M_OUTPUT


@dataclass
class CallRecord:
    """Immutable record of one LLM call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates token usage and estimated cost across LLM calls in a session."""

    records: list[CallRecord] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def record(self, model: str, input_tokens: int, output_tokens: int) -> CallRecord:
        """Compute cost, store a CallRecord, and return it."""
        cost = _compute_cost(model, input_tokens, output_tokens)
        rec = CallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self.records.append(rec)
        return rec

    def reset(self) -> None:
        """Clear all accumulated data (e.g. for a new session)."""
        self.records.clear()

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def summary(self) -> dict:
        """Human-readable dict suitable for display in the Streamlit sidebar."""
        return {
            "calls": len(self.records),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate = COST_PER_M_INPUT.get(model, 0.0)
    output_rate = COST_PER_M_OUTPUT.get(model, 0.0)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
