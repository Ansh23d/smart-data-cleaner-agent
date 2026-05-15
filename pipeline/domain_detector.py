"""Heuristic domain detector — no LLM required.

Scores a DataFrame's column names against keyword patterns to infer the
most likely business domain.

Public API
----------
detect(columns) → DomainResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

Confidence = Literal["high", "medium", "low"]


@dataclass
class DomainGuess:
    domain: str
    score: int          # raw keyword hit count
    confidence: Confidence


@dataclass
class DomainResult:
    """Top-2 domain guesses ordered by score (highest first)."""

    guesses: list[DomainGuess] = field(default_factory=list)

    @property
    def top(self) -> DomainGuess | None:
        return self.guesses[0] if self.guesses else None

    @property
    def context_string(self) -> str:
        """Human-readable summary for use in LLM prompts."""
        if not self.guesses:
            return "Unknown domain"
        parts = [f"{g.domain} ({g.confidence} confidence)" for g in self.guesses]
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Domain keyword map
# Each entry: (domain_name, [keyword_patterns])
# Patterns are matched as whole words (word-boundary regex) against each
# column name lowercased.
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "Financial Services / Credit Card",
        [
            "transaction", "amount", "merchant", "credit", "debit", "balance",
            "limit", "account", "card", "payment", "fraud", "charge", "fee",
            "interest", "loan", "bank", "currency", "exchange", "transfer",
            "statement", "billing",
        ],
    ),
    (
        "E-commerce / Retail",
        [
            "order", "product", "quantity", "price", "sku", "cart", "basket",
            "shipping", "delivery", "refund", "return", "discount", "coupon",
            "inventory", "stock", "catalogue", "catalog", "vendor", "supplier",
            "purchase", "sale", "unit", "item",
        ],
    ),
    (
        "Healthcare",
        [
            "patient", "diagnosis", "icd", "admission", "discharge", "medication",
            "prescription", "dosage", "symptom", "condition", "procedure",
            "doctor", "physician", "nurse", "hospital", "clinic", "lab",
            "test", "result", "vital", "blood", "weight", "height", "bmi",
            "insurance", "claim", "treatment",
        ],
    ),
    (
        "HR / People Analytics",
        [
            "employee", "staff", "headcount", "department", "salary", "wage",
            "bonus", "hire", "termination", "tenure", "performance", "rating",
            "review", "leave", "absence", "overtime", "payroll", "benefit",
            "role", "title", "manager", "team", "attrition", "retention",
            "recruitment", "onboard",
        ],
    ),
    (
        "Marketing / CRM",
        [
            "campaign", "lead", "conversion", "impression", "click", "ctr",
            "cpc", "cpm", "roas", "channel", "source", "medium", "utm",
            "email", "open", "bounce", "unsubscribe", "segment", "cohort",
            "lifetime", "ltv", "churn", "nps", "satisfaction", "survey",
            "customer", "contact", "prospect",
        ],
    ),
    (
        "SaaS / Product Analytics",
        [
            "user", "session", "event", "pageview", "feature", "subscription",
            "plan", "trial", "mrr", "arr", "churn", "dau", "mau", "wau",
            "retention", "activation", "engagement", "funnel", "onboard",
            "login", "signup", "install", "device", "platform", "version",
        ],
    ),
    (
        "Logistics / Supply Chain",
        [
            "shipment", "freight", "carrier", "route", "warehouse", "location",
            "origin", "destination", "weight", "volume", "tracking", "status",
            "dispatch", "transit", "delay", "eta", "lead_time", "reorder",
            "supplier", "vendor", "po", "invoice", "customs",
        ],
    ),
    (
        "Real Estate",
        [
            "property", "listing", "price", "sqft", "bedroom", "bathroom",
            "zip", "zipcode", "postcode", "neighborhood", "city", "address",
            "rent", "mortgage", "appraisal", "agent", "broker", "lease",
            "vacancy", "occupancy",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(columns: list[str]) -> DomainResult:
    """Score column names against keyword patterns and return top-2 guesses.

    Parameters
    ----------
    columns:
        List of column name strings (raw or snake_case).

    Returns
    -------
    DomainResult
        Up to 2 guesses ordered by score, each with a confidence level.
        If no keywords match, returns an empty guesses list.
    """
    normalised = [_normalise(c) for c in columns]
    scores: dict[str, int] = {}

    for domain, keywords in _DOMAIN_KEYWORDS:
        score = _score(normalised, keywords)
        if score > 0:
            scores[domain] = score

    if not scores:
        return DomainResult(guesses=[])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top2 = ranked[:2]

    guesses = [
        DomainGuess(
            domain=domain,
            score=score,
            confidence=_confidence(score, len(columns)),
        )
        for domain, score in top2
    ]

    return DomainResult(guesses=guesses)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(col: str) -> str:
    """Lowercase and replace separators with spaces for uniform matching."""
    return re.sub(r"[_\-\s]+", " ", col.lower()).strip()


def _score(normalised_cols: list[str], keywords: list[str]) -> int:
    """Count how many distinct keywords appear across all column names."""
    hits = 0
    for keyword in keywords:
        kw = keyword.replace("_", " ")
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b")
        if any(pattern.search(col) for col in normalised_cols):
            hits += 1
    return hits


def _confidence(score: int, n_cols: int) -> Confidence:
    """Map raw hit count to a confidence level."""
    if n_cols == 0:
        return "low"
    ratio = score / max(n_cols, 1)
    if score >= 4 or ratio >= 0.4:
        return "high"
    if score >= 2 or ratio >= 0.2:
        return "medium"
    return "low"
