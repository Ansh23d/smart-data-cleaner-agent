"""Tests for pipeline/domain_detector.py."""

from __future__ import annotations

import pytest

from pipeline.domain_detector import DomainResult, detect, _confidence, _normalise, _score


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_lowercase(self):
        assert _normalise("Revenue") == "revenue"

    def test_underscores_become_spaces(self):
        assert _normalise("transaction_amount") == "transaction amount"

    def test_hyphens_become_spaces(self):
        assert _normalise("order-id") == "order id"

    def test_mixed_separators(self):
        assert _normalise("first_name-Last") == "first name last"

    def test_leading_trailing_stripped(self):
        assert _normalise("  salary  ") == "salary"


# ---------------------------------------------------------------------------
# _confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_zero_cols_returns_low(self):
        assert _confidence(3, 0) == "low"

    def test_high_score_returns_high(self):
        assert _confidence(4, 10) == "high"

    def test_high_ratio_returns_high(self):
        assert _confidence(2, 4) == "high"  # ratio = 0.5 >= 0.4

    def test_medium_score(self):
        assert _confidence(2, 20) == "medium"  # score >= 2 but ratio < 0.2

    def test_low_score(self):
        assert _confidence(1, 20) == "low"


# ---------------------------------------------------------------------------
# detect — domain identification
# ---------------------------------------------------------------------------

class TestDetect:
    def test_returns_domain_result(self):
        result = detect(["transaction_amount", "merchant", "credit_limit"])
        assert isinstance(result, DomainResult)

    def test_financial_detected(self):
        cols = ["transaction_amount", "merchant_name", "credit_limit", "card_number", "balance"]
        result = detect(cols)
        assert result.top is not None
        assert "Financial" in result.top.domain

    def test_ecommerce_detected(self):
        cols = ["order_id", "product_name", "quantity", "unit_price", "shipping_address"]
        result = detect(cols)
        assert result.top is not None
        assert "E-commerce" in result.top.domain or "Retail" in result.top.domain

    def test_healthcare_detected(self):
        cols = ["patient_id", "diagnosis", "admission_date", "medication", "doctor_name"]
        result = detect(cols)
        assert result.top is not None
        assert "Healthcare" in result.top.domain

    def test_hr_detected(self):
        cols = ["employee_id", "department", "salary", "hire_date", "performance_rating"]
        result = detect(cols)
        assert result.top is not None
        assert "HR" in result.top.domain

    def test_marketing_detected(self):
        cols = ["campaign_id", "lead_source", "conversion_rate", "email_open_rate", "churn"]
        result = detect(cols)
        assert result.top is not None
        assert "Marketing" in result.top.domain or "CRM" in result.top.domain

    def test_saas_detected(self):
        cols = ["user_id", "session_count", "mrr", "churn_rate", "dau", "subscription_plan"]
        result = detect(cols)
        assert result.top is not None
        assert "SaaS" in result.top.domain or "Product" in result.top.domain

    def test_no_match_returns_empty(self):
        result = detect(["a", "b", "c", "x", "y"])
        assert result.guesses == []
        assert result.top is None

    def test_returns_at_most_two_guesses(self):
        cols = ["transaction", "amount", "patient", "diagnosis", "order", "product"]
        result = detect(cols)
        assert len(result.guesses) <= 2

    def test_guesses_ordered_by_score(self):
        # 5 financial keywords vs 1 healthcare keyword
        cols = ["transaction", "amount", "credit", "balance", "payment", "diagnosis"]
        result = detect(cols)
        if len(result.guesses) >= 2:
            assert result.guesses[0].score >= result.guesses[1].score

    def test_empty_columns_returns_empty(self):
        result = detect([])
        assert result.guesses == []

    def test_single_strong_match(self):
        cols = ["salary", "employee_id", "department", "bonus", "hire_date",
                "termination_date", "performance"]
        result = detect(cols)
        assert result.top is not None
        assert result.top.confidence in ("high", "medium")

    def test_context_string_non_empty(self):
        cols = ["transaction_amount", "merchant", "credit_limit"]
        result = detect(cols)
        assert len(result.context_string) > 0

    def test_context_string_empty_guesses(self):
        result = detect(["foo", "bar"])
        assert result.context_string == "Unknown domain"

    def test_context_string_contains_domain(self):
        cols = ["order_id", "product_name", "quantity", "price", "sku"]
        result = detect(cols)
        if result.top:
            assert result.top.domain in result.context_string

    def test_confidence_high_for_strong_match(self):
        cols = ["transaction_amount", "merchant", "credit_limit", "card_number",
                "balance", "payment", "fraud_flag"]
        result = detect(cols)
        assert result.top is not None
        assert result.top.confidence == "high"

    def test_confidence_low_for_weak_match(self):
        # Only one keyword hit
        cols = ["date", "value", "amount", "flag"]
        result = detect(cols)
        if result.top:
            # 'amount' matches financial; with 4 cols: ratio=1/4=0.25 → medium
            assert result.top.confidence in ("low", "medium")
