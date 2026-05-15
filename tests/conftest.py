"""Shared pytest fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Alice", None],
            "revenue": [100.0, 200.0, None, 400.0, 500.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "  First Name  ": ["Alice", "Bob", None, "Alice", ""],
            "orderDate": ["2023-01-01", "not-a-date", "2023-03-01", "2023-01-01", None],
            "Revenue (USD)": ["100", "200.5", "abc", None, "500"],
            "Status": ["active", "Active", "ACTIVE", "inactive", None],
        }
    )
