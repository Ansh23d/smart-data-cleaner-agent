"""Tests for pipeline/standardizer.py — all edge cases from the spec plus extras."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.standardizer import build_mapping, standardize, to_snake_case


# ---------------------------------------------------------------------------
# to_snake_case — spec-mandated cases
# ---------------------------------------------------------------------------

class TestSpecCases:
    """Every example explicitly listed in the task spec."""

    def test_space_separated_words(self):
        assert to_snake_case("First Name") == "first_name"

    def test_camel_case(self):
        assert to_snake_case("orderDate") == "order_date"

    def test_upper_with_hyphen(self):
        assert to_snake_case("ORDER-ID") == "order_id"

    def test_spaces_around(self):
        assert to_snake_case("  Spaces Around  ") == "spaces_around"

    def test_parentheses_and_unit(self):
        assert to_snake_case("Column (USD)") == "column_usd"

    def test_collapse_double_underscores(self):
        assert to_snake_case("column__name") == "column_name"

    def test_numeric_start_prepends_col(self):
        assert to_snake_case("123_numeric_start") == "col_123_numeric_start"


# ---------------------------------------------------------------------------
# to_snake_case — camelCase / PascalCase
# ---------------------------------------------------------------------------

class TestCamelAndPascal:
    def test_simple_camel(self):
        assert to_snake_case("camelCase") == "camel_case"

    def test_pascal_case(self):
        assert to_snake_case("PascalCase") == "pascal_case"

    def test_multi_word_pascal(self):
        assert to_snake_case("FirstNameLast") == "first_name_last"

    def test_all_caps(self):
        assert to_snake_case("ORDERID") == "orderid"

    def test_all_caps_with_sep(self):
        assert to_snake_case("ORDER_ID") == "order_id"

    def test_caps_run_then_lower(self):
        # "ABCDef" → "ABC_Def" → "abc_def"
        assert to_snake_case("ABCDef") == "abc_def"

    def test_already_snake(self):
        assert to_snake_case("already_snake") == "already_snake"

    def test_mixed_case_with_numbers(self):
        assert to_snake_case("myColumn2Name") == "my_column2_name"


# ---------------------------------------------------------------------------
# to_snake_case — special characters
# ---------------------------------------------------------------------------

class TestSpecialCharacters:
    def test_hyphen(self):
        assert to_snake_case("my-column") == "my_column"

    def test_period(self):
        assert to_snake_case("my.column") == "my_column"

    def test_slash(self):
        assert to_snake_case("revenue/cost") == "revenue_cost"

    def test_ampersand(self):
        assert to_snake_case("profit & loss") == "profit_loss"

    def test_percent_sign(self):
        assert to_snake_case("growth %") == "growth"

    def test_parentheses_only(self):
        assert to_snake_case("(value)") == "value"

    def test_brackets(self):
        assert to_snake_case("value [USD]") == "value_usd"

    def test_multiple_special_chars(self):
        assert to_snake_case("col--name__test") == "col_name_test"

    def test_hash(self):
        assert to_snake_case("order #") == "order"

    def test_dollar_sign(self):
        assert to_snake_case("$amount") == "amount"


# ---------------------------------------------------------------------------
# to_snake_case — whitespace
# ---------------------------------------------------------------------------

class TestWhitespace:
    def test_leading_trailing_spaces(self):
        assert to_snake_case("  col  ") == "col"

    def test_tab_separated(self):
        assert to_snake_case("col\tname") == "col_name"

    def test_multiple_internal_spaces(self):
        assert to_snake_case("col   name") == "col_name"

    def test_newline_in_name(self):
        assert to_snake_case("col\nname") == "col_name"


# ---------------------------------------------------------------------------
# to_snake_case — numeric starts
# ---------------------------------------------------------------------------

class TestNumericStart:
    def test_starts_with_single_digit(self):
        assert to_snake_case("1_column") == "col_1_column"

    def test_starts_with_multi_digit(self):
        assert to_snake_case("123abc") == "col_123abc"

    def test_digit_only(self):
        assert to_snake_case("42") == "col_42"

    def test_digit_with_special(self):
        assert to_snake_case("1st place") == "col_1st_place"


# ---------------------------------------------------------------------------
# to_snake_case — edge / degenerate inputs
# ---------------------------------------------------------------------------

class TestEdgeInputs:
    def test_empty_string(self):
        assert to_snake_case("") == "unnamed"

    def test_only_spaces(self):
        assert to_snake_case("   ") == "unnamed"

    def test_only_special_chars(self):
        assert to_snake_case("---") == "unnamed"

    def test_only_underscores(self):
        assert to_snake_case("___") == "unnamed"

    def test_single_letter(self):
        assert to_snake_case("X") == "x"

    def test_single_word_lowercase(self):
        assert to_snake_case("name") == "name"

    def test_unicode_letters_pass_through(self):
        # Non-ASCII letters are replaced by underscores, stripped
        result = to_snake_case("café")
        assert result == "caf"  # 'é' → '_', stripped from end

    def test_already_valid_snake(self):
        assert to_snake_case("valid_snake_case") == "valid_snake_case"


# ---------------------------------------------------------------------------
# build_mapping — duplicate disambiguation
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_no_duplicates(self):
        mapping = build_mapping(["Alpha", "Beta", "Gamma"])
        assert mapping == {"Alpha": "alpha", "Beta": "beta", "Gamma": "gamma"}

    def test_two_colliding_names(self):
        # "First Name" and "first_name" both → "first_name"
        mapping = build_mapping(["First Name", "first_name"])
        assert mapping["First Name"] == "first_name"
        assert mapping["first_name"] == "first_name_2"

    def test_three_colliding_names(self):
        mapping = build_mapping(["col", "Col", "COL"])
        assert mapping["col"] == "col"
        assert mapping["Col"] == "col_2"
        assert mapping["COL"] == "col_3"

    def test_four_collisions(self):
        cols = ["Revenue", "revenue", "REVENUE", "Revenue "]
        mapping = build_mapping(cols)
        results = list(mapping.values())
        assert results[0] == "revenue"
        assert results[1] == "revenue_2"
        assert results[2] == "revenue_3"
        assert results[3] == "revenue_4"

    def test_no_collision_when_results_differ(self):
        mapping = build_mapping(["First Name", "LastName"])
        assert mapping["First Name"] == "first_name"
        assert mapping["LastName"] == "last_name"

    def test_numeric_start_collision(self):
        mapping = build_mapping(["1_col", "col_1_col"])
        # "1_col" → "col_1_col", "col_1_col" → "col_1_col" → collide
        assert mapping["1_col"] == "col_1_col"
        assert mapping["col_1_col"] == "col_1_col_2"

    def test_empty_string_collision(self):
        mapping = build_mapping(["---", "___", "   "])
        assert mapping["---"] == "unnamed"
        assert mapping["___"] == "unnamed_2"
        assert mapping["   "] == "unnamed_3"


# ---------------------------------------------------------------------------
# standardize() — DataFrame-level
# ---------------------------------------------------------------------------

class TestStandardize:
    def test_returns_renamed_df_and_mapping(self):
        df = pd.DataFrame({"First Name": [1], "Last Name": [2]})
        renamed, mapping = standardize(df)
        assert list(renamed.columns) == ["first_name", "last_name"]
        assert mapping == {"First Name": "first_name", "Last Name": "last_name"}

    def test_original_df_is_not_mutated(self):
        df = pd.DataFrame({"My Col": [1, 2, 3]})
        original_cols = list(df.columns)
        standardize(df)
        assert list(df.columns) == original_cols

    def test_data_is_preserved(self):
        df = pd.DataFrame({"Revenue (USD)": [100, 200, 300]})
        renamed, _ = standardize(df)
        assert list(renamed["revenue_usd"]) == [100, 200, 300]

    def test_spec_fixture(self):
        """Run all spec-example columns through the full standardize path."""
        cols = [
            "First Name", "orderDate", "ORDER-ID",
            "  Spaces Around  ", "Column (USD)", "column__name",
        ]
        df = pd.DataFrame(columns=cols)
        _, mapping = standardize(df)
        assert mapping["First Name"] == "first_name"
        assert mapping["orderDate"] == "order_date"
        assert mapping["ORDER-ID"] == "order_id"
        assert mapping["  Spaces Around  "] == "spaces_around"
        assert mapping["Column (USD)"] == "column_usd"
        assert mapping["column__name"] == "column_name"

    def test_numeric_start_in_df(self):
        df = pd.DataFrame({"123_numeric_start": [1]})
        _, mapping = standardize(df)
        assert mapping["123_numeric_start"] == "col_123_numeric_start"

    def test_empty_df_no_error(self):
        df = pd.DataFrame()
        renamed, mapping = standardize(df)
        assert list(renamed.columns) == []
        assert mapping == {}

    def test_duplicate_columns_in_df(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["Col", "col", "COL"])
        renamed, mapping = standardize(df)
        assert list(renamed.columns) == ["col", "col_2", "col_3"]

    def test_mapping_keys_match_original_columns(self):
        df = pd.DataFrame({"A B": [1], "C D": [2], "E F": [3]})
        _, mapping = standardize(df)
        assert set(mapping.keys()) == set(df.columns)
