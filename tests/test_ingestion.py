"""Tests for pipeline/ingestion.py."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pipeline.ingestion import IngestionError, ingest


# ---------------------------------------------------------------------------
# Helpers — build raw bytes for each format
# ---------------------------------------------------------------------------

def _csv_bytes(content: str, encoding: str = "utf-8") -> bytes:
    return content.encode(encoding)


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _json_bytes(records: list[dict]) -> bytes:
    return json.dumps(records).encode()


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CSV — valid
# ---------------------------------------------------------------------------

class TestCSVValid:
    def test_basic_csv(self):
        raw = _csv_bytes("id,name,value\n1,Alice,100\n2,Bob,200\n")
        df = ingest(raw, filename="data.csv")
        assert list(df.columns) == ["id", "name", "value"]
        assert len(df) == 2

    def test_csv_from_path(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        df = ingest(p)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_csv_with_extra_whitespace(self):
        raw = _csv_bytes("col1, col2 \n hello , world \n")
        df = ingest(raw, filename="test.csv")
        assert len(df) == 1


# ---------------------------------------------------------------------------
# CSV — encoding fallback
# ---------------------------------------------------------------------------

class TestCSVEncoding:
    def test_latin1_csv(self):
        content = "name,city\nJosé,São Paulo\nMüller,München\n"
        raw = _csv_bytes(content, encoding="latin-1")
        df = ingest(raw, filename="latin.csv")
        assert len(df) == 2

    def test_cp1252_csv(self):
        content = "item\ncafé\nnaïve\n"
        raw = _csv_bytes(content, encoding="cp1252")
        df = ingest(raw, filename="win.csv")
        assert len(df) == 2

    def test_utf8_csv(self):
        raw = _csv_bytes("col\n日本語\n中文\n", encoding="utf-8")
        df = ingest(raw, filename="utf8.csv")
        assert len(df) == 2


# ---------------------------------------------------------------------------
# CSV — edge cases
# ---------------------------------------------------------------------------

class TestCSVEdgeCases:
    def test_header_only_returns_empty_df(self):
        raw = _csv_bytes("col1,col2,col3\n")
        df = ingest(raw, filename="headers_only.csv")
        assert len(df) == 0
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_single_row(self):
        raw = _csv_bytes("x\n42\n")
        df = ingest(raw, filename="single.csv")
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

class TestExcel:
    def test_valid_xlsx(self):
        source_df = pd.DataFrame({"product": ["A", "B"], "price": [10.0, 20.0]})
        raw = _excel_bytes(source_df)
        df = ingest(raw, filename="data.xlsx")
        assert list(df.columns) == ["product", "price"]
        assert len(df) == 2

    def test_corrupt_xlsx_raises(self):
        bad_bytes = b"PK\x03\x04" + b"\x00" * 100  # valid magic, garbage body
        with pytest.raises(IngestionError, match="Excel"):
            ingest(bad_bytes, filename="bad.xlsx")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class TestJSON:
    def test_valid_json_records(self):
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        raw = _json_bytes(records)
        df = ingest(raw, filename="data.json")
        assert len(df) == 2
        assert set(df.columns) == {"a", "b"}

    def test_corrupt_json_raises(self):
        with pytest.raises(IngestionError, match="JSON"):
            ingest(b"{not valid json", filename="bad.json")


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

class TestParquet:
    def test_valid_parquet(self):
        source_df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        raw = _parquet_bytes(source_df)
        df = ingest(raw, filename="data.parquet")
        assert len(df) == 3
        assert list(df.columns) == ["x", "y"]


# ---------------------------------------------------------------------------
# Empty file
# ---------------------------------------------------------------------------

class TestEmptyFile:
    def test_completely_empty_csv_raises(self):
        with pytest.raises(IngestionError):
            ingest(b"", filename="empty.csv")

    def test_completely_empty_file_path(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_bytes(b"")
        with pytest.raises(IngestionError):
            ingest(p)


# ---------------------------------------------------------------------------
# Oversized file (mock stat / len)
# ---------------------------------------------------------------------------

class TestOversizedFile:
    def test_oversized_bytes_raises(self):
        from app.config import MAX_FILE_SIZE_BYTES
        big = b"a,b\n1,2\n" + b"x" * MAX_FILE_SIZE_BYTES
        with pytest.raises(IngestionError, match="too large"):
            ingest(big, filename="big.csv")

    def test_oversized_path_raises(self, tmp_path):
        from app.config import MAX_FILE_SIZE_BYTES
        p = tmp_path / "huge.csv"
        # Mock stat so we don't actually write 500MB
        import os
        original_stat = os.stat

        class FakeStat:
            st_size = MAX_FILE_SIZE_BYTES + 1

        with patch("pipeline.ingestion.Path.stat", return_value=FakeStat()):
            # Also mock read_bytes so it doesn't fail trying to read a tiny file
            with patch("pipeline.ingestion.Path.read_bytes", return_value=b"a\n1\n"):
                p.write_text("a\n1\n")
                with pytest.raises(IngestionError, match="too large"):
                    ingest(p)


# ---------------------------------------------------------------------------
# Unsupported format
# ---------------------------------------------------------------------------

class TestUnsupportedFormat:
    def test_unknown_extension_raises(self):
        with pytest.raises(IngestionError, match="Unsupported"):
            ingest(b"some data", filename="data.xml")

    def test_no_extension_no_magic_raises(self):
        with pytest.raises(IngestionError, match="Unsupported"):
            ingest(b"this is just text", filename="data")

    def test_txt_extension_raises(self):
        with pytest.raises(IngestionError, match="Unsupported"):
            ingest(b"col\nval\n", filename="data.txt")


# ---------------------------------------------------------------------------
# Format auto-detection from magic bytes
# ---------------------------------------------------------------------------

class TestMagicDetection:
    def test_xlsx_detected_by_magic_when_no_extension(self):
        source_df = pd.DataFrame({"a": [1]})
        raw = _excel_bytes(source_df)
        # No filename extension — should still work via magic bytes
        df = ingest(raw, filename="no_ext")
        assert len(df) == 1

    def test_parquet_detected_by_magic(self):
        source_df = pd.DataFrame({"col": [1, 2]})
        raw = _parquet_bytes(source_df)
        df = ingest(raw, filename="data.parquet")
        assert len(df) == 2
