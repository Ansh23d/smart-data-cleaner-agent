"""File ingestion — accepts a file path or raw bytes, returns a pandas DataFrame.

Supported formats: CSV, XLSX/XLS, JSON, Parquet.
Format is auto-detected from the file extension first, then magic bytes.
CSV encoding is attempted in order: UTF-8 → Latin-1 → cp1252.
Files over MAX_FILE_SIZE_BYTES (500 MB) are rejected before any parsing.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

from config import (
    ENCODING_FALLBACKS,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
)

# Magic-byte signatures → canonical extension
_MAGIC: list[tuple[bytes, str]] = [
    (b"PK\x03\x04", ".xlsx"),           # ZIP-based (xlsx, xlsm, …)
    (b"\xd0\xcf\x11\xe0", ".xls"),      # Legacy OLE2 (xls)
    (b"PAR1", ".parquet"),
    (b"\x89HDF", ".hdf"),
]


class IngestionError(Exception):
    """Raised for any user-facing ingestion failure."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(source: str | Path | bytes, filename: str | None = None) -> pd.DataFrame:
    """Parse *source* into a DataFrame.

    Parameters
    ----------
    source:
        Either a file-system path (str / Path) or raw file bytes
        (e.g. from ``st.file_uploader``).
    filename:
        Original file name — required when *source* is bytes so that the
        extension can be used as a format hint.

    Returns
    -------
    pd.DataFrame
        Non-empty DataFrame with at least one column.

    Raises
    ------
    IngestionError
        For every failure case with a user-readable message.
    """
    raw: bytes
    ext: str

    if isinstance(source, (str, Path)):
        path = Path(source)
        _check_size_path(path)
        raw = path.read_bytes()
        ext = path.suffix.lower()
    elif isinstance(source, bytes):
        if len(source) > MAX_FILE_SIZE_BYTES:
            raise IngestionError(
                f"File is too large ({len(source) / 1_048_576:.0f} MB). "
                f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // 1_048_576} MB."
            )
        raw = source
        ext = Path(filename).suffix.lower() if filename else ""
    else:
        raise TypeError(f"source must be str, Path, or bytes — got {type(source)}")

    fmt = _detect_format(raw, ext, filename or "")
    df = _parse(raw, fmt)
    _validate(df)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_size_path(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise IngestionError(
            f"File is too large ({size / 1_048_576:.0f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // 1_048_576} MB."
        )


def _detect_format(raw: bytes, ext: str, filename: str) -> str:
    """Return a canonical format string: 'csv', 'xlsx', 'xls', 'json', 'parquet'."""
    # Extension wins if it's unambiguous and supported
    if ext in SUPPORTED_EXTENSIONS:
        # For .xlsx / .xls also confirm via magic to catch mislabelled files
        if ext in (".xlsx", ".xls"):
            magic_ext = _magic_ext(raw)
            if magic_ext in (".xlsx", ".xls"):
                return magic_ext.lstrip(".")  # trust magic for Office formats
        return ext.lstrip(".")

    # Fall back to magic bytes
    magic_ext = _magic_ext(raw)
    if magic_ext:
        return magic_ext.lstrip(".")

    # Give up
    suffix_display = ext or "(no extension)"
    raise IngestionError(
        f"Unsupported file format '{suffix_display}'. "
        f"Supported formats: CSV, XLSX, JSON, Parquet. "
        f"Re-export your file in one of these formats and try again."
    )


def _magic_ext(raw: bytes) -> str:
    """Return an extension string from magic bytes, or empty string."""
    head = raw[:8]
    for magic, ext in _MAGIC:
        if head.startswith(magic):
            return ext
    return ""


def _parse(raw: bytes, fmt: str) -> pd.DataFrame:
    buf = io.BytesIO(raw)

    if fmt == "csv":
        return _parse_csv(raw)

    if fmt in ("xlsx", "xls"):
        try:
            return pd.read_excel(buf, engine="openpyxl" if fmt == "xlsx" else "xlrd")
        except Exception as exc:
            raise IngestionError(
                f"Could not read the Excel file. It may be corrupt or password-protected. "
                f"Try re-saving it as .xlsx and upload again. (Detail: {exc})"
            ) from exc

    if fmt == "json":
        try:
            # Try records orientation first, then default pandas orient
            try:
                return pd.read_json(buf, orient="records")
            except ValueError:
                buf.seek(0)
                return pd.read_json(buf)
        except Exception as exc:
            raise IngestionError(
                f"Could not parse the JSON file. Ensure it is an array of objects "
                f"or a pandas-compatible JSON structure. (Detail: {exc})"
            ) from exc

    if fmt == "parquet":
        try:
            return pd.read_parquet(buf)
        except Exception as exc:
            raise IngestionError(
                f"Could not read the Parquet file. It may be corrupt or use an "
                f"unsupported compression codec. (Detail: {exc})"
            ) from exc

    raise IngestionError(f"Internal error: unhandled format '{fmt}'.")


def _parse_csv(raw: bytes) -> pd.DataFrame:
    """Try each encoding in ENCODING_FALLBACKS; raise IngestionError on total failure."""
    last_exc: Exception | None = None
    for encoding in ENCODING_FALLBACKS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            raise IngestionError(
                f"Could not parse the CSV file. It may be corrupt or malformed. "
                f"(Detail: {exc})"
            ) from exc

    raise IngestionError(
        f"Could not decode the CSV file. Tried encodings: "
        f"{', '.join(ENCODING_FALLBACKS)}. "
        f"Try re-exporting as UTF-8 CSV from Excel (File → Save As → CSV UTF-8)."
    )


def _validate(df: pd.DataFrame) -> None:
    if df.empty and len(df.columns) == 0:
        raise IngestionError(
            "The file appears to be empty (no rows and no columns). "
            "Upload a file that contains at least a header row."
        )
    if len(df.columns) == 0:
        raise IngestionError(
            "The file has no columns. "
            "Make sure the file is not empty and has a valid header row."
        )
    if len(df) == 0:
        # Header-only file: return the empty DataFrame — callers can decide how
        # to surface this (profiler will flag it, but it's not an ingest error)
        return
