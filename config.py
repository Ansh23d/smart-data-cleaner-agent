"""All application-wide constants. Import from here — never hardcode values elsewhere."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
UPLOADS_DIR = DATA_DIR / "uploads"
CLEANED_DIR = DATA_DIR / "cleaned"
SESSIONS_DIR = DATA_DIR / "sessions"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = DATA_DIR / "logs"

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Token limits
# ---------------------------------------------------------------------------
MAX_TOKENS_CLEANING_PLAN = 4096
MAX_TOKENS_KPI = 4096
MAX_TOKENS_INSIGHTS = 2048
MAX_TOKENS_CODE_FIX = 1024

# ---------------------------------------------------------------------------
# File size limits
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024
WARN_FILE_SIZE_BYTES = 100 * 1024 * 1024

# ---------------------------------------------------------------------------
# Supported file formats
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

# ---------------------------------------------------------------------------
# Encoding fallback order
# ---------------------------------------------------------------------------
ENCODING_FALLBACKS = ["utf-8", "latin-1", "cp1252"]

# ---------------------------------------------------------------------------
# Executor sandbox
# ---------------------------------------------------------------------------
EXECUTOR_TIMEOUT_SECONDS = 30
EXECUTOR_MAX_RETRIES = 3
BLOCKED_CODE_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import requests",
    "import urllib",
    "open(",
    "__import__",
    "eval(",
    "exec(",
]

# ---------------------------------------------------------------------------
# LLM retry config
# ---------------------------------------------------------------------------
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAYS = [1, 2, 4]

# ---------------------------------------------------------------------------
# LLM temperature defaults
# ---------------------------------------------------------------------------
TEMPERATURE_CLEANING_PLAN = 0.1
TEMPERATURE_KPI_SUGGEST   = 0.1
TEMPERATURE_CODE_FIX      = 0.0
TEMPERATURE_INSIGHTS      = 0.15

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
ROW_DROP_CRITICAL_PCT = 0.50
ROW_DROP_WARNING_PCT = 0.20
MEAN_SHIFT_STD_WARNING = 2.0
NULL_FLAG_THRESHOLD_PCT = 0.90

# ---------------------------------------------------------------------------
# Profiler config
# ---------------------------------------------------------------------------
SAMPLE_ROWS_COUNT = 5
HIGH_CARDINALITY_RATIO = 0.95
CONSTANT_COLUMN_UNIQUE = 1
TOP_CATEGORIES_COUNT = 5

# ---------------------------------------------------------------------------
# Cleaning execution order
# ---------------------------------------------------------------------------
CLEANING_OPERATION_ORDER = [
    "rename_columns",
    "fix_whitespace",
    "fix_dtype",
    "fix_mixed_types",
    "fill_missing",
    "drop_missing",
    "drop_duplicates",
    "drop_column",
    "clip_outliers",
    "standardize_categories",
]

# ---------------------------------------------------------------------------
# Prompt template paths
# ---------------------------------------------------------------------------
PROMPTS_DIR = ROOT_DIR / "agents" / "prompts"
PROMPT_CLEANING_SYSTEM = PROMPTS_DIR / "cleaning_system.txt"
PROMPT_CLEANING_PLAN   = PROMPTS_DIR / "cleaning_plan.txt"
PROMPT_CLEANING_FIX    = PROMPTS_DIR / "cleaning_fix.txt"
PROMPT_KPI_SYSTEM      = PROMPTS_DIR / "kpi_system.txt"
PROMPT_KPI_SUGGEST     = PROMPTS_DIR / "kpi_suggest.txt"
PROMPT_KPI_FIX         = PROMPTS_DIR / "kpi_fix.txt"
PROMPT_INSIGHT_SYSTEM  = PROMPTS_DIR / "insight_system.txt"
PROMPT_INSIGHT_GENERATE = PROMPTS_DIR / "insight_generate.txt"

# ---------------------------------------------------------------------------
# Cost per million tokens (USD)
# ---------------------------------------------------------------------------
COST_PER_M_INPUT = {
    MODEL_SONNET: 3.00,
    MODEL_HAIKU: 0.25,
}
COST_PER_M_OUTPUT = {
    MODEL_SONNET: 15.00,
    MODEL_HAIKU: 1.25,
}
