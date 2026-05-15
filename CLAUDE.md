# CLAUDE.md — AI Data Agent

## Project Overview
An AI-powered data cleaning and KPI generation agent. Users upload messy datasets, the agent
cleans them with human-in-the-loop approval, then auto-generates business KPIs and a dashboard.

## Tech Stack
- **Language:** Python 3.11+
- **LLM:** Anthropic Claude API (claude-sonnet-4-20250514 for planning, claude-haiku-4-5-20251001 for code fixes)
- **Frontend:** Streamlit (multi-page app)
- **Data processing:** pandas, numpy
- **Visualization:** Plotly (in Streamlit dashboard)
- **Data models:** Pydantic v2
- **Testing:** pytest, pytest-cov
- **Logging:** Python logging with structlog formatting

## Architecture Principles

### LLM calls are expensive — minimize them
Only two pipeline phases use the LLM: cleaning plan generation and KPI suggestion.
Everything else (profiling, standardization, validation, domain detection, execution) is
deterministic Python. Never send raw data to the LLM — always send the DataProfile summary.

### Human-in-the-loop for destructive operations
The cleaning agent PROPOSES a plan. The user APPROVES it. No rows or columns are dropped
without explicit user consent. Medium/low confidence operations show alternatives.

### Sandboxed execution
LLM-generated pandas code runs in a restricted exec() environment. Only pandas, numpy, and
datetime are available. No filesystem, network, or dangerous builtins. 30-second timeout.
Automatic rollback on failure.

### Structured LLM output
All LLM calls must return valid JSON matching a Pydantic model schema. System prompts
explicitly define the output schema with examples. JSON parsing includes: stripping markdown
fences, retry on malformed output, validation against the Pydantic model.

### Session persistence
Every pipeline state is persisted to disk as JSON. Server restarts resume from the last
completed state. Sessions are identified by UUID.

## Key Dependencies

```
anthropic>=0.40.0          # Claude API SDK
streamlit>=1.40.0          # UI framework
pandas>=2.2.0              # Data processing
numpy>=1.26.0              # Numerical operations
pydantic>=2.0.0            # Data models and validation
plotly>=5.24.0             # Dashboard charts
openpyxl>=3.1.0            # Excel read/write
python-dotenv>=1.0.0       # .env loading
structlog>=24.0.0          # Structured logging
pytest>=8.0.0              # Testing
pytest-cov>=5.0.0          # Coverage
pyarrow>=15.0.0            # Parquet support
scipy>=1.12.0              # Statistical functions (skewness, kurtosis)
```

## Data Flow

```
File Upload → Ingestion → Profiling → Cleaning Agent (LLM) → User Review
→ Sandboxed Execution → Validation → Domain Detection → KPI Agent (LLM)
→ Insight Agent (LLM) → Dashboard Render → Export
```

## Session State Machine

```
UPLOADED → PROFILED → PLAN_PROPOSED → PLAN_APPROVED → CLEANING →
VALIDATED → CONTEXT_SET → KPI_COMPUTING → COMPLETE
```

Each transition is logged with timestamp and cost. Invalid transitions raise an error.

## File Conventions
- All prompts live in `agents/prompts/*.txt` as plain text templates with {placeholders}
- All data models are Pydantic v2 models in `models/`
- All deterministic processing is in `pipeline/`
- All LLM-powered logic is in `agents/`
- Tests mirror the source structure in `tests/`
- Runtime data goes in `data/` (gitignored)

## Prompt Design Rules
- System prompts define the agent's role and output format
- User prompts include: the DataProfile JSON, the expected output schema, one complete example
- Every prompt explicitly says: "Return ONLY valid JSON. No markdown, no explanation, no preamble."
- Output schema is defined as a JSON example in the prompt, matching the Pydantic model

## Cleaning Operations — Allowed Types
The cleaning agent may only propose operations of these types:
- `drop_duplicates` — remove exact duplicate rows
- `rename_columns` — standardize to snake_case
- `fill_missing` — impute with median, mean, mode, forward-fill, constant, or interpolate
- `drop_missing` — drop rows or columns above a null threshold
- `drop_column` — remove useless columns (constant, >90% null, artifact index)
- `fix_dtype` — type coercion (string→datetime, string→numeric, etc.)
- `fix_whitespace` — strip leading/trailing whitespace
- `fix_mixed_types` — coerce mixed-type columns to dominant type
- `clip_outliers` — cap extreme values at IQR boundaries
- `standardize_categories` — normalize inconsistent category labels

Each operation includes: id, type, description, impact, confidence (high/medium/low),
reversible flag, pandas code string, and optional alternatives list.

## Execution Order (Hardcoded)
1. rename_columns
2. fix_whitespace
3. fix_dtype
4. fix_mixed_types
5. fill_missing
6. drop_missing
7. drop_duplicates
8. drop_column
9. clip_outliers
10. standardize_categories

## Sandbox Restrictions
The executor blocks these patterns in code strings before execution:
- `import os`, `import sys`, `import subprocess`
- `open(`, `__import__`, `eval(`, `exec(`
- `import socket`, `import requests`, `import urllib`

Allowed in the execution namespace:
- `pd` (pandas), `np` (numpy), `datetime`, `re`
- The DataFrame variable `df`

## Validation Thresholds
- Row drop >50% → CRITICAL (block, ask user)
- Row drop >20% → WARNING (proceed with notice)
- Mean shift >2 std deviations → WARNING
- Schema mismatch → FAIL (revert)

## KPI Output Schema
Each KPI must include:
- `name`: display name (e.g., "Average Transaction Value")
- `category`: grouping (Revenue, Retention, Engagement, etc.)
- `formula_description`: human-readable formula
- `code`: executable pandas code string
- `computed_value`: the actual computed result (number, series, or DataFrame)
- `business_value`: why this matters (1-2 sentences)
- `priority`: high / medium / low
- `chart_config`: {chart_type, x_axis, y_axis, title}

## Dashboard Layout
1. Header: dataset name, row/column count, "Auto-generated" badge
2. Scorecard row: 3-6 metric cards (value + delta)
3. Charts: 2-4 Plotly charts (line for trends, bar for categories)
4. Insights: 3-5 AI-generated text insights with priority indicators
5. Actions: "Add KPI" input, "Export Excel" button, "Export for Power BI" button

## Cost Targets
- Profiling: $0 (no LLM)
- Cleaning plan: ~$0.01-0.05 per generation (1 Sonnet call)
- Revision loops: ~$0.01-0.03 per revision (max 3)
- Code fixes: ~$0.001 per fix (Haiku)
- KPI generation: ~$0.01-0.05 (1 Sonnet call)
- Insights: ~$0.01-0.03 (1 Sonnet call)
- **Total per session: $0.05-0.20 typical**

## Testing Requirements
- Unit tests for all pipeline/ modules
- Unit tests for all models/ (serialization, validation)
- Integration tests for agents/ with mocked LLM responses
- End-to-end test: upload fixture CSV → cleaned CSV + KPIs (mocked LLM)
- Coverage target: >80% on pipeline/ and models/

## Error Messages (User-Facing)
Always be specific and actionable:
- BAD: "An error occurred"
- GOOD: "Could not parse this file. It appears to be encoded in a non-UTF-8 format. Try re-exporting as UTF-8 CSV from Excel."
- BAD: "Cleaning failed"
- GOOD: "Operation 'fill missing values in revenue' failed because the column contains non-numeric values. The agent is attempting a fix (attempt 2/3)..."

## Structured Logging
`utils/logger.py` provides `get_logger(name)` backed by structlog + stdlib rotating file handler.
- Log file: `data/logs/agent.log` (10 MB × 3 backups, JSON format)
- Import: `from utils.logger import get_logger; log = get_logger(__name__)`
- Used in: `services/llm_client.py` (every call, every retry), `pipeline/executor.py` (every execution result)
- Console output: ERROR level only (to avoid polluting Streamlit's terminal)

## LLMClient status_callback
`LLMClient.__init__` accepts `status_callback: Callable[[str], None] | None`.
During API rate-limit retries the callback is invoked once per second with a countdown message.
All pages pass `st.toast` as the callback:
```python
llm = LLMClient(tracker=tracker, status_callback=st.toast)
```

## Shared Sidebar Component
`app/components/sidebar.py` exports `render_sidebar()`.
Call it near the top of every page (after guards, before the main content).
Shows: pipeline stage progress · session LLM cost · "Start Over" button.
"Start Over" clears all `st.session_state` keys and switches to `pages/01_upload.py`.

## Edge-Case Handling (upload page)
`app/pages/01_upload.py` checks immediately after ingestion:
- 0 rows → `st.error()` + `st.stop()`
- 1 row → `st.warning()` (proceeds)
- >500 columns → `st.warning()` (proceeds, cleaning plan may be truncated)

## Retry Spinner Text
`app/pages/04_cleaning_result.py` updates the progress bar label during LLM fix retries:
`"[2/5] Fixing 'drop nulls in revenue…' (attempt 2/3)…"`

## Integration Tests
`tests/test_integration.py` — 30 tests covering the full pipeline on all 4 fixture datasets
with mocked LLM responses. Tests are parametrized over fixtures where sensible.
The sandbox restricts `__import__`, so KPI/cleaning codes must NOT use `import` statements —
`pd`, `np`, `datetime`, `re`, and `df` are pre-injected into the execution namespace.
