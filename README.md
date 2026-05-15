# AI Data Agent

An AI-powered data cleaning and KPI dashboard tool. Upload a messy CSV/Excel/JSON/Parquet file, have Claude propose a cleaning plan you approve step-by-step, then auto-generate business KPIs and an interactive dashboard — all for a few cents of API cost.

---

## Features

- **Human-in-the-loop cleaning** — Claude proposes operations; you approve, skip, or revise each one before any data is modified
- **Sandboxed execution** — LLM-generated pandas code runs in a restricted environment with a 30-second timeout and automatic rollback on failure
- **Automatic KPI generation** — domain-aware KPIs with formulas, Plotly charts, and business context
- **AI insights** — 3–5 prioritised natural-language observations about your data
- **Export** — Download cleaned data as CSV, Excel workbook (two sheets), or Power BI–ready CSV
- **Session persistence** — reload the app mid-pipeline without losing your work
- **Cost transparency** — every LLM call is tracked; typical session costs $0.05–$0.20

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd ai-data-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Anthropic API key

```bash
cp .env.example .env
# Edit .env and set:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
streamlit run app/main.py
```

The app opens at `http://localhost:8501`.

---

## Usage walkthrough

### Step 1 — Upload

Drag-and-drop or browse for a CSV, Excel (.xlsx/.xls), JSON, or Parquet file (up to 500 MB).

> **Screenshot placeholder** — upload page with file picker and size indicator

The profiler runs automatically and detects column types, null rates, duplicate rows, and data quality issues.

### Step 2 — Review profile

Inspect the auto-generated profile: schema table, null distribution, numeric statistics, and top categorical values. The domain detector guesses your industry (e.g. "E-commerce / Retail").

> **Screenshot placeholder** — profile page showing schema table and quality flag badges

### Step 3 — Review the cleaning plan

Claude proposes up to ~10 cleaning operations ordered by safety. For each operation you can:

- ✅ Keep it (default)
- ❌ Uncheck to skip
- 🔄 Pick an alternative (shown for medium/low-confidence ops)
- 💬 Reject all and give feedback to regenerate

> **Screenshot placeholder** — cleaning plan cards with confidence badges and code preview

### Step 4 — Apply cleaning

Approved operations execute in sequence with a live progress bar. Failed operations are auto-fixed (up to 3 LLM retries). A before/after comparison shows rows removed, null cells filled, and duplicates dropped.

> **Screenshot placeholder** — cleaning result with before/after metrics and execution log

### Step 5 — Configure KPIs

Confirm the detected domain or type your own, then click **Generate KPIs**. Claude proposes 5–8 domain-relevant KPIs with executable pandas code. Click **Generate insights** for 3–5 natural-language observations.

> **Screenshot placeholder** — KPI config page with scorecard preview

### Step 6 — Dashboard

The full interactive dashboard shows:

| Section | Contents |
|---------|----------|
| **Header** | File name, row/column count, LLM cost badge |
| **Scorecard** | Up to 6 KPI metric cards with priority colours |
| **Charts** | 2–4 Plotly charts (bar, line, histogram, scatter, pie, heatmap) |
| **Insights** | Priority-coloured insight cards (High / Medium / Info) |
| **Actions** | Add custom KPI · Export Excel · Export Power BI CSV |

> **Screenshot placeholder** — full dashboard with scorecard, charts, and insight cards

---

## Architecture

```
app/main.py               Landing page + sidebar (Streamlit entry point)
app/pages/
  01_upload.py            File upload → ingest → profile → session create
  02_profile.py           Show DataProfile, run domain detection
  03_cleaning_plan.py     Generate / review / approve cleaning plan
  04_cleaning_result.py   Execute plan, validate, show before/after
  05_kpi_config.py        Domain confirm, generate KPIs + insights
  06_dashboard.py         Full KPI dashboard + exports
app/components/
  profile_summary.py      Profile visualisation component
  cleaning_plan_card.py   Per-operation review card
  metric_card.py          KPI scorecard card
  chart_renderer.py       Plotly chart grid
  insight_card.py         Priority-coloured insight card
  sidebar.py              Shared sidebar (cost display, Start Over)

pipeline/
  ingestion.py            Read CSV/Excel/JSON/Parquet → DataFrame
  profiler.py             Build DataProfile (schema, nulls, stats, flags)
  domain_detector.py      Keyword-based domain inference
  executor.py             Sandboxed exec() with timeout + rollback
  validator.py            Post-cleaning health checks
  standardizer.py         Column name normalisation

agents/
  cleaning_agent.py       CleaningAgent: plan() + fix_operation()
  kpi_agent.py            KPIAgent: suggest() + fix_kpi()
  insight_agent.py        InsightAgent: generate()
  prompts/                Plain-text prompt templates

models/
  cleaning_plan.py        CleaningPlan, CleaningOperation (Pydantic v2)
  kpi.py                  KPI, ChartConfig
  insight.py              Insight
  profile.py              DataProfile, ColumnSchema, QualityFlag, …
  session.py              Session, SessionState (state machine)

services/
  llm_client.py           Anthropic SDK wrapper (retry, JSON parse, cost)
  session_store.py        JSON-file session persistence

utils/
  cost_tracker.py         Per-session token + USD accumulator
  logger.py               structlog rotating file logger
```

### Data flow

```
File bytes
  → ingestion.ingest()          # parse + type-detect
  → profiler.profile()          # build DataProfile summary
  → domain_detector.detect()    # guess industry from column names
  → CleaningAgent.plan()        # LLM call → CleaningPlan JSON
  → User approves in UI
  → executor.execute() × N      # sandboxed pandas exec, per op
  → validator.validate()        # row-drop check, schema check, …
  → KPIAgent.suggest()          # LLM call → list[KPI] JSON
  → executor.execute() × N      # run KPI code, capture values
  → InsightAgent.generate()     # LLM call → list[Insight] JSON
  → Dashboard render + export
```

### LLM calls (and when they happen)

| Call | Model | When | Typical cost |
|------|-------|------|-------------|
| Cleaning plan | Sonnet | Once per upload | $0.01–$0.05 |
| Operation fix | Haiku | On execution failure (max 3×) | ~$0.001 each |
| KPI suggestion | Sonnet | Once per session | $0.01–$0.05 |
| KPI fix | Haiku | On execution failure (max 3×) | ~$0.001 each |
| Insight generation | Sonnet | Once per session | $0.01–$0.03 |
| Revision loop | Sonnet | Per user feedback cycle | $0.01–$0.03 |

**Total per session: $0.05–$0.20 typical**

All LLM calls use structured JSON output validated against Pydantic models. On rate-limit errors the client retries with exponential backoff (1 s → 2 s → 4 s) and shows a toast notification in the UI.

---

## Session state machine

```
UPLOADED → PROFILED → PLAN_PROPOSED → PLAN_APPROVED → CLEANING
         → VALIDATED → CONTEXT_SET → KPI_COMPUTING → COMPLETE
```

Each transition is persisted to `data/sessions/{uuid}.json`. Reloading the app (or clicking the session link) restores the pipeline exactly where you left off.

---

## Testing

```bash
# All tests (includes unit + integration)
pytest

# Integration tests only
pytest tests/test_integration.py -v

# Coverage report (pipeline/ + models/ only)
pytest --cov=pipeline --cov=models --cov-report=term-missing
```

Current coverage: **97%** on `pipeline/` and `models/`.

Test matrix:

| File | Scope | Tests |
|------|-------|-------|
| `test_integration.py` | Full pipeline, all 4 fixtures | 30 |
| `test_cleaning_agent.py` | CleaningAgent | 33 |
| `test_kpi_agent.py` | KPIAgent | 33 |
| `test_insight_agent.py` | InsightAgent | 30 |
| `test_llm_client.py` | LLMClient | ~35 |
| `test_executor.py` | Sandbox executor | ~40 |
| `test_validator.py` | Post-cleaning validator | 52 |
| `test_profiler.py` | DataProfile builder | ~40 |
| `test_session.py` | Session state machine | 56 |
| `test_ingestion.py` | File ingestion | ~30 |
| `test_domain_detector.py` | Domain inference | 17 |
| `test_standardizer.py` | Column normalisation | ~25 |

---

## Configuration

All constants live in [app/config.py](app/config.py). Notable settings:

| Constant | Default | Purpose |
|----------|---------|---------|
| `MODEL_SONNET` | `claude-sonnet-4-20250514` | Planning / KPI calls |
| `MODEL_HAIKU` | `claude-haiku-4-5-20251001` | Code-fix retries |
| `EXECUTOR_MAX_RETRIES` | `3` | LLM fix attempts per failed op |
| `EXECUTOR_TIMEOUT_SECONDS` | `30` | Sandbox execution timeout |
| `LLM_MAX_RETRIES` | `3` | API retry attempts on errors |
| `ROW_DROP_CRITICAL_PCT` | `0.50` | Block if >50% rows dropped |
| `ROW_DROP_WARNING_PCT` | `0.20` | Warn if >20% rows dropped |

---

## Supported file formats

| Format | Extensions |
|--------|-----------|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| JSON (records or columns orientation) | `.json` |
| Parquet | `.parquet` |

Maximum file size: 500 MB (warning at 100 MB).

---

## Logs

Structured JSON logs are written to `data/logs/agent.log` (rotating, 10 MB × 3 backups). Every LLM call, sandboxed execution, and validation check is logged with timestamps and session context.

---

## Requirements

- Python 3.11+
- Anthropic API key (`ANTHROPIC_API_KEY` in `.env`)
- See `requirements.txt` for the full dependency list
