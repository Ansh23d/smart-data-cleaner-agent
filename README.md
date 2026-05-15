# Smart Data Cleaner Agent

A web app that takes a messy dataset, cleans it with your approval at every step, and then generates business KPIs and a dashboard automatically. You stay in control throughout — the AI proposes, you decide.

---

## What it does

Upload a CSV, Excel, JSON, or Parquet file and the app walks you through a six-step pipeline:

1. **Upload** — drop your file and get an instant summary of rows, columns, and file size
2. **Profile** — the app scans every column for null rates, data types, quality issues, and flags anything suspicious
3. **Cleaning plan** — the AI proposes a set of cleaning operations (rename columns, fix types, fill nulls, remove duplicates, etc.). You tick the ones you want and skip the rest
4. **Results** — the selected operations run and you see a before/after comparison showing exactly what changed
5. **KPIs** — the app detects what kind of data you have and generates business KPIs
6. **Dashboard** — a clean visual summary with metric scorecards and auto-generated charts, plus the option to download your cleaned data

---

## Setup

You need Python 3.11 or higher and an Anthropic API key.

```bash
git clone https://github.com/Ansh23d/smart-data-cleaner-agent.git
cd smart-data-cleaner-agent
pip install -r requirements.txt
```

Create a `.env` file in the root folder with your API key:

```
ANTHROPIC_API_KEY=your-key-here
```

Then run the app:

```bash
streamlit run app.py
```

---

## Supported file formats

CSV, Excel (.xlsx and .xls), JSON, and Parquet. Maximum file size is 500 MB.

---

## How the AI is used

The app makes LLM calls in exactly two places — everything else is deterministic Python:

- **Cleaning plan generation** — one call to Claude Sonnet, which reads a summary of your data profile and proposes cleaning steps as structured JSON
- **KPI generation** — one call to Claude Sonnet, which reads the column names, types, and domain context and returns four KPIs with pandas code

If a generated code snippet fails to execute, Claude Haiku automatically attempts a fix (up to two retries). All other steps: profiling, execution, validation, chart generation — run locally with no LLM involvement.

---

## Project structure

```
app.py                      Main application (single-file Streamlit app)
config.py                   All constants and settings
agents/
  cleaning_agent.py         Generates and fixes cleaning operations
  kpi_agent.py              Generates and fixes KPIs
  prompts/                  Prompt templates for each agent
models/
  cleaning_plan.py          Data model for the cleaning plan
  kpi.py                    Data model for KPIs
  profile.py                Data model for the data profile
pipeline/
  ingestion.py              Reads files into a DataFrame
  profiler.py               Analyses columns and builds the data profile
  domain_detector.py        Guesses the business domain from column names
  executor.py               Runs LLM-generated code in a sandboxed environment
  validator.py              Checks the cleaned data for unexpected changes
services/
  llm_client.py             Anthropic API wrapper with retry and cost tracking
utils/
  cost_tracker.py           Tracks token usage and cost per session
  logger.py                 Structured logging to file
```

---

## Requirements

- Python 3.11+
- Anthropic API key
- All dependencies are listed in `requirements.txt`
