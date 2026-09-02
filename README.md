# Finance Reconciliation & Reporting AI Agent

## Project Overview

Finance Reconciliation & Reporting AI Agent is a local portfolio application that demonstrates how an LLM-assisted Finance workflow can operate within deterministic and reviewable controls. It compares synthetic ledger and bank transactions, identifies reconciliation discrepancies, answers Finance questions through validated analytical SQL, retrieves relevant local policy evidence, and pauses for human approval before producing a final report.

Authoritative financial totals and reconciliation classifications are calculated by Python and SQLite logic. Anthropic is used only for planning, candidate SQL generation, discrepancy commentary, and report generation; model output cannot bypass SQL validation or replace deterministic results.

This repository is a portfolio demonstration. It does not use real financial, customer, or banking data and is not a production financial system.

## Architecture

```text
Streamlit UI
    |
    v
FastAPI API
    |
    v
LangGraph workflow
    |-- Anthropic client: planning, candidate SQL, explanations, reports
    |-- Local BM25 retriever: Finance policy evidence
    |-- sqlglot guard: SQL parsing, allowlist, and schema validation
    |-- SQLite database: synthetic ledger, bank, and local telemetry data
    |-- Deterministic reconciliation engine: totals and discrepancy records
    `-- Human review interrupt: approve or reject
```

The Streamlit interface communicates with FastAPI rather than duplicating backend business logic. FastAPI owns an in-memory, resumable LangGraph workflow and exposes only workflow, decision, health, and telemetry-summary endpoints. SQLite stores the synthetic transaction dataset and local telemetry.

The API provides:

- `GET /health`
- `POST /workflow/start`
- `POST /workflow/{workflow_id}/decision`
- `GET /telemetry/summary`

There is no arbitrary SQL execution endpoint.

## Workflow

```text
Finance Question
→ Planning
→ RAG
→ NL-to-SQL
→ SQL Validation
→ SQL Execution
→ Deterministic Reconciliation
→ Discrepancy Analysis
→ Human Review
→ Report Generation
→ Telemetry
```

1. The user submits a Finance reconciliation question.
2. Anthropic creates a workflow plan.
3. The local BM25 retriever selects evidence from the reconciliation, reporting, and SQL-safety policy documents.
4. Anthropic generates candidate SQL using the supplied SQLite schema.
5. `sqlglot` parses and validates the query before execution.
6. Validated SQL runs through a read-only SQLite connection.
7. Deterministic logic reconciles ledger and bank transactions by reference and detects matched transactions, missing transactions, amount mismatches, duplicate references, currency mismatches, and status mismatches. Amount comparisons use a `$0.01` tolerance.
8. Anthropic generates discrepancy commentary grounded in the deterministic results and retrieved policy evidence.
9. LangGraph pauses using its interrupt/resume mechanism for an `approve` or `reject` decision.
10. Approval produces a structured final report. Rejection preserves the decision and does not produce an approved report.
11. Local telemetry records workflow usage and configured estimates.

## Safety Design

- **Read-only SQL:** Candidate queries execute through a read-only SQLite connection only after validation.
- **Schema validation:** `sqlglot` parses SQL and checks referenced tables and columns against the actual SQLite schema. Queries are limited to `ledger_transactions` and `bank_transactions`; write operations, multiple statements, system objects, and unapproved tables are rejected.
- **Deterministic calculations:** Financial totals, discrepancy amounts, classifications, and supporting records come from deterministic Python and SQL operations. The LLM does not calculate authoritative financial results.
- **Grounded evidence:** Local BM25 retrieval supplies policy chunks and document names to the explanation and reporting stages. Prompts require assumptions and unsupported conclusions to be surfaced rather than fabricated.
- **Human approval:** The workflow pauses before approval-dependent report generation. Users can approve or reject the result, and rejection does not generate an approved report.
- **Synthetic data only:** The bundled database is reproducibly generated from a fixed random seed and synthetic merchant, reference, and transaction data.

## Technology

- Python 3.11+
- SQL and SQLite
- Anthropic Python API client
- LangGraph
- Local RAG with BM25 (`rank-bm25`)
- `sqlglot`
- FastAPI and Uvicorn
- Streamlit
- Pydantic
- Docker and Docker Compose
- Pytest

## Local Setup

### Prerequisites

- Python 3.11 or newer
- An Anthropic API key and supported Anthropic model name for live LLM workflow execution
- Docker Desktop or another Docker Engine only if using Docker

Create and activate a virtual environment, then install the existing dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, activate and copy the environment template with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your local configuration. Never commit this file or an API key.

The repository includes `data/finance.db`. To recreate it deterministically from the synthetic seed script:

```bash
python data/seed_finance.py
```

Running the seed script replaces the existing demonstration tables and telemetry table in that database.

## Environment Variables

| Variable | Purpose | Required |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Anthropic credential used by the official Python client | Required for live LLM workflow execution |
| `ANTHROPIC_MODEL` | Anthropic model name used for planning, SQL generation, explanations, and reports | Required for live LLM workflow execution |
| `INPUT_COST_PER_MILLION_TOKENS` | User-supplied input-token rate used for estimated inference cost | Optional |
| `OUTPUT_COST_PER_MILLION_TOKENS` | User-supplied output-token rate used for estimated inference cost | Optional |
| `MANUAL_TASK_BASELINE_MINUTES` | Configurable baseline used for estimated time savings | Optional; defaults to `15` |

Vendor pricing is not hardcoded. If cost rates are configured, reported inference cost remains an estimate. Estimated time savings are based on the configured manual-task baseline and are also estimates, not measured ROI.

## Running without Docker

Start FastAPI from the repository root:

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal with the same environment, start Streamlit:

```bash
streamlit run app/ui.py --server.port 8501
```

Open:

- Streamlit UI: `http://localhost:8501`
- FastAPI service: `http://localhost:8000`

## Running with Docker

Create `.env` from `.env.example`, provide the required live Anthropic settings, and run:

```bash
docker compose up --build
```

Docker Compose starts exactly two services from the same project image:

- FastAPI on `http://localhost:8000`
- Streamlit on `http://localhost:8501`

The local `./data` directory is mounted into both containers so the SQLite database persists outside the containers. Environment variables are read at runtime from `.env`; secrets are not embedded in the image.

Stop the services with:

```bash
docker compose down
```

## Testing

Run the complete automated test suite from the repository root:

```bash
python -m pytest -ra
```

The tests cover the database layer, deterministic reconciliation categories, SQL safety, local RAG, mocked Anthropic integration, LangGraph approval and rejection routing, telemetry, and FastAPI behavior. Anthropic calls are mocked in automated tests, so running the suite does not require an API key or consume paid requests.

## Example Finance Questions

- `Show me transactions with reconciliation discrepancies and summarize the major causes.`
- `Which ledger transactions are missing from the bank transactions?`
- `Find amount mismatches and explain the applicable reconciliation policy.`
- `Summarize currency and status mismatches with supporting evidence.`
- `Which bank transactions are missing from the ledger?`

Generated SQL remains subject to the same table, column, statement-type, and read-only validation regardless of the question.

## Project Limitations

- The dataset is entirely synthetic and intentionally includes deterministic discrepancy examples.
- This is a local portfolio demonstration, not a production financial system.
- It has no real banking integrations, authentication, user accounts, or production deployment configuration.
- Workflow checkpoint state is held in the running application process; restarting the API loses pending human-review workflows.
- RAG uses a small local policy-document collection and BM25 retrieval, not an external vector database.
- Telemetry is stored locally in SQLite and does not represent real production users or workloads.
- Inference-cost figures are configuration-based estimates and depend on user-provided token rates.
- Time-savings figures are estimates based on a configurable manual-task baseline; they are not measured savings or ROI claims.
- Automated tests validate LLM integration with mocked responses. Live model authentication, response quality, token metadata, and latency require a configured Anthropic account and API key.
- Human review and deterministic controls reduce risk but do not make the application suitable for financial reporting, audit, compliance, or operational decision-making.
