# PROJECT GOAL

Build a local portfolio application demonstrating:

1. Finance reconciliation using deterministic Python logic.
2. Natural-language-to-SQL using an LLM.
3. Schema-aware and read-only SQL validation before execution.
4. RAG over a small set of Finance policy/reconciliation documents.
5. A LangGraph workflow coordinating planning, retrieval, SQL generation, reconciliation, discrepancy analysis, human approval, and report generation.
6. FastAPI endpoints exposing the workflow.
7. A Streamlit UI for asking Finance questions, reviewing evidence/results, approving or rejecting AI output, and viewing telemetry.
8. Telemetry for workflow usage, token usage, estimated inference cost, latency, validation failures, human acceptance, completion rate, and estimated time savings.
9. Docker support.
10. Automated tests for important deterministic components.

# STRICT SCOPE RULES

Do NOT add features unless explicitly requested in a future prompt.

Specifically do NOT add:

- authentication
- user accounts
- OAuth
- Kubernetes
- Terraform
- AWS deployment
- Azure deployment
- GCP deployment
- Redis
- Celery
- Kafka
- message queues
- React
- Next.js
- separate frontend frameworks
- PostgreSQL
- MySQL
- external vector databases
- OCR
- payment processing
- real banking integrations
- real financial data
- CI/CD
- GitHub Actions
- monitoring platforms
- extra agents
- extra APIs
- speculative features

Use SQLite for the demonstration database.

Use synthetic financial data only.

Use Python 3.11+.

Use Anthropic's Python API client for the LLM integration.

Use LangGraph for orchestration.

Use FastAPI for the API.

Use Streamlit for the UI.

Use a lightweight local retrieval implementation for RAG. Do not introduce an external vector database.

For SQL safety use sqlglot or another proper SQL parser rather than relying only on regex.

All financial reconciliation calculations must be deterministic Python/SQL logic. The LLM must NOT calculate authoritative financial totals itself.

The LLM may:

- understand questions
- create workflow plans
- generate candidate SQL
- explain discrepancies
- generate reports

The LLM may NOT bypass deterministic validation.

Do not fabricate performance numbers or ROI claims.

When telemetry contains estimated time savings, clearly label it as an estimate.

# CODE CHANGE RULES

For every future prompt:

1. Modify only the files required by that prompt.
2. Do not refactor unrelated files.
3. Do not create extra features.
4. Do not change architecture unless explicitly requested.
5. Do not silently add dependencies.
6. If a new dependency is necessary, explain why in the completion summary.
7. Never commit API keys or secrets.
8. Keep `.env` files out of Git.
9. Do not use real financial/customer data.
10. Stop when the requested task is complete.

After completing each future task, report:

- files created
- files modified
- tests/commands run
- whether they passed
- any manual action required from me

Do not continue to the next project phase unless explicitly requested.
