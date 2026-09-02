"""Narrow Anthropic integrations for Finance workflow language tasks."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()


class LLMConfigurationError(RuntimeError):
    """Raised when required Anthropic configuration is unavailable."""


class LLMResponseError(RuntimeError):
    """Raised when the model does not return the required structured data."""


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class LLMResult(TypedDict):
    data: dict[str, Any]
    usage: TokenUsage
    model: str


PLAN_SYSTEM_PROMPT = """
Create a concise reconciliation workflow plan for the supplied Finance question.
Plan only the requested workflow activities; do not calculate financial totals.
Deterministic Python and SQL components will produce all authoritative numbers.
Return only a JSON object with this shape:
{"steps": [{"name": "...", "purpose": "...", "requires_human_approval": true}],
 "assumptions": ["..."]}
""".strip()

SQL_SYSTEM_PROMPT = """
Convert the Finance question into one candidate SQLite query.
- only generate SELECT statements
- use only the supplied schema
- never invent tables or columns
- SQL will be independently validated before execution
Do not calculate or state financial results yourself.
Return only a JSON object with this shape: {"sql": "SELECT ..."}
""".strip()

GROUNDED_RESULTS_RULES = """
- deterministic reconciliation results are authoritative
- do not modify numerical results
- cite or surface the supplied evidence
- identify unsupported conclusions
- identify assumptions
- never fabricate evidence
""".strip()

DISCREPANCY_SYSTEM_PROMPT = f"""
Explain the supplied deterministic reconciliation discrepancies for a Finance reviewer.
Follow these rules:
{GROUNDED_RESULTS_RULES}
Return only a JSON object with this shape:
{{"summary": "...", "evidence_citations": ["..."],
  "unsupported_conclusions": ["..."], "assumptions": ["..."]}}
""".strip()

REPORT_SYSTEM_PROMPT = f"""
Create a structured final reconciliation report from the supplied materials.
Follow these rules:
{GROUNDED_RESULTS_RULES}
Reflect the supplied human approval decision and never present unapproved output as final.
Return only a JSON object with this shape:
{{"title": "...", "executive_summary": "...",
  "deterministic_results": {{}}, "discrepancies": [],
  "evidence_citations": ["..."], "assumptions": [],
  "unsupported_conclusions": [], "approval_status": "..."}}
""".strip()


def _anthropic_settings() -> tuple[str, str]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "").strip()
    if not api_key:
        raise LLMConfigurationError("ANTHROPIC_API_KEY is required.")
    if not model:
        raise LLMConfigurationError("ANTHROPIC_MODEL is required.")
    return api_key, model


def _response_text(content_blocks: Sequence[Any]) -> str:
    text_parts = [
        block.text
        for block in content_blocks
        if getattr(block, "type", None) == "text"
        and isinstance(getattr(block, "text", None), str)
    ]
    if not text_parts:
        raise LLMResponseError("Anthropic returned no text content.")
    return "".join(text_parts).strip()


def _usage_value(usage: Any, field: str) -> int:
    if isinstance(usage, Mapping):
        value = usage.get(field, 0)
    else:
        value = getattr(usage, field, 0)
    return int(value or 0)


def _call_json(
    system_prompt: str,
    payload: Mapping[str, Any],
    *,
    max_tokens: int,
) -> LLMResult:
    api_key, model = _anthropic_settings()
    response = Anthropic(api_key=api_key).messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            }
        ],
    )

    response_text = _response_text(response.content)
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("Anthropic response was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("Anthropic response JSON must be an object.")

    return {
        "data": data,
        "usage": {
            "input_tokens": _usage_value(response.usage, "input_tokens"),
            "output_tokens": _usage_value(response.usage, "output_tokens"),
        },
        "model": model,
    }


def create_reconciliation_workflow_plan(
    question: str,
    *,
    policy_evidence: Sequence[Mapping[str, Any]] = (),
) -> LLMResult:
    """Create a structured plan without performing financial calculations."""
    return _call_json(
        PLAN_SYSTEM_PROMPT,
        {"finance_question": question, "policy_evidence": list(policy_evidence)},
        max_tokens=1_200,
    )


def generate_candidate_sql(
    question: str,
    schema: Mapping[str, Any],
) -> LLMResult:
    """Generate candidate SELECT SQL for independent deterministic validation."""
    return _call_json(
        SQL_SYSTEM_PROMPT,
        {"finance_question": question, "database_schema": schema},
        max_tokens=600,
    )


def generate_discrepancy_explanation(
    reconciliation_results: Mapping[str, Any],
    policy_evidence: Sequence[Mapping[str, Any]],
) -> LLMResult:
    """Explain discrepancies without altering deterministic results."""
    return _call_json(
        DISCREPANCY_SYSTEM_PROMPT,
        {
            "deterministic_reconciliation_results": reconciliation_results,
            "policy_evidence": list(policy_evidence),
        },
        max_tokens=1_400,
    )


def generate_final_reconciliation_report(
    reconciliation_results: Mapping[str, Any],
    policy_evidence: Sequence[Mapping[str, Any]],
    discrepancy_explanation: Mapping[str, Any],
    *,
    human_decision: str,
) -> LLMResult:
    """Generate a grounded structured report reflecting human approval."""
    return _call_json(
        REPORT_SYSTEM_PROMPT,
        {
            "deterministic_reconciliation_results": reconciliation_results,
            "policy_evidence": list(policy_evidence),
            "discrepancy_explanation": discrepancy_explanation,
            "human_decision": human_decision,
        },
        max_tokens=2_000,
    )
