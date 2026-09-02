import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import llm


def anthropic_response(payload, *, input_tokens=23, output_tokens=11):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


@pytest.fixture()
def mocked_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-model")
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr(llm, "Anthropic", constructor)
    return constructor, client


def test_requires_api_key_from_environment(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-model")

    with pytest.raises(llm.LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        llm.create_reconciliation_workflow_plan("Reconcile the ledger")


def test_requires_model_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    with pytest.raises(llm.LLMConfigurationError, match="ANTHROPIC_MODEL"):
        llm.create_reconciliation_workflow_plan("Reconcile the ledger")


def test_creates_structured_workflow_plan_and_captures_usage(mocked_anthropic):
    constructor, client = mocked_anthropic
    client.messages.create.return_value = anthropic_response(
        {
            "steps": [
                {
                    "name": "retrieve_policy",
                    "purpose": "Find matching rules",
                    "requires_human_approval": False,
                }
            ],
            "assumptions": [],
        },
        input_tokens=31,
        output_tokens=17,
    )

    result = llm.create_reconciliation_workflow_plan(
        "Find unreconciled transactions",
        policy_evidence=[{"document_name": "reconciliation_policy.md"}],
    )

    constructor.assert_called_once_with(api_key="test-api-key")
    assert result["model"] == "test-model"
    assert result["data"]["steps"][0]["name"] == "retrieve_policy"
    assert result["usage"] == {"input_tokens": 31, "output_tokens": 17}


def test_sql_prompt_contains_all_safety_requirements(mocked_anthropic):
    _, client = mocked_anthropic
    client.messages.create.return_value = anthropic_response(
        {"sql": "SELECT reference FROM ledger_transactions"}
    )

    result = llm.generate_candidate_sql(
        "Show ledger references",
        {"ledger_transactions": ["id", "reference"]},
    )

    call = client.messages.create.call_args.kwargs
    prompt = call["system"]
    assert "only generate SELECT statements" in prompt
    assert "use only the supplied schema" in prompt
    assert "never invent tables or columns" in prompt
    assert "independently validated before execution" in prompt
    assert call["model"] == "test-model"
    assert result["data"]["sql"].startswith("SELECT")


def test_discrepancy_prompt_preserves_authoritative_results_and_evidence(
    mocked_anthropic,
):
    _, client = mocked_anthropic
    client.messages.create.return_value = anthropic_response(
        {
            "summary": "One mismatch requires review.",
            "evidence_citations": ["reconciliation_policy.md"],
            "unsupported_conclusions": [],
            "assumptions": [],
        }
    )
    deterministic_results = {"summary": {"amount_mismatches": 1}}

    result = llm.generate_discrepancy_explanation(
        deterministic_results,
        [{"document_name": "reconciliation_policy.md", "chunk_text": "Evidence"}],
    )

    call = client.messages.create.call_args.kwargs
    prompt = call["system"]
    assert "deterministic reconciliation results are authoritative" in prompt
    assert "do not modify numerical results" in prompt
    assert "cite or surface the supplied evidence" in prompt
    assert "identify unsupported conclusions" in prompt
    assert "identify assumptions" in prompt
    assert "never fabricate evidence" in prompt
    assert result["data"]["evidence_citations"] == [
        "reconciliation_policy.md"
    ]


def test_final_report_prompt_is_grounded_and_structured(mocked_anthropic):
    _, client = mocked_anthropic
    client.messages.create.return_value = anthropic_response(
        {
            "title": "Synthetic reconciliation report",
            "executive_summary": "Review complete.",
            "deterministic_results": {"total_ledger_amount": 100.0},
            "discrepancies": [],
            "evidence_citations": ["reporting_policy.md"],
            "assumptions": [],
            "unsupported_conclusions": [],
            "approval_status": "approved",
        }
    )

    result = llm.generate_final_reconciliation_report(
        {"total_ledger_amount": 100.0},
        [{"document_name": "reporting_policy.md", "chunk_text": "Evidence"}],
        {"summary": "No unresolved discrepancies."},
        human_decision="approved",
    )

    prompt = client.messages.create.call_args.kwargs["system"]
    assert "deterministic reconciliation results are authoritative" in prompt
    assert "do not modify numerical results" in prompt
    assert "cite or surface the supplied evidence" in prompt
    assert "identify unsupported conclusions" in prompt
    assert "identify assumptions" in prompt
    assert "never fabricate evidence" in prompt
    assert result["data"]["approval_status"] == "approved"


def test_rejects_non_json_model_output(mocked_anthropic):
    _, client = mocked_anthropic
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="not json")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )

    with pytest.raises(llm.LLMResponseError, match="valid JSON"):
        llm.create_reconciliation_workflow_plan("Reconcile transactions")
