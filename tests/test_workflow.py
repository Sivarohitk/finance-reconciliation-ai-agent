"""Tests for the LangGraph Finance reconciliation workflow."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from langgraph.types import Command

from app import llm, telemetry
from app.workflow import build_workflow, create_workflow_input


def llm_result(data: dict, input_tokens: int, output_tokens: int) -> dict:
    return {
        "data": data,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "model": "mock-model",
    }


@pytest.fixture
def mocked_llm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setattr(telemetry, "record_workflow_telemetry", Mock())

    functions = {
        "plan": Mock(
            return_value=llm_result(
                {"steps": [{"name": "reconcile"}], "assumptions": []},
                10,
                4,
            )
        ),
        "sql": Mock(
            return_value=llm_result(
                {
                    "sql": (
                        "SELECT reference, amount FROM ledger_transactions "
                        "ORDER BY reference"
                    )
                },
                12,
                5,
            )
        ),
        "analysis": Mock(
            return_value=llm_result(
                {
                    "summary": "Deterministic discrepancies require review.",
                    "evidence_citations": ["reconciliation_policy.md"],
                    "unsupported_conclusions": [],
                    "assumptions": [],
                },
                20,
                8,
            )
        ),
        "report": Mock(
            return_value=llm_result(
                {
                    "title": "Reconciliation report",
                    "deterministic_results": {"preserved": True},
                    "approval_status": "approved",
                },
                25,
                10,
            )
        ),
    }
    monkeypatch.setattr(
        llm, "create_reconciliation_workflow_plan", functions["plan"]
    )
    monkeypatch.setattr(llm, "generate_candidate_sql", functions["sql"])
    monkeypatch.setattr(
        llm, "generate_discrepancy_explanation", functions["analysis"]
    )
    monkeypatch.setattr(
        llm, "generate_final_reconciliation_report", functions["report"]
    )
    return functions


def test_approved_workflow_interrupts_then_generates_report(
    mocked_llm: dict[str, Mock],
) -> None:
    graph = build_workflow()
    config = {"configurable": {"thread_id": "approved-workflow"}}

    paused = graph.invoke(
        create_workflow_input(
            "Show ledger transactions and reconcile discrepancies.",
            workflow_id="workflow-approved",
        ),
        config,
    )

    assert paused["workflow_id"] == "workflow-approved"
    assert paused["sql_validation"]["is_valid"] is True
    assert paused["sql_results"]
    assert paused["reconciliation_results"]["summary"]
    assert paused["human_decision"] is None
    assert paused["token_usage"] == {
        "input_tokens": 42,
        "output_tokens": 17,
    }
    assert paused["__interrupt__"]
    mocked_llm["report"].assert_not_called()

    completed = graph.invoke(Command(resume="approve"), config)

    assert completed["human_decision"] == "approve"
    assert completed["final_report"]["approval_status"] == "approved"
    assert completed["final_report"]["deterministic_results"] == completed[
        "reconciliation_results"
    ]
    assert completed["token_usage"] == {
        "input_tokens": 67,
        "output_tokens": 27,
    }
    mocked_llm["report"].assert_called_once()


def test_invalid_sql_is_not_executed_and_rejection_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    mocked_llm: dict[str, Mock],
) -> None:
    mocked_llm["sql"].return_value = llm_result(
        {"sql": "DELETE FROM ledger_transactions"}, 12, 5
    )
    execute_spy = Mock(side_effect=AssertionError("invalid SQL was executed"))
    monkeypatch.setattr(
        "app.workflow.sql_guard.execute_validated_sql", execute_spy
    )
    graph = build_workflow()
    config = {"configurable": {"thread_id": "rejected-workflow"}}

    paused = graph.invoke(
        create_workflow_input(
            "Delete ledger rows.", workflow_id="workflow-rejected"
        ),
        config,
    )

    assert paused["sql_validation"]["is_valid"] is False
    assert paused["sql_results"] == []
    assert paused["validation_failures"] == 1
    assert paused["reconciliation_results"]["summary"]
    execute_spy.assert_not_called()

    rejected = graph.invoke(Command(resume="reject"), config)

    assert rejected["human_decision"] == "reject"
    assert rejected.get("final_report") is None
    mocked_llm["report"].assert_not_called()


def test_insufficient_evidence_lowers_confidence_and_flags_review(
    mocked_llm: dict[str, Mock],
) -> None:
    retriever = Mock()
    retriever.retrieve.return_value = []
    graph = build_workflow(retriever=retriever)
    config = {"configurable": {"thread_id": "low-confidence"}}

    paused = graph.invoke(
        create_workflow_input("Reconcile Finance activity."), config
    )

    assert paused["confidence_score"] == 0.0
    assert "Insufficient policy evidence for grounded conclusions." in paused[
        "errors"
    ]
    assert paused["__interrupt__"]


def test_human_review_rejects_unsupported_decisions(
    mocked_llm: dict[str, Mock],
) -> None:
    graph = build_workflow()
    config = {"configurable": {"thread_id": "invalid-decision"}}
    graph.invoke(create_workflow_input("Reconcile transactions."), config)

    with pytest.raises(ValueError, match="approve or reject"):
        graph.invoke(Command(resume="revise"), config)
