"""Tests for the minimal FastAPI layer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import llm, telemetry
from app.api import create_app
from data.seed_finance import create_database


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
                {"sql": "SELECT reference FROM ledger_transactions"},
                12,
                5,
            )
        ),
        "analysis": Mock(
            return_value=llm_result(
                {
                    "summary": "Review deterministic discrepancies.",
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


@pytest.fixture
def client(
    tmp_path: Path,
    mocked_llm: dict[str, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    database_path = create_database(tmp_path / "finance.db")
    monkeypatch.setattr(
        telemetry.config, "INPUT_COST_PER_MILLION_TOKENS", 1.0
    )
    monkeypatch.setattr(
        telemetry.config, "OUTPUT_COST_PER_MILLION_TOKENS", 2.0
    )
    with TestClient(create_app(database_path=database_path)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "finance-reconciliation-ai-agent",
    }


def test_start_workflow_returns_review_payload(client: TestClient) -> None:
    response = client.post(
        "/workflow/start",
        json={"finance_question": "Reconcile ledger and bank activity."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["workflow_id"]
    assert body["status"] == "awaiting_human_review"
    assert body["workflow_plan"]["steps"]
    assert body["candidate_sql"].startswith("SELECT")
    assert body["sql_validation"]["is_valid"] is True
    assert body["reconciliation_summary"]["counts"]
    assert body["deterministic_reconciliation_results"]["summary"]
    assert body["deterministic_reconciliation_results"]["records"]
    assert body["retrieved_evidence"]
    assert body["discrepancy_analysis"]["summary"]
    assert body["confidence_score"] == 1.0
    assert body["grounding_sufficient"] is True
    assert body["validation_warnings"] == []
    assert body["human_approval_required"] is True


def test_approve_resumes_workflow_and_returns_final_report(
    client: TestClient,
) -> None:
    started = client.post(
        "/workflow/start", json={"finance_question": "Reconcile activity."}
    ).json()

    response = client.post(
        f"/workflow/{started['workflow_id']}/decision",
        json={"decision": "approve"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["workflow_id"] == started["workflow_id"]
    assert body["status"] == "approved"
    assert body["human_decision"] == "approve"
    assert body["final_report"]["approval_status"] == "approved"
    assert body["final_report"]["deterministic_results"]["summary"]


def test_reject_does_not_generate_approved_report(
    client: TestClient,
    mocked_llm: dict[str, Mock],
) -> None:
    started = client.post(
        "/workflow/start", json={"finance_question": "Reconcile activity."}
    ).json()

    response = client.post(
        f"/workflow/{started['workflow_id']}/decision",
        json={"decision": "reject"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "workflow_id": started["workflow_id"],
        "status": "rejected",
        "human_decision": "reject",
        "final_report": None,
    }
    mocked_llm["report"].assert_not_called()


def test_unknown_or_completed_workflow_cannot_be_resumed(
    client: TestClient,
) -> None:
    missing = client.post(
        "/workflow/missing/decision", json={"decision": "approve"}
    )
    assert missing.status_code == 404

    started = client.post(
        "/workflow/start", json={"finance_question": "Reconcile activity."}
    ).json()
    decision_url = f"/workflow/{started['workflow_id']}/decision"
    assert client.post(decision_url, json={"decision": "reject"}).status_code == 200

    repeated = client.post(decision_url, json={"decision": "approve"})
    assert repeated.status_code == 409


def test_request_validation_rejects_bad_inputs(client: TestClient) -> None:
    assert client.post(
        "/workflow/start", json={"finance_question": "   "}
    ).status_code == 422
    assert client.post(
        "/workflow/anything/decision", json={"decision": "revise"}
    ).status_code == 422


def test_telemetry_summary_returns_local_aggregate_metrics(
    client: TestClient,
) -> None:
    started = client.post(
        "/workflow/start", json={"finance_question": "Reconcile activity."}
    ).json()
    client.post(
        f"/workflow/{started['workflow_id']}/decision",
        json={"decision": "approve"},
    )

    response = client.get("/telemetry/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["total_workflows"] == 1
    assert body["completed_workflows"] == 1
    assert body["completion_rate"] == 1.0
    assert body["approval_rate"] == 1.0
    assert body["total_input_tokens"] == 67
    assert body["total_output_tokens"] == 27
    assert body["estimated_total_minutes_saved"] == 15.0


def test_no_arbitrary_sql_endpoint_is_exposed(client: TestClient) -> None:
    assert client.post("/sql", json={"sql": "SELECT 1"}).status_code == 404
