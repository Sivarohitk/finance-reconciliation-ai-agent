"""Tests for local application telemetry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from langgraph.types import Command

from app import llm, telemetry
from app.database import execute_select
from app.workflow import build_workflow, create_workflow_input
from data.seed_finance import create_database


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return create_database(tmp_path / "finance.db")


def test_estimated_cost_uses_configured_token_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        telemetry.config, "INPUT_COST_PER_MILLION_TOKENS", 3.0
    )
    monkeypatch.setattr(
        telemetry.config, "OUTPUT_COST_PER_MILLION_TOKENS", 15.0
    )

    cost = telemetry.calculate_estimated_inference_cost(2_000_000, 500_000)

    assert cost == pytest.approx(13.5)


def test_missing_cost_configuration_returns_no_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        telemetry.config, "INPUT_COST_PER_MILLION_TOKENS", None
    )
    monkeypatch.setattr(
        telemetry.config, "OUTPUT_COST_PER_MILLION_TOKENS", None
    )

    assert telemetry.calculate_estimated_inference_cost(100, 50) is None


def test_record_workflow_telemetry_uses_local_configuration(
    monkeypatch: pytest.MonkeyPatch,
    database_path: Path,
) -> None:
    monkeypatch.setattr(
        telemetry.config, "INPUT_COST_PER_MILLION_TOKENS", 2.0
    )
    monkeypatch.setattr(
        telemetry.config, "OUTPUT_COST_PER_MILLION_TOKENS", 8.0
    )
    monkeypatch.setattr(
        telemetry.config, "MANUAL_TASK_BASELINE_MINUTES", 15
    )

    record = telemetry.record_workflow_telemetry(
        workflow_id="workflow-1",
        latency_ms=1_250.0,
        input_tokens=1_000,
        output_tokens=500,
        validation_failures=2,
        human_decision="approve",
        completed=True,
        database_path=database_path,
        timestamp="2026-09-01T12:00:00+00:00",
    )

    rows = execute_select(
        "SELECT * FROM telemetry WHERE workflow_id = ?",
        ("workflow-1",),
        database_path=database_path,
    )
    assert len(rows) == 1
    assert rows[0]["created_at"] == record["timestamp"]
    assert rows[0]["estimated_cost_usd"] == pytest.approx(0.006)
    assert rows[0]["estimated_minutes_saved"] == pytest.approx(15.0)
    assert rows[0]["completed"] == 1


def test_record_workflow_telemetry_closes_its_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = MagicMock()
    connection.execute.return_value.rowcount = 1
    monkeypatch.setattr(telemetry, "get_connection", Mock(return_value=connection))

    telemetry.record_workflow_telemetry(
        workflow_id="workflow-close",
        latency_ms=1.0,
        input_tokens=0,
        output_tokens=0,
        validation_failures=0,
        human_decision=None,
        completed=False,
    )

    connection.close.assert_called_once_with()


def test_aggregate_metrics_are_derived_from_recorded_workflows(
    monkeypatch: pytest.MonkeyPatch,
    database_path: Path,
) -> None:
    monkeypatch.setattr(
        telemetry.config, "INPUT_COST_PER_MILLION_TOKENS", 1.0
    )
    monkeypatch.setattr(
        telemetry.config, "OUTPUT_COST_PER_MILLION_TOKENS", 2.0
    )
    monkeypatch.setattr(
        telemetry.config, "MANUAL_TASK_BASELINE_MINUTES", 15
    )
    records = [
        {
            "workflow_id": "one",
            "latency_ms": 100.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "validation_failures": 1,
            "human_decision": "approve",
            "completed": True,
        },
        {
            "workflow_id": "two",
            "latency_ms": 200.0,
            "input_tokens": 200,
            "output_tokens": 100,
            "validation_failures": 2,
            "human_decision": "reject",
            "completed": True,
        },
        {
            "workflow_id": "three",
            "latency_ms": 300.0,
            "input_tokens": 300,
            "output_tokens": 150,
            "validation_failures": 0,
            "human_decision": None,
            "completed": False,
        },
    ]
    for record in records:
        telemetry.record_workflow_telemetry(
            **record,
            database_path=database_path,
        )

    metrics = telemetry.get_aggregate_metrics(database_path)

    assert metrics == pytest.approx(
        {
            "total_workflows": 3,
            "completed_workflows": 2,
            "completion_rate": 2 / 3,
            "approval_rate": 0.5,
            "average_latency_ms": 200.0,
            "total_input_tokens": 600,
            "total_output_tokens": 300,
            "estimated_total_inference_cost_usd": 0.0012,
            "sql_validation_failure_count": 3,
            "estimated_total_minutes_saved": 30.0,
        }
    )


def test_workflow_records_terminal_telemetry_without_live_llm(
    monkeypatch: pytest.MonkeyPatch,
    database_path: Path,
) -> None:
    monkeypatch.setattr(
        telemetry.config, "INPUT_COST_PER_MILLION_TOKENS", 1.0
    )
    monkeypatch.setattr(
        telemetry.config, "OUTPUT_COST_PER_MILLION_TOKENS", 2.0
    )
    monkeypatch.setattr(
        telemetry.config, "MANUAL_TASK_BASELINE_MINUTES", 15
    )
    monkeypatch.setattr(
        llm,
        "create_reconciliation_workflow_plan",
        Mock(return_value=_llm_result({"steps": []}, 10, 2)),
    )
    monkeypatch.setattr(
        llm,
        "generate_candidate_sql",
        Mock(
            return_value=_llm_result(
                {"sql": "SELECT reference FROM ledger_transactions"}, 8, 3
            )
        ),
    )
    monkeypatch.setattr(
        llm,
        "generate_discrepancy_explanation",
        Mock(
            return_value=_llm_result(
                {
                    "summary": "Review deterministic results.",
                    "unsupported_conclusions": [],
                    "assumptions": [],
                },
                12,
                4,
            )
        ),
    )
    report_mock = Mock(
        return_value=_llm_result(
            {"title": "Report", "approval_status": "approved"}, 20, 6
        )
    )
    monkeypatch.setattr(
        llm, "generate_final_reconciliation_report", report_mock
    )
    graph = build_workflow(database_path=database_path)
    config = {"configurable": {"thread_id": "telemetry-integration"}}

    graph.invoke(
        create_workflow_input(
            "Reconcile the ledger.", workflow_id="workflow-integrated"
        ),
        config,
    )
    completed = graph.invoke(Command(resume="approve"), config)

    rows = execute_select(
        "SELECT * FROM telemetry WHERE workflow_id = ?",
        ("workflow-integrated",),
        database_path=database_path,
    )
    assert completed["human_decision"] == "approve"
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 50
    assert rows[0]["output_tokens"] == 15
    assert rows[0]["human_decision"] == "approve"
    assert rows[0]["completed"] == 1
    assert rows[0]["estimated_minutes_saved"] == pytest.approx(15.0)
    assert rows[0]["latency_ms"] >= 0


def _llm_result(data: dict, input_tokens: int, output_tokens: int) -> dict:
    return {
        "data": data,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "model": "mock-model",
    }
