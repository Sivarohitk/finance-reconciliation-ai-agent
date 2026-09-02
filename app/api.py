"""Minimal FastAPI surface for the Finance reconciliation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, status
from langgraph.types import Command
from pydantic import BaseModel, StringConstraints

from app import database, telemetry
from app.workflow import (
    MIN_GROUNDING_CONFIDENCE,
    build_workflow,
    create_workflow_input,
)


NonEmptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class WorkflowStartRequest(BaseModel):
    finance_question: NonEmptyText


class WorkflowStartResponse(BaseModel):
    workflow_id: str
    status: Literal["awaiting_human_review"]
    workflow_plan: dict[str, Any]
    candidate_sql: str
    sql_validation: dict[str, Any]
    reconciliation_summary: dict[str, Any]
    deterministic_reconciliation_results: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]
    discrepancy_analysis: dict[str, Any]
    confidence_score: float
    grounding_sufficient: bool
    validation_warnings: list[str]
    human_approval_required: bool


class WorkflowDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]


class WorkflowDecisionResponse(BaseModel):
    workflow_id: str
    status: Literal["approved", "rejected"]
    human_decision: Literal["approve", "reject"]
    final_report: dict[str, Any] | None


class TelemetrySummaryResponse(BaseModel):
    total_workflows: int
    completed_workflows: int
    completion_rate: float
    approval_rate: float
    average_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_total_inference_cost_usd: float
    sql_validation_failure_count: int
    estimated_total_minutes_saved: float


def create_app(
    *, database_path: str | Path = database.DATABASE_PATH
) -> FastAPI:
    """Create the API with one resumable in-memory LangGraph instance."""
    application = FastAPI(title="Finance Reconciliation & Reporting AI Agent")
    workflow_graph = build_workflow(database_path=database_path)

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", service="finance-reconciliation-ai-agent"
        )

    @application.post(
        "/workflow/start", response_model=WorkflowStartResponse
    )
    def start_workflow(
        request: WorkflowStartRequest,
    ) -> WorkflowStartResponse:
        initial_state = create_workflow_input(request.finance_question)
        workflow_id = initial_state["workflow_id"]
        graph_config = {"configurable": {"thread_id": workflow_id}}
        state = workflow_graph.invoke(initial_state, graph_config)
        confidence_score = float(state.get("confidence_score", 0.0))
        return WorkflowStartResponse(
            workflow_id=workflow_id,
            status="awaiting_human_review",
            workflow_plan=state.get("plan", {}),
            candidate_sql=state.get("candidate_sql", ""),
            sql_validation=state.get("sql_validation", {}),
            reconciliation_summary=state.get(
                "reconciliation_results", {}
            ).get("summary", {}),
            deterministic_reconciliation_results=state.get(
                "reconciliation_results", {}
            ),
            retrieved_evidence=state.get("retrieved_evidence", []),
            discrepancy_analysis=state.get("discrepancy_analysis", {}),
            confidence_score=confidence_score,
            grounding_sufficient=(
                confidence_score >= MIN_GROUNDING_CONFIDENCE
            ),
            validation_warnings=state.get("errors", []),
            human_approval_required=True,
        )

    @application.post(
        "/workflow/{workflow_id}/decision",
        response_model=WorkflowDecisionResponse,
    )
    def decide_workflow(
        workflow_id: str,
        request: WorkflowDecisionRequest,
    ) -> WorkflowDecisionResponse:
        graph_config = {"configurable": {"thread_id": workflow_id}}
        snapshot = workflow_graph.get_state(graph_config)
        if not snapshot.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow was not found.",
            )
        if snapshot.values.get("human_decision") in telemetry.HUMAN_DECISIONS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Workflow decision has already been recorded.",
            )

        state = workflow_graph.invoke(
            Command(resume=request.decision), graph_config
        )
        response_status: Literal["approved", "rejected"] = (
            "approved" if request.decision == "approve" else "rejected"
        )
        return WorkflowDecisionResponse(
            workflow_id=workflow_id,
            status=response_status,
            human_decision=request.decision,
            final_report=(
                state.get("final_report")
                if request.decision == "approve"
                else None
            ),
        )

    @application.get(
        "/telemetry/summary", response_model=TelemetrySummaryResponse
    )
    def telemetry_summary() -> TelemetrySummaryResponse:
        return TelemetrySummaryResponse(
            **telemetry.get_aggregate_metrics(database_path)
        )

    return application


app = create_app()
