"""LangGraph orchestration for the Finance reconciliation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app import database, llm, reconciliation, sql_guard, telemetry
from app.rag import PolicyRetriever


MIN_GROUNDING_CONFIDENCE = 0.75


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    user_question: str
    plan: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]
    candidate_sql: str
    sql_validation: dict[str, Any]
    sql_results: list[dict[str, Any]]
    reconciliation_results: dict[str, Any]
    discrepancy_analysis: dict[str, Any]
    confidence_score: float
    validation_failures: int
    human_decision: str | None
    final_report: dict[str, Any] | None
    token_usage: TokenUsage
    errors: list[str]
    workflow_started_at: float


def create_workflow_input(
    user_question: str,
    *,
    workflow_id: str | None = None,
) -> WorkflowState:
    """Create a complete initial state for a new workflow run."""
    return {
        "workflow_id": workflow_id or str(uuid4()),
        "user_question": user_question,
        "plan": {},
        "retrieved_evidence": [],
        "candidate_sql": "",
        "sql_validation": {},
        "sql_results": [],
        "reconciliation_results": {},
        "discrepancy_analysis": {},
        "confidence_score": 0.0,
        "validation_failures": 0,
        "human_decision": None,
        "final_report": None,
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "errors": [],
        "workflow_started_at": time(),
    }


def _token_usage_update(
    state: WorkflowState,
    result: Mapping[str, Any],
) -> TokenUsage:
    current = state.get(
        "token_usage", {"input_tokens": 0, "output_tokens": 0}
    )
    usage = result.get("usage", {})
    return {
        "input_tokens": int(current.get("input_tokens", 0))
        + int(usage.get("input_tokens", 0)),
        "output_tokens": int(current.get("output_tokens", 0))
        + int(usage.get("output_tokens", 0)),
    }


def _query_schema(database_path: str | Path) -> dict[str, Any]:
    return {
        table_name: database.get_table_schema(table_name, database_path)
        for table_name in sorted(sql_guard.APPROVED_QUERY_TABLES)
    }


def build_workflow(
    *,
    database_path: str | Path = database.DATABASE_PATH,
    retriever: PolicyRetriever | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build a resumable Finance workflow with a human approval interrupt."""
    policy_retriever = retriever or PolicyRetriever()

    def plan(state: WorkflowState) -> WorkflowState:
        result = llm.create_reconciliation_workflow_plan(
            state["user_question"]
        )
        return {
            "plan": result["data"],
            "token_usage": _token_usage_update(state, result),
        }

    def retrieve_policy_evidence(state: WorkflowState) -> WorkflowState:
        evidence = policy_retriever.retrieve(state["user_question"], top_k=3)
        return {"retrieved_evidence": evidence}

    def generate_candidate_sql(state: WorkflowState) -> WorkflowState:
        result = llm.generate_candidate_sql(
            state["user_question"], _query_schema(database_path)
        )
        candidate_sql = result["data"].get("sql", "")
        return {
            "candidate_sql": (
                candidate_sql if isinstance(candidate_sql, str) else ""
            ),
            "token_usage": _token_usage_update(state, result),
        }

    def validate_sql(state: WorkflowState) -> WorkflowState:
        validation = sql_guard.validate_sql(
            state["candidate_sql"], database_path=database_path
        )
        if validation["is_valid"]:
            return {"sql_validation": validation}

        errors = list(state.get("errors", []))
        errors.extend(validation["errors"])
        return {
            "sql_validation": validation,
            "sql_results": [],
            "validation_failures": state.get("validation_failures", 0) + 1,
            "errors": errors,
        }

    def execute_validated_sql(state: WorkflowState) -> WorkflowState:
        results = sql_guard.execute_validated_sql(
            state["sql_validation"], database_path=database_path
        )
        return {"sql_results": results}

    def run_deterministic_reconciliation(
        state: WorkflowState,
    ) -> WorkflowState:
        results = reconciliation.reconcile_transactions(database_path)
        return {"reconciliation_results": results}

    def analyze_discrepancies(state: WorkflowState) -> WorkflowState:
        result = llm.generate_discrepancy_explanation(
            state["reconciliation_results"], state["retrieved_evidence"]
        )
        return {
            "discrepancy_analysis": result["data"],
            "token_usage": _token_usage_update(state, result),
        }

    def assess_grounding(state: WorkflowState) -> WorkflowState:
        has_evidence = bool(state.get("retrieved_evidence"))
        unsupported = state.get("discrepancy_analysis", {}).get(
            "unsupported_conclusions", []
        )
        if not has_evidence:
            confidence = 0.0
        elif unsupported:
            confidence = 0.5
        else:
            confidence = 1.0

        errors = list(state.get("errors", []))
        if confidence < MIN_GROUNDING_CONFIDENCE:
            errors.append(
                "Insufficient policy evidence for grounded conclusions."
            )
        return {"confidence_score": confidence, "errors": errors}

    def request_human_review(state: WorkflowState) -> WorkflowState:
        decision = interrupt(
            {
                "workflow_id": state["workflow_id"],
                "allowed_decisions": sorted(telemetry.HUMAN_DECISIONS),
                "confidence_score": state["confidence_score"],
                "grounding_insufficient": (
                    state["confidence_score"] < MIN_GROUNDING_CONFIDENCE
                ),
                "validation_failures": state["validation_failures"],
                "sql_validation": state["sql_validation"],
                "reconciliation_summary": state[
                    "reconciliation_results"
                ].get("summary", {}),
                "discrepancy_analysis": state["discrepancy_analysis"],
                "retrieved_evidence": state["retrieved_evidence"],
            }
        )
        if decision not in telemetry.HUMAN_DECISIONS:
            raise ValueError("Human decision must be approve or reject.")
        return {"human_decision": decision}

    def generate_final_report(state: WorkflowState) -> WorkflowState:
        result = llm.generate_final_reconciliation_report(
            state["reconciliation_results"],
            state["retrieved_evidence"],
            state["discrepancy_analysis"],
            human_decision="approved",
        )
        report = dict(result["data"])
        report["deterministic_results"] = state["reconciliation_results"]
        report["approval_status"] = "approved"
        return {
            "final_report": report,
            "token_usage": _token_usage_update(state, result),
        }

    def record_telemetry(state: WorkflowState) -> WorkflowState:
        usage = state.get(
            "token_usage", {"input_tokens": 0, "output_tokens": 0}
        )
        telemetry.record_workflow_telemetry(
            workflow_id=state["workflow_id"],
            latency_ms=max(
                0.0, (time() - state["workflow_started_at"]) * 1_000
            ),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            validation_failures=state.get("validation_failures", 0),
            human_decision=state.get("human_decision"),
            completed=True,
            database_path=database_path,
        )
        return {}

    def route_after_validation(state: WorkflowState) -> str:
        if state["sql_validation"].get("is_valid") is True:
            return "execute_validated_sql"
        return "run_deterministic_reconciliation"

    def route_after_human_review(state: WorkflowState) -> str:
        if state.get("human_decision") == "approve":
            return "generate_final_report"
        return "record_telemetry"

    graph = StateGraph(WorkflowState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve_policy_evidence", retrieve_policy_evidence)
    graph.add_node("generate_candidate_sql", generate_candidate_sql)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("execute_validated_sql", execute_validated_sql)
    graph.add_node(
        "run_deterministic_reconciliation", run_deterministic_reconciliation
    )
    graph.add_node("analyze_discrepancies", analyze_discrepancies)
    graph.add_node("assess_grounding", assess_grounding)
    graph.add_node("request_human_review", request_human_review)
    graph.add_node("generate_final_report", generate_final_report)
    graph.add_node("record_telemetry", record_telemetry)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve_policy_evidence")
    graph.add_edge("retrieve_policy_evidence", "generate_candidate_sql")
    graph.add_edge("generate_candidate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "execute_validated_sql": "execute_validated_sql",
            "run_deterministic_reconciliation": (
                "run_deterministic_reconciliation"
            ),
        },
    )
    graph.add_edge(
        "execute_validated_sql", "run_deterministic_reconciliation"
    )
    graph.add_edge(
        "run_deterministic_reconciliation", "analyze_discrepancies"
    )
    graph.add_edge("analyze_discrepancies", "assess_grounding")
    graph.add_edge("assess_grounding", "request_human_review")
    graph.add_conditional_edges(
        "request_human_review",
        route_after_human_review,
        {
            "generate_final_report": "generate_final_report",
            "record_telemetry": "record_telemetry",
        },
    )
    graph.add_edge("generate_final_report", "record_telemetry")
    graph.add_edge("record_telemetry", END)

    return graph.compile(
        checkpointer=MemorySaver() if checkpointer is None else checkpointer
    )
