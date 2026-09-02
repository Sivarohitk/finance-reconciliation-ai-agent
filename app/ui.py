"""Streamlit interface backed exclusively by the FastAPI application."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120
EXAMPLE_QUESTION = (
    "Show me transactions with reconciliation discrepancies and summarize "
    "the major causes."
)


class APIRequestError(RuntimeError):
    """Raised when the local FastAPI application cannot fulfill a request."""


def _api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise APIRequestError(
            "The FastAPI service is unavailable. Start it and try again."
        ) from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except requests.JSONDecodeError:
            detail = response.text
        raise APIRequestError(f"API request failed: {detail}")
    return response.json()


def _submit_decision(workflow_id: str, decision: str) -> None:
    with st.spinner(f"Recording {decision} decision..."):
        result = _api_request(
            "POST",
            f"/workflow/{workflow_id}/decision",
            json={"decision": decision},
        )
    st.session_state["decision_result"] = result


st.set_page_config(
    page_title="Finance Reconciliation AI Agent",
    page_icon="💼",
    layout="wide",
)
st.title("Finance Reconciliation & Reporting AI Agent")
st.caption(
    "Local portfolio demonstration using synthetic financial data. "
    "Deterministic reconciliation results remain authoritative."
)

if "workflow_result" not in st.session_state:
    st.session_state["workflow_result"] = None
if "decision_result" not in st.session_state:
    st.session_state["decision_result"] = None


st.header("1. Finance Question")
with st.form("finance_question_form"):
    finance_question = st.text_area(
        "Finance reconciliation question",
        value=EXAMPLE_QUESTION,
        height=100,
    )
    start_submitted = st.form_submit_button(
        "Start reconciliation", type="primary"
    )

if start_submitted:
    try:
        with st.spinner("Running the reconciliation workflow..."):
            st.session_state["workflow_result"] = _api_request(
                "POST",
                "/workflow/start",
                json={"finance_question": finance_question},
            )
        st.session_state["decision_result"] = None
    except APIRequestError as exc:
        st.error(str(exc))


st.header("2. Review & Approval")
workflow_result = st.session_state["workflow_result"]
decision_result = st.session_state["decision_result"]

if workflow_result is None:
    st.info("Submit a Finance question to begin review.")
else:
    st.markdown("**Workflow plan**")
    st.json(workflow_result["workflow_plan"])

    st.markdown("**Generated SQL**")
    st.code(workflow_result["candidate_sql"], language="sql")

    st.markdown("**SQL validation status**")
    sql_validation = workflow_result["sql_validation"]
    if sql_validation.get("is_valid"):
        st.success("Candidate SQL passed deterministic validation.")
    else:
        st.error("Candidate SQL failed deterministic validation.")
    st.json(sql_validation)

    st.markdown("**Retrieved RAG evidence**")
    for index, evidence in enumerate(
        workflow_result["retrieved_evidence"], start=1
    ):
        document_name = evidence.get("document_name", f"Evidence {index}")
        with st.expander(f"{index}. {document_name}"):
            st.write(evidence.get("chunk_text", ""))
            st.caption(
                f"Retrieval score: {evidence.get('retrieval_score', 0):.4f}"
            )

    st.markdown("**Deterministic reconciliation results**")
    deterministic_results = workflow_result[
        "deterministic_reconciliation_results"
    ]
    st.json(deterministic_results.get("summary", {}))
    with st.expander("Supporting reconciliation records"):
        st.json(deterministic_results.get("records", {}))

    st.markdown("**Discrepancy analysis**")
    st.json(workflow_result["discrepancy_analysis"])

    st.markdown("**Confidence and grounding**")
    confidence_column, grounding_column = st.columns(2)
    confidence_column.metric(
        "Confidence score", f"{workflow_result['confidence_score']:.2f}"
    )
    grounding_column.metric(
        "Grounding status",
        "Sufficient"
        if workflow_result["grounding_sufficient"]
        else "Needs review",
    )

    st.markdown("**Validation warnings**")
    validation_warnings = workflow_result["validation_warnings"]
    if validation_warnings:
        for warning in validation_warnings:
            st.warning(warning)
    else:
        st.success("No validation warnings were recorded.")

    if decision_result is None:
        approve_column, reject_column = st.columns(2)
        if approve_column.button(
            "Approve", type="primary", use_container_width=True
        ):
            try:
                _submit_decision(workflow_result["workflow_id"], "approve")
                decision_result = st.session_state["decision_result"]
            except APIRequestError as exc:
                st.error(str(exc))
        if reject_column.button("Reject", use_container_width=True):
            try:
                _submit_decision(workflow_result["workflow_id"], "reject")
                decision_result = st.session_state["decision_result"]
            except APIRequestError as exc:
                st.error(str(exc))

    if decision_result is not None:
        if decision_result["status"] == "approved":
            st.success("Workflow output approved.")
            st.markdown("**Final structured report**")
            st.json(decision_result["final_report"])
        else:
            st.warning(
                "Workflow output rejected. No approved final report was generated."
            )


st.header("3. Telemetry Dashboard")
try:
    telemetry = _api_request("GET", "/telemetry/summary")
except APIRequestError as exc:
    st.error(str(exc))
else:
    first_row = st.columns(4)
    first_row[0].metric("Total workflows", telemetry["total_workflows"])
    first_row[1].metric(
        "Completion rate", f"{telemetry['completion_rate'] * 100:.1f}%"
    )
    first_row[2].metric(
        "Human approval rate", f"{telemetry['approval_rate'] * 100:.1f}%"
    )
    first_row[3].metric(
        "Average latency", f"{telemetry['average_latency_ms']:.0f} ms"
    )

    second_row = st.columns(4)
    second_row[0].metric(
        "Token usage (input / output)",
        f"{telemetry['total_input_tokens']:,} / "
        f"{telemetry['total_output_tokens']:,}",
    )
    second_row[1].metric(
        "Estimated inference cost (USD)",
        f"${telemetry['estimated_total_inference_cost_usd']:.6f}",
    )
    second_row[2].metric(
        "SQL validation failures",
        telemetry["sql_validation_failure_count"],
    )
    second_row[3].metric(
        "Estimated time savings (minutes)",
        f"{telemetry['estimated_total_minutes_saved']:.1f}",
    )
    st.caption(
        "Inference cost and time savings are configuration-based estimates, "
        "not production performance or ROI claims."
    )
