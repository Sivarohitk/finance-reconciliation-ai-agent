"""Local SQLite telemetry for portfolio workflow usage estimates."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path
from typing import Any

from app import config
from app.database import DATABASE_PATH, execute_select, get_connection


TOKEN_RATE_DENOMINATOR = 1_000_000
HUMAN_DECISIONS = frozenset({"approve", "reject"})


def _require_non_negative(value: int | float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def calculate_estimated_inference_cost(
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate inference cost using configured per-million-token rates."""
    _require_non_negative(input_tokens, "input_tokens")
    _require_non_negative(output_tokens, "output_tokens")
    input_rate = config.INPUT_COST_PER_MILLION_TOKENS
    output_rate = config.OUTPUT_COST_PER_MILLION_TOKENS
    if input_rate is None or output_rate is None:
        return None
    _require_non_negative(input_rate, "INPUT_COST_PER_MILLION_TOKENS")
    _require_non_negative(output_rate, "OUTPUT_COST_PER_MILLION_TOKENS")
    return (
        input_tokens * input_rate + output_tokens * output_rate
    ) / TOKEN_RATE_DENOMINATOR


def calculate_estimated_minutes_saved(completed: bool) -> float:
    """Estimate saved minutes from the configured manual-task baseline."""
    baseline = float(config.MANUAL_TASK_BASELINE_MINUTES)
    _require_non_negative(baseline, "MANUAL_TASK_BASELINE_MINUTES")
    return baseline if completed else 0.0


def record_workflow_telemetry(
    *,
    workflow_id: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    validation_failures: int,
    human_decision: str | None,
    completed: bool,
    database_path: str | Path = DATABASE_PATH,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Persist one idempotent local workflow telemetry record."""
    if not workflow_id.strip():
        raise ValueError("workflow_id must be non-empty.")
    _require_non_negative(latency_ms, "latency_ms")
    _require_non_negative(validation_failures, "validation_failures")
    if human_decision is not None and human_decision not in HUMAN_DECISIONS:
        raise ValueError("human_decision must be approve, reject, or None.")

    created_at = timestamp or datetime.now(timezone.utc).isoformat()
    estimated_cost = calculate_estimated_inference_cost(
        input_tokens, output_tokens
    )
    estimated_minutes_saved = calculate_estimated_minutes_saved(completed)
    record = {
        "workflow_id": workflow_id,
        "timestamp": created_at,
        "latency_ms": float(latency_ms),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimated_cost,
        "validation_failures": validation_failures,
        "human_decision": human_decision,
        "completed": completed,
        "estimated_minutes_saved": estimated_minutes_saved,
    }

    database_record = {
        **record,
        "created_at": record["timestamp"],
        "completed": int(completed),
    }
    with closing(get_connection(database_path)) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE telemetry SET
                created_at = :created_at,
                latency_ms = :latency_ms,
                input_tokens = :input_tokens,
                output_tokens = :output_tokens,
                estimated_cost_usd = :estimated_cost_usd,
                validation_failures = :validation_failures,
                human_decision = :human_decision,
                completed = :completed,
                estimated_minutes_saved = :estimated_minutes_saved
            WHERE workflow_id = :workflow_id
            """,
            database_record,
        )
        if cursor.rowcount == 0:
            connection.execute(
            """
            INSERT INTO telemetry (
                workflow_id,
                created_at,
                latency_ms,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
                validation_failures,
                human_decision,
                completed,
                estimated_minutes_saved
            ) VALUES (
                :workflow_id,
                :created_at,
                :latency_ms,
                :input_tokens,
                :output_tokens,
                :estimated_cost_usd,
                :validation_failures,
                :human_decision,
                :completed,
                :estimated_minutes_saved
            )
            """,
                database_record,
            )

    return record


def get_aggregate_metrics(
    database_path: str | Path = DATABASE_PATH,
) -> dict[str, int | float]:
    """Return aggregate metrics for local demonstration workflows only."""
    query = """
        SELECT
            COUNT(*) AS total_workflows,
            COALESCE(SUM(completed), 0) AS completed_workflows,
            COALESCE(AVG(latency_ms), 0.0) AS average_latency_ms,
            COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
            COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
            COALESCE(SUM(estimated_cost_usd), 0.0)
                AS estimated_total_inference_cost_usd,
            COALESCE(SUM(validation_failures), 0)
                AS sql_validation_failure_count,
            COALESCE(SUM(estimated_minutes_saved), 0.0)
                AS estimated_total_minutes_saved,
            COALESCE(
                SUM(CASE WHEN human_decision = 'approve' THEN 1 ELSE 0 END)
                * 1.0
                / NULLIF(
                    SUM(
                        CASE
                            WHEN human_decision IN ('approve', 'reject')
                            THEN 1 ELSE 0
                        END
                    ),
                    0
                ),
                0.0
            ) AS approval_rate
        FROM telemetry
    """
    row = execute_select(query, database_path=database_path)[0]

    total_workflows = int(row["total_workflows"])
    completed_workflows = int(row["completed_workflows"])
    completion_rate = (
        completed_workflows / total_workflows if total_workflows else 0.0
    )
    return {
        "total_workflows": total_workflows,
        "completed_workflows": completed_workflows,
        "completion_rate": completion_rate,
        "approval_rate": float(row["approval_rate"]),
        "average_latency_ms": float(row["average_latency_ms"]),
        "total_input_tokens": int(row["total_input_tokens"]),
        "total_output_tokens": int(row["total_output_tokens"]),
        "estimated_total_inference_cost_usd": float(
            row["estimated_total_inference_cost_usd"]
        ),
        "sql_validation_failure_count": int(
            row["sql_validation_failure_count"]
        ),
        "estimated_total_minutes_saved": float(
            row["estimated_total_minutes_saved"]
        ),
    }
