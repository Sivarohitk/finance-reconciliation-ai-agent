"""Schema-aware validation and read-only execution for analytical SQL."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import OptimizeError

from app.database import DATABASE_PATH, execute_select, get_table_schema


APPROVED_QUERY_TABLES = frozenset(
    {"ledger_transactions", "bank_transactions"}
)
DEFAULT_ROW_LIMIT = 100
DISALLOWED_NODE_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Pragma,
    exp.Attach,
    exp.Command,
    exp.Into,
)


class SQLValidationResult(TypedDict):
    is_valid: bool
    validated_sql: str | None
    errors: list[str]
    referenced_tables: list[str]
    referenced_columns: list[str]


def _result(
    *,
    errors: list[str],
    validated_sql: str | None = None,
    referenced_tables: set[str] | None = None,
    referenced_columns: set[str] | None = None,
) -> SQLValidationResult:
    return {
        "is_valid": not errors,
        "validated_sql": validated_sql if not errors else None,
        "errors": errors,
        "referenced_tables": sorted(referenced_tables or set()),
        "referenced_columns": sorted(referenced_columns or set()),
    }


def _database_schema(database_path: str | Path) -> dict[str, dict[str, str]]:
    return {
        table_name: {
            column["name"]: column["type"] or "TEXT"
            for column in get_table_schema(table_name, database_path)
        }
        for table_name in APPROVED_QUERY_TABLES
    }


def _has_row_aggregate(expression: exp.Expression) -> bool:
    return any(
        aggregate.find_ancestor(exp.Window) is None
        for aggregate in expression.find_all(exp.AggFunc)
    )


def validate_sql(
    candidate_sql: str,
    *,
    database_path: str | Path = DATABASE_PATH,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> SQLValidationResult:
    """Validate one candidate query and return a fail-closed result."""
    errors: list[str] = []
    referenced_tables: set[str] = set()
    referenced_columns: set[str] = set()

    if not isinstance(candidate_sql, str) or not candidate_sql.strip():
        return _result(errors=["SQL must be a non-empty string."])
    if row_limit <= 0:
        return _result(errors=["Row limit must be a positive integer."])

    try:
        statements = [
            statement
            for statement in sqlglot.parse(candidate_sql, read="sqlite")
            if statement is not None
        ]
    except ParseError as exc:
        return _result(errors=[f"SQL parsing failed: {exc}"])

    if len(statements) != 1:
        return _result(errors=["Exactly one SQL statement is allowed."])

    expression = statements[0]
    referenced_columns = {
        column.name.lower()
        for column in expression.find_all(exp.Column)
        if column.name and column.name != "*"
    }
    cte_names = {
        cte.alias_or_name.lower()
        for cte in expression.find_all(exp.CTE)
        if cte.alias_or_name
    }
    physical_tables = [
        table
        for table in expression.find_all(exp.Table)
        if table.name.lower() not in cte_names
    ]
    referenced_tables = {table.name.lower() for table in physical_tables}

    if not isinstance(expression, exp.Query) or not expression.find(exp.Select):
        errors.append("Only SELECT queries are allowed.")
    if any(expression.find(node_type) for node_type in DISALLOWED_NODE_TYPES):
        errors.append("Query contains a prohibited SQL operation.")

    qualified_tables = [
        table.sql(dialect="sqlite")
        for table in physical_tables
        if table.db or table.catalog
    ]
    if qualified_tables:
        errors.append(
            "Database-qualified table names are not allowed: "
            + ", ".join(sorted(qualified_tables))
        )

    unapproved_tables = referenced_tables - APPROVED_QUERY_TABLES
    if unapproved_tables:
        errors.append(
            "Unapproved table reference(s): "
            + ", ".join(sorted(unapproved_tables))
        )

    if not errors:
        try:
            qualified_expression = qualify(
                expression.copy(),
                dialect="sqlite",
                schema=_database_schema(database_path),
                validate_qualify_columns=True,
            )
            referenced_columns = {
                column.name.lower()
                for column in qualified_expression.find_all(exp.Column)
                if column.name and column.name != "*"
            }
        except (OptimizeError, ValueError) as exc:
            errors.append(f"Unknown column or invalid column reference: {exc}")

    if errors:
        return _result(
            errors=errors,
            referenced_tables=referenced_tables,
            referenced_columns=referenced_columns,
        )

    if expression.args.get("limit") is None and not _has_row_aggregate(expression):
        expression = expression.limit(row_limit)

    return _result(
        errors=[],
        validated_sql=expression.sql(dialect="sqlite"),
        referenced_tables=referenced_tables,
        referenced_columns=referenced_columns,
    )


def execute_validated_sql(
    validation_result: Mapping[str, Any],
    *,
    database_path: str | Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Execute SQL only when supplied through a successful validation result."""
    validated_sql = validation_result.get("validated_sql")
    if (
        validation_result.get("is_valid") is not True
        or validation_result.get("errors")
        or not isinstance(validated_sql, str)
        or not validated_sql.strip()
    ):
        raise ValueError("A successful validation result with valid SQL is required.")

    return execute_select(validated_sql, database_path=database_path)
