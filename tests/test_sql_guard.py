import sqlite3

import pytest
from sqlglot import exp, parse_one

from app.sql_guard import execute_validated_sql, validate_sql
from data.seed_finance import create_database


@pytest.fixture(scope="module")
def database_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("sql_guard") / "finance.db"
    create_database(path)
    return path


def test_accepts_safe_select_and_adds_limit(database_path):
    result = validate_sql(
        "SELECT reference, merchant FROM ledger_transactions ORDER BY id",
        database_path=database_path,
    )

    assert result["is_valid"] is True
    assert result["errors"] == []
    assert result["referenced_tables"] == ["ledger_transactions"]
    assert result["referenced_columns"] == ["id", "merchant", "reference"]
    parsed = parse_one(result["validated_sql"], read="sqlite")
    assert parsed.args["limit"].expression.this == "100"


def test_accepts_safe_aggregate_without_adding_limit(database_path):
    result = validate_sql(
        "SELECT status, COUNT(*) AS total "
        "FROM bank_transactions GROUP BY status",
        database_path=database_path,
    )

    assert result["is_valid"] is True
    assert parse_one(result["validated_sql"], read="sqlite").args.get("limit") is None


def test_preserves_existing_limit(database_path):
    result = validate_sql(
        "SELECT reference FROM ledger_transactions LIMIT 5",
        database_path=database_path,
    )

    parsed = parse_one(result["validated_sql"], read="sqlite")
    assert parsed.args["limit"].expression.this == "5"


@pytest.mark.parametrize(
    "candidate_sql",
    [
        "INSERT INTO ledger_transactions (reference) VALUES ('unsafe')",
        "UPDATE ledger_transactions SET amount = 0",
        "DELETE FROM ledger_transactions",
        "DROP TABLE ledger_transactions",
        "ALTER TABLE ledger_transactions ADD COLUMN unsafe TEXT",
        "CREATE TABLE unsafe (id INTEGER)",
        "PRAGMA table_info(ledger_transactions)",
        "ATTACH DATABASE 'other.db' AS other",
    ],
)
def test_rejects_non_select_statements(candidate_sql, database_path):
    result = validate_sql(candidate_sql, database_path=database_path)

    assert result["is_valid"] is False
    assert result["validated_sql"] is None
    assert result["errors"]


def test_rejects_multiple_statements(database_path):
    result = validate_sql(
        "SELECT * FROM ledger_transactions; DROP TABLE ledger_transactions",
        database_path=database_path,
    )

    assert result["is_valid"] is False
    assert any("one SQL statement" in error for error in result["errors"])


@pytest.mark.parametrize("table_name", ["telemetry", "sqlite_master", "sqlite_schema"])
def test_rejects_unapproved_and_system_tables(table_name, database_path):
    result = validate_sql(
        f"SELECT * FROM {table_name}",
        database_path=database_path,
    )

    assert result["is_valid"] is False
    assert table_name in result["referenced_tables"]


def test_rejects_unknown_columns(database_path):
    result = validate_sql(
        "SELECT account_number FROM ledger_transactions",
        database_path=database_path,
    )

    assert result["is_valid"] is False
    assert result["referenced_columns"] == ["account_number"]
    assert any("Unknown column" in error for error in result["errors"])


def test_validates_qualified_columns_against_the_correct_table(database_path):
    result = validate_sql(
        """
        SELECT ledger.reference, bank.amount
        FROM ledger_transactions AS ledger
        JOIN bank_transactions AS bank
            ON bank.reference = ledger.reference
        """,
        database_path=database_path,
    )

    assert result["is_valid"] is True
    assert result["referenced_tables"] == [
        "bank_transactions",
        "ledger_transactions",
    ]
    assert result["referenced_columns"] == ["amount", "reference"]


def test_executes_only_successfully_validated_sql(database_path):
    valid_result = validate_sql(
        "SELECT reference FROM ledger_transactions ORDER BY id LIMIT 2",
        database_path=database_path,
    )
    rows = execute_validated_sql(valid_result, database_path=database_path)
    assert len(rows) == 2

    invalid_result = validate_sql(
        "DELETE FROM ledger_transactions",
        database_path=database_path,
    )
    with pytest.raises(ValueError, match="valid SQL"):
        execute_validated_sql(invalid_result, database_path=database_path)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM ledger_transactions"
        ).fetchone()[0] > 0
