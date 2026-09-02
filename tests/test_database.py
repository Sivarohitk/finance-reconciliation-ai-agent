import sqlite3
from unittest.mock import Mock

import pytest

from app.database import (
    APPROVED_TABLES,
    execute_select,
    get_table_schema,
    list_approved_tables,
)
from app import database
from data.seed_finance import SYNTHETIC_MERCHANTS, create_database


@pytest.fixture(scope="module")
def database_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("finance") / "finance.db"
    create_database(path)
    return path


def scalar(connection, query):
    return connection.execute(query).fetchone()[0]


def test_database_creation_and_expected_tables(database_path):
    assert database_path.exists()
    assert set(list_approved_tables(database_path)) == APPROVED_TABLES

    with sqlite3.connect(database_path) as connection:
        actual_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert APPROVED_TABLES <= actual_tables


def test_transaction_data_exists_and_is_synthetic(database_path):
    with sqlite3.connect(database_path) as connection:
        for table in ("ledger_transactions", "bank_transactions"):
            row_count = scalar(connection, f"SELECT COUNT(*) FROM {table}")
            non_synthetic_references = scalar(
                connection,
                f"SELECT COUNT(*) FROM {table} "
                "WHERE reference NOT GLOB 'SYN-[0-9][0-9][0-9][0-9][0-9][0-9]'",
            )
            merchants = {
                row[0]
                for row in connection.execute(
                    f"SELECT DISTINCT merchant FROM {table}"
                )
            }

            assert 1_900 <= row_count <= 2_100
            assert non_synthetic_references == 0
            assert merchants <= set(SYNTHETIC_MERCHANTS)


def test_all_requested_discrepancy_examples_exist(database_path):
    with sqlite3.connect(database_path) as connection:
        missing_bank = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM ledger_transactions AS ledger
            LEFT JOIN bank_transactions AS bank
                ON bank.reference = ledger.reference
            WHERE bank.id IS NULL
            """,
        )
        missing_ledger = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM bank_transactions AS bank
            LEFT JOIN ledger_transactions AS ledger
                ON ledger.reference = bank.reference
            WHERE ledger.id IS NULL
            """,
        )
        amount_mismatches = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM ledger_transactions AS ledger
            JOIN bank_transactions AS bank USING (reference)
            WHERE ABS(ledger.amount - bank.amount) > 0.0001
            """,
        )
        currency_mismatches = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM ledger_transactions AS ledger
            JOIN bank_transactions AS bank USING (reference)
            WHERE ledger.currency != bank.currency
            """,
        )
        status_mismatches = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM ledger_transactions AS ledger
            JOIN bank_transactions AS bank USING (reference)
            WHERE ledger.status != bank.status
            """,
        )
        duplicate_references = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT reference
                FROM ledger_transactions
                GROUP BY reference
                HAVING COUNT(*) > 1
            )
            """,
        )

    assert missing_bank > 0
    assert missing_ledger > 0
    assert amount_mismatches > 0
    assert currency_mismatches > 0
    assert status_mismatches > 0
    assert duplicate_references > 0


def test_database_helpers_expose_schema_and_read_only_selects(database_path):
    schema = get_table_schema("ledger_transactions", database_path)
    assert [column["name"] for column in schema] == [
        "id",
        "transaction_date",
        "reference",
        "merchant",
        "amount",
        "currency",
        "status",
    ]

    rows = execute_select(
        "SELECT reference, amount FROM ledger_transactions ORDER BY id LIMIT 2",
        database_path=database_path,
    )
    assert len(rows) == 2
    assert set(rows[0]) == {"reference", "amount"}

    with pytest.raises(sqlite3.OperationalError):
        execute_select(
            "DELETE FROM ledger_transactions",
            database_path=database_path,
        )

    with pytest.raises(ValueError, match="not approved"):
        get_table_schema("sqlite_master", database_path)


def test_execute_select_closes_its_connection(monkeypatch):
    connection = Mock()
    connection.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr(database, "get_connection", Mock(return_value=connection))

    assert database.execute_select("SELECT 1") == []
    connection.close.assert_called_once_with()
