import sqlite3

import pytest

from app.reconciliation import reconcile_transactions


TRANSACTION_SCHEMA = """
CREATE TABLE ledger_transactions (
    id INTEGER PRIMARY KEY,
    transaction_date TEXT NOT NULL,
    reference TEXT NOT NULL,
    merchant TEXT NOT NULL,
    amount NUMERIC NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE bank_transactions (
    id INTEGER PRIMARY KEY,
    transaction_date TEXT NOT NULL,
    reference TEXT NOT NULL,
    merchant TEXT NOT NULL,
    amount NUMERIC NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL
);
"""


def transaction(reference, amount, currency="USD", status="posted"):
    return (
        "2025-01-15",
        reference,
        "Synthetic Test Merchant",
        amount,
        currency,
        status,
    )


@pytest.fixture()
def reconciliation_database(tmp_path):
    path = tmp_path / "reconciliation.db"
    ledger_rows = [
        transaction("MATCHED", 100.00),
        transaction("TOLERANCE", 10.00),
        transaction("MISSING-BANK", 25.00),
        transaction("AMOUNT", 100.00),
        transaction("DUPLICATE", 40.00),
        transaction("DUPLICATE", 40.00),
        transaction("CURRENCY", 50.00, currency="USD"),
        transaction("STATUS", 60.00, status="posted"),
    ]
    bank_rows = [
        transaction("MATCHED", 100.00),
        transaction("TOLERANCE", 10.01),
        transaction("MISSING-LEDGER", 30.00),
        transaction("AMOUNT", 100.02),
        transaction("DUPLICATE", 40.00),
        transaction("CURRENCY", 50.00, currency="EUR"),
        transaction("STATUS", 60.00, status="pending"),
    ]

    with sqlite3.connect(path) as connection:
        connection.executescript(TRANSACTION_SCHEMA)
        connection.executemany(
            """
            INSERT INTO ledger_transactions (
                transaction_date, reference, merchant, amount, currency, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ledger_rows,
        )
        connection.executemany(
            """
            INSERT INTO bank_transactions (
                transaction_date, reference, merchant, amount, currency, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            bank_rows,
        )

    return path


def record_references(result, category):
    return {record["reference"] for record in result["records"][category]}


def test_reconciliation_summary_and_authoritative_totals(reconciliation_database):
    result = reconcile_transactions(reconciliation_database)

    assert result["summary"] == {
        "total_ledger_amount": 425.0,
        "total_bank_amount": 390.03,
        "total_absolute_discrepancy_amount": 95.02,
        "counts": {
            "matched_transactions": 2,
            "missing_from_bank": 1,
            "missing_from_ledger": 1,
            "amount_mismatches": 1,
            "duplicate_references": 1,
            "currency_mismatches": 1,
            "status_mismatches": 1,
        },
    }


def test_matches_by_reference_and_honors_one_cent_tolerance(
    reconciliation_database,
):
    result = reconcile_transactions(reconciliation_database)
    assert record_references(result, "matched_transactions") == {
        "MATCHED",
        "TOLERANCE",
    }


def test_detects_transactions_missing_from_each_source(reconciliation_database):
    result = reconcile_transactions(reconciliation_database)
    assert record_references(result, "missing_from_bank") == {"MISSING-BANK"}
    assert record_references(result, "missing_from_ledger") == {
        "MISSING-LEDGER"
    }


def test_detects_amount_mismatches_with_supporting_records(
    reconciliation_database,
):
    result = reconcile_transactions(reconciliation_database)
    mismatch = result["records"]["amount_mismatches"][0]

    assert mismatch["reference"] == "AMOUNT"
    assert mismatch["ledger_transaction"]["amount"] == 100
    assert mismatch["bank_transaction"]["amount"] == 100.02
    assert mismatch["absolute_difference"] == 0.02


def test_detects_duplicate_references(reconciliation_database):
    result = reconcile_transactions(reconciliation_database)
    duplicate = result["records"]["duplicate_references"][0]

    assert duplicate["reference"] == "DUPLICATE"
    assert duplicate["ledger_count"] == 2
    assert duplicate["bank_count"] == 1
    assert len(duplicate["ledger_transactions"]) == 2


def test_detects_currency_mismatches(reconciliation_database):
    result = reconcile_transactions(reconciliation_database)
    assert record_references(result, "currency_mismatches") == {"CURRENCY"}


def test_detects_status_mismatches(reconciliation_database):
    result = reconcile_transactions(reconciliation_database)
    assert record_references(result, "status_mismatches") == {"STATUS"}
