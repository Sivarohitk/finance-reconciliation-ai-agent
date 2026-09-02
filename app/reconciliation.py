"""Deterministic reconciliation of ledger and bank transactions."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.database import DATABASE_PATH, execute_select


MONETARY_TOLERANCE = Decimal("0.01")
MONEY_QUANTUM = Decimal("0.01")
RECORD_CATEGORIES = (
    "matched_transactions",
    "missing_from_bank",
    "missing_from_ledger",
    "amount_mismatches",
    "duplicate_references",
    "currency_mismatches",
    "status_mismatches",
)


def _decimal_amount(value: object) -> Decimal:
    return Decimal(str(value))


def _money_value(value: Decimal) -> float:
    return float(value.quantize(MONEY_QUANTUM))


def _load_transactions(
    table_name: str,
    database_path: str | Path,
) -> list[dict[str, Any]]:
    return execute_select(
        f"""
        SELECT id, transaction_date, reference, merchant, amount, currency, status
        FROM {table_name}
        ORDER BY reference, id
        """,
        database_path=database_path,
    )


def _group_by_reference(
    transactions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for transaction in transactions:
        grouped[str(transaction["reference"])].append(transaction)
    return dict(grouped)


def reconcile_transactions(
    database_path: str | Path = DATABASE_PATH,
    *,
    monetary_tolerance: Decimal = MONETARY_TOLERANCE,
) -> dict[str, Any]:
    """Reconcile Finance transactions using deterministic reference matching."""
    ledger_transactions = _load_transactions(
        "ledger_transactions", database_path
    )
    bank_transactions = _load_transactions("bank_transactions", database_path)
    ledger_by_reference = _group_by_reference(ledger_transactions)
    bank_by_reference = _group_by_reference(bank_transactions)
    records: dict[str, list[dict[str, Any]]] = {
        category: [] for category in RECORD_CATEGORIES
    }
    total_absolute_discrepancy = Decimal("0")

    references = sorted(ledger_by_reference.keys() | bank_by_reference.keys())
    for reference in references:
        ledger_rows = ledger_by_reference.get(reference, [])
        bank_rows = bank_by_reference.get(reference, [])
        ledger_reference_total = sum(
            (_decimal_amount(row["amount"]) for row in ledger_rows),
            start=Decimal("0"),
        )
        bank_reference_total = sum(
            (_decimal_amount(row["amount"]) for row in bank_rows),
            start=Decimal("0"),
        )
        reference_difference = abs(
            ledger_reference_total - bank_reference_total
        )
        if reference_difference > monetary_tolerance:
            total_absolute_discrepancy += reference_difference

        is_duplicate = len(ledger_rows) > 1 or len(bank_rows) > 1
        if is_duplicate:
            records["duplicate_references"].append(
                {
                    "reference": reference,
                    "ledger_count": len(ledger_rows),
                    "bank_count": len(bank_rows),
                    "ledger_transactions": ledger_rows,
                    "bank_transactions": bank_rows,
                }
            )

        if not bank_rows:
            records["missing_from_bank"].extend(
                {
                    "reference": reference,
                    "ledger_transaction": ledger_row,
                }
                for ledger_row in ledger_rows
            )
            continue

        if not ledger_rows:
            records["missing_from_ledger"].extend(
                {
                    "reference": reference,
                    "bank_transaction": bank_row,
                }
                for bank_row in bank_rows
            )
            continue

        if is_duplicate:
            continue

        ledger_row = ledger_rows[0]
        bank_row = bank_rows[0]
        amount_difference = abs(
            _decimal_amount(ledger_row["amount"])
            - _decimal_amount(bank_row["amount"])
        )
        amount_matches = amount_difference <= monetary_tolerance
        currency_matches = ledger_row["currency"] == bank_row["currency"]
        status_matches = ledger_row["status"] == bank_row["status"]
        supporting_record = {
            "reference": reference,
            "ledger_transaction": ledger_row,
            "bank_transaction": bank_row,
        }

        if amount_matches and currency_matches and status_matches:
            records["matched_transactions"].append(supporting_record)
            continue

        if not amount_matches:
            records["amount_mismatches"].append(
                {
                    **supporting_record,
                    "absolute_difference": _money_value(amount_difference),
                }
            )
        if not currency_matches:
            records["currency_mismatches"].append(supporting_record.copy())
        if not status_matches:
            records["status_mismatches"].append(supporting_record.copy())

    total_ledger_amount = sum(
        (_decimal_amount(row["amount"]) for row in ledger_transactions),
        start=Decimal("0"),
    )
    total_bank_amount = sum(
        (_decimal_amount(row["amount"]) for row in bank_transactions),
        start=Decimal("0"),
    )

    return {
        "summary": {
            "total_ledger_amount": _money_value(total_ledger_amount),
            "total_bank_amount": _money_value(total_bank_amount),
            "total_absolute_discrepancy_amount": _money_value(
                total_absolute_discrepancy
            ),
            "counts": {
                category: len(records[category])
                for category in RECORD_CATEGORIES
            },
        },
        "records": records,
    }
