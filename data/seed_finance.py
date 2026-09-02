"""Create a reproducible SQLite database containing synthetic finance data."""

from __future__ import annotations

import random
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple


DATABASE_PATH = Path(__file__).resolve().parent / "finance.db"
RANDOM_SEED = 20250901
BASE_TRANSACTION_COUNT = 2_000

SYNTHETIC_MERCHANTS = (
    "Synthetic Office Supply",
    "Synthetic Transit Services",
    "Synthetic Cloud Hosting",
    "Synthetic Equipment Rental",
    "Synthetic Business Catering",
    "Synthetic Parcel Delivery",
    "Synthetic Software Studio",
    "Synthetic Facility Services",
)

CURRENCIES = ("USD", "EUR", "GBP", "CAD")
STATUSES = ("posted", "cleared", "pending")

MISSING_BANK_INDEXES = frozenset(range(0, 20))
MISSING_LEDGER_INDEXES = frozenset(range(20, 40))
AMOUNT_MISMATCH_INDEXES = frozenset(range(40, 60))
CURRENCY_MISMATCH_INDEXES = frozenset(range(60, 75))
STATUS_MISMATCH_INDEXES = frozenset(range(75, 90))
DUPLICATE_REFERENCE_INDEXES = frozenset(range(90, 100))


class Transaction(NamedTuple):
    transaction_date: str
    reference: str
    merchant: str
    amount: float
    currency: str
    status: str


SCHEMA_SQL = """
DROP TABLE IF EXISTS telemetry;
DROP TABLE IF EXISTS bank_transactions;
DROP TABLE IF EXISTS ledger_transactions;

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

CREATE INDEX ledger_transactions_reference_idx
    ON ledger_transactions (reference);
CREATE INDEX bank_transactions_reference_idx
    ON bank_transactions (reference);

CREATE TABLE telemetry (
    workflow_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd REAL,
    validation_failures INTEGER NOT NULL DEFAULT 0,
    human_decision TEXT,
    completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
    estimated_minutes_saved REAL
);
"""


def _base_transaction(index: int, generator: random.Random) -> Transaction:
    transaction_date = date(2025, 1, 1) + timedelta(
        days=generator.randint(0, 364)
    )
    amount = generator.randint(500, 250_000) / 100
    return Transaction(
        transaction_date.isoformat(),
        f"SYN-{index + 1:06d}",
        generator.choice(SYNTHETIC_MERCHANTS),
        amount,
        generator.choice(CURRENCIES),
        generator.choice(STATUSES),
    )


def _generate_transactions() -> tuple[list[Transaction], list[Transaction]]:
    generator = random.Random(RANDOM_SEED)
    ledger_rows: list[Transaction] = []
    bank_rows: list[Transaction] = []

    for index in range(BASE_TRANSACTION_COUNT):
        transaction = _base_transaction(index, generator)

        if index not in MISSING_LEDGER_INDEXES:
            ledger_rows.append(transaction)

        if index not in MISSING_BANK_INDEXES:
            bank_transaction = transaction

            if index in AMOUNT_MISMATCH_INDEXES:
                bank_transaction = bank_transaction._replace(
                    amount=round(
                        bank_transaction.amount + ((index % 7) + 1) * 0.11,
                        2,
                    )
                )
            if index in CURRENCY_MISMATCH_INDEXES:
                bank_transaction = bank_transaction._replace(
                    currency=CURRENCIES[
                        (CURRENCIES.index(bank_transaction.currency) + 1)
                        % len(CURRENCIES)
                    ]
                )
            if index in STATUS_MISMATCH_INDEXES:
                bank_transaction = bank_transaction._replace(
                    status=STATUSES[
                        (STATUSES.index(bank_transaction.status) + 1)
                        % len(STATUSES)
                    ]
                )

            bank_rows.append(bank_transaction)

        if index in DUPLICATE_REFERENCE_INDEXES:
            ledger_rows.append(transaction)

    return ledger_rows, bank_rows


def create_database(database_path: str | Path = DATABASE_PATH) -> Path:
    """Create or replace the synthetic finance tables at ``database_path``."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger_rows, bank_rows = _generate_transactions()

    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(SCHEMA_SQL)
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


if __name__ == "__main__":
    created_path = create_database()
    print(f"Created synthetic finance database: {created_path}")
