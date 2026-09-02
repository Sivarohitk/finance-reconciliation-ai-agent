"""Small SQLite access helpers for approved Finance tables."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "finance.db"
APPROVED_TABLES = frozenset(
    {"ledger_transactions", "bank_transactions", "telemetry"}
)


def get_connection(
    database_path: str | Path = DATABASE_PATH,
    *,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Return a SQLite connection with dictionary-like result rows."""
    path = Path(database_path)
    if read_only:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
    else:
        connection = sqlite3.connect(path)

    connection.row_factory = sqlite3.Row
    return connection


def list_approved_tables(
    database_path: str | Path = DATABASE_PATH,
) -> list[str]:
    """List approved application tables that exist in the database."""
    placeholders = ", ".join("?" for _ in APPROVED_TABLES)
    query = (
        "SELECT name FROM sqlite_master "
        f"WHERE type = 'table' AND name IN ({placeholders}) ORDER BY name"
    )
    with closing(get_connection(database_path, read_only=True)) as connection:
        return [
            row["name"]
            for row in connection.execute(query, sorted(APPROVED_TABLES)).fetchall()
        ]


def get_table_schema(
    table_name: str,
    database_path: str | Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return column metadata for an approved table."""
    if table_name not in APPROVED_TABLES:
        raise ValueError(f"Table is not approved: {table_name}")

    with closing(get_connection(database_path, read_only=True)) as connection:
        rows = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()

    return [
        {
            "cid": row["cid"],
            "name": row["name"],
            "type": row["type"],
            "not_null": bool(row["notnull"]),
            "default_value": row["dflt_value"],
            "primary_key": bool(row["pk"]),
        }
        for row in rows
    ]


def execute_select(
    query: str,
    parameters: Sequence[object] | Mapping[str, object] = (),
    *,
    database_path: str | Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Execute an already-validated query through a read-only connection."""
    with closing(get_connection(database_path, read_only=True)) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]
