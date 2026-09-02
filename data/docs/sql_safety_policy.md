# Synthetic Finance SQL Safety Policy

This demonstration policy governs analytical access to the local SQLite database.

## Read-Only Access

Database queries must run through a read-only SQLite connection. Candidate SQL must never be executed until deterministic validation succeeds.

## Approved Tables

Analytical queries may reference only `ledger_transactions` and `bank_transactions`. The telemetry table, SQLite system objects such as `sqlite_master`, and any attached database objects are not approved for analytical questions.

## Prohibited Operations

INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, and multiple SQL statements are prohibited. Only a single analytical SELECT query is eligible for execution.

## Validation Requirements

Candidate SQL must be parsed with a proper SQL parser. Validation must confirm a single SELECT statement, approved table references, real columns from the current SQLite schema, absence of system objects, and a reasonable row limit for non-aggregate queries.
