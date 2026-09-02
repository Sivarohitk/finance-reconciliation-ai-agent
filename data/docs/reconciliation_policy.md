# Synthetic Finance Reconciliation Policy

This demonstration policy applies only to synthetic portfolio data.

## Transaction Matching Rules

Ledger and bank transactions are matched primarily by transaction reference. A reference must be unique on both sides before the records can be treated as an unambiguous match. Deterministic code compares the amount, currency, and status of each matched pair.

## Amount Tolerance

An amount difference of $0.01 or less is within tolerance. A difference greater than $0.01 is an amount mismatch. Authoritative totals and differences must be calculated by deterministic Python or SQL logic.

## Discrepancy Categories

The reconciliation output identifies matched transactions, transactions missing from the bank, transactions missing from the ledger, amount mismatches, duplicate references, currency mismatches, and status mismatches. Each discrepancy must retain its supporting transaction records.

## Escalation Expectations

Duplicate references, unresolved missing transactions, and material amount, currency, or status mismatches require human Finance review. The reviewer should examine the supporting records and approve or reject the proposed resolution before a final report is issued.
