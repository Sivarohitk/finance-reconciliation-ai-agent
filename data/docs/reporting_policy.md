# Synthetic Finance Reporting Policy

This demonstration policy defines reporting expectations for the portfolio application.

## Evidence Requirements

Every reported total and discrepancy must link to deterministic query or reconciliation evidence. Reports should identify the source tables, relevant transaction references, validation results, and retrieved policy excerpts used in the workflow.

## Assumptions

Any assumption must be labeled explicitly as an assumption and kept separate from verified facts. Estimated values, including estimated time savings, must be clearly labeled as estimates and must not be presented as observed performance.

## Deterministic Results and AI Commentary

Deterministic Python and SQL results are authoritative for financial totals, counts, and discrepancy amounts. AI commentary may summarize or explain those results, but it must be clearly distinguished from deterministic evidence and must not replace or modify authoritative calculations.

## Approval Before Final Reports

A human Finance reviewer must approve the evidence, assumptions, and proposed commentary before a report is marked final. A rejected review returns the report for correction; the workflow must not present rejected or unreviewed output as final.
