from app.rag import PolicyRetriever, load_policy_documents, split_markdown


def test_loads_all_synthetic_policy_documents():
    documents = load_policy_documents()

    assert {document["document_name"] for document in documents} == {
        "reconciliation_policy.md",
        "reporting_policy.md",
        "sql_safety_policy.md",
    }
    assert all(document["text"].strip() for document in documents)


def test_splits_markdown_into_heading_based_chunks():
    chunks = split_markdown(
        "example.md",
        "# Example\n\nIntro text.\n\n## Rules\n\nSpecific rule text.",
    )

    assert len(chunks) == 2
    assert chunks[0]["document_name"] == "example.md"
    assert chunks[1]["text"].startswith("## Rules")


def test_retrieves_reconciliation_amount_tolerance_policy():
    results = PolicyRetriever().retrieve(
        "What amount tolerance is allowed when reconciling transactions?",
        top_k=2,
    )

    assert results[0]["document_name"] == "reconciliation_policy.md"
    assert "$0.01" in results[0]["chunk_text"]


def test_retrieves_reporting_approval_policy():
    results = PolicyRetriever().retrieve(
        "Who must approve evidence and assumptions before the final report?",
        top_k=2,
    )

    assert results[0]["document_name"] == "reporting_policy.md"
    assert "approve" in results[0]["chunk_text"].lower()


def test_retrieves_sql_safety_policy_for_prohibited_operations():
    results = PolicyRetriever().retrieve(
        "Which SQL operations are prohibited and which tables are approved?",
        top_k=2,
    )

    assert results[0]["document_name"] == "sql_safety_policy.md"
    assert any(
        operation in results[0]["chunk_text"]
        for operation in ("INSERT", "UPDATE", "DELETE")
    )


def test_retrieval_results_are_structured_and_score_sorted():
    results = PolicyRetriever().retrieve(
        "deterministic evidence and AI commentary",
        top_k=3,
    )

    assert len(results) == 3
    assert all(
        set(result) == {"document_name", "chunk_text", "retrieval_score"}
        for result in results
    )
    assert [result["retrieval_score"] for result in results] == sorted(
        (result["retrieval_score"] for result in results),
        reverse=True,
    )
