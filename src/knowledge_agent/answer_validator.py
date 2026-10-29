"""Mechanical validation of a model's finish proposal against tool observations."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .knowledge_base import ReadResult


def _read_sections(trace: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Only successful, schema-valid read_document returns count as evidence."""
    documents: dict[str, dict[str, str]] = {}
    for event in trace:
        if event.get("phase") != "tool" or event.get("name") != "read_document":
            continue
        if event.get("status") != "success":
            continue
        try:
            result = ReadResult.model_validate(event.get("result"))
        except ValidationError:
            continue
        if result.document is not None:
            documents[result.document.doc_id] = {
                section.section_id: section.text for section in result.document.sections
            }
    return documents


def validate_finish(proposal: dict[str, Any], trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return structured issues; an empty list means the format and quotes pass.

    This checks source identity and literal quotations. It cannot determine whether
    a quoted passage semantically supports every claim in the answer.
    """
    issues: list[dict[str, Any]] = []
    read_sections = _read_sections(trace)
    status = proposal["status"]

    if status == "completed":
        if not proposal["answer"].strip():
            issues.append({"code": "EMPTY_ANSWER", "message": "completed answer must not be empty"})
        if not proposal["citations"]:
            issues.append({"code": "MISSING_CITATIONS", "message": "completed answer requires a source quotation"})
        if proposal["missing_info"]:
            issues.append({
                "code": "COMPLETED_WITH_MISSING_INFO",
                "message": "completed answer must not report unresolved information gaps",
            })
    elif status in {"needs_clarification", "insufficient_evidence"}:
        if not proposal["answer"].strip():
            issues.append({"code": "EMPTY_EXPLANATION", "message": f"{status} requires an explanation"})
        if not proposal["missing_info"]:
            issues.append({"code": "MISSING_INFO_REQUIRED", "message": f"{status} must state the information gap"})

    for index, citation in enumerate(proposal["citations"]):
        doc_id = citation["doc_id"]
        section_id = citation["section_id"]
        quote = citation["quote"]
        if doc_id not in read_sections:
            issues.append({
                "code": "SOURCE_NOT_READ",
                "message": f"citation {index} refers to a document not successfully read",
                "index": index,
            })
        elif section_id not in read_sections[doc_id]:
            issues.append({
                "code": "SECTION_NOT_FOUND",
                "message": f"citation {index} section is absent from the read document",
                "index": index,
            })
        elif not quote.strip() or quote not in read_sections[doc_id][section_id]:
            issues.append({
                "code": "QUOTE_MISMATCH",
                "message": f"citation {index} quote is not an exact nonempty source substring",
                "index": index,
            })

    for index, conflict in enumerate(proposal["conflicts"]):
        doc_ids = conflict["doc_ids"]
        if len(set(doc_ids)) < 2 or not conflict["description"].strip():
            issues.append({
                "code": "INVALID_CONFLICT",
                "message": f"conflict {index} needs two distinct documents and a description",
                "index": index,
            })
        elif any(doc_id not in read_sections for doc_id in doc_ids):
            issues.append({
                "code": "CONFLICT_SOURCE_NOT_READ",
                "message": f"conflict {index} refers to a document not successfully read",
                "index": index,
            })

    return issues
