"""Evidence validation against actual read_document results."""

from copy import deepcopy
from pathlib import Path

import pytest

from knowledge_agent.answer_validator import validate_finish
from knowledge_agent.knowledge_base import KnowledgeBase


ROOT = Path(__file__).resolve().parents[1]


def read_trace():
    base = KnowledgeBase.from_json(ROOT / "data" / "documents.json")
    return [
        {
            "phase": "tool",
            "name": "read_document",
            "status": "success",
            "result": base.read_document("D01"),
        }
    ]


def valid_proposal():
    return {
        "status": "completed",
        "answer": "员工应在费用发生后30个自然日内提交报销申请。",
        "citations": [
            {
                "doc_id": "D01",
                "section_id": "D01-S1",
                "quote": "员工应在费用发生后30个自然日内提交报销申请。",
            }
        ],
        "conflicts": [],
        "missing_info": [],
    }


def test_exact_quote_from_read_document_passes():
    assert validate_finish(valid_proposal(), read_trace()) == []


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("doc_id", "D02", "SOURCE_NOT_READ"),
        ("section_id", "D01-S99", "SECTION_NOT_FOUND"),
        ("quote", "不存在的原文", "QUOTE_MISMATCH"),
        ("quote", "", "QUOTE_MISMATCH"),
    ],
)
def test_invalid_citation_is_rejected(field, value, expected_code):
    proposal = deepcopy(valid_proposal())
    proposal["citations"][0][field] = value
    issues = validate_finish(proposal, read_trace())
    assert expected_code in {issue["code"] for issue in issues}


def test_search_hit_alone_does_not_authorize_citation():
    proposal = valid_proposal()
    assert validate_finish(proposal, [])[0]["code"] == "SOURCE_NOT_READ"


def test_completed_requires_answer_and_citation():
    proposal = valid_proposal()
    proposal["answer"] = " "
    proposal["citations"] = []
    assert {item["code"] for item in validate_finish(proposal, read_trace())} == {
        "EMPTY_ANSWER",
        "MISSING_CITATIONS",
    }


def test_completed_cannot_report_unresolved_missing_information():
    proposal = valid_proposal()
    proposal["missing_info"] = ["仍需确定适用范围"]
    assert validate_finish(proposal, read_trace())[0]["code"] == "COMPLETED_WITH_MISSING_INFO"


def test_missing_information_status_requires_explanation_and_gap():
    proposal = {
        "status": "insufficient_evidence",
        "answer": "",
        "citations": [],
        "conflicts": [],
        "missing_info": [],
    }
    assert {item["code"] for item in validate_finish(proposal, [])} == {
        "EMPTY_EXPLANATION",
        "MISSING_INFO_REQUIRED",
    }


def test_conflict_must_reference_two_read_documents():
    proposal = valid_proposal()
    proposal["conflicts"] = [{"doc_ids": ["D01", "D02"], "description": "期限冲突"}]
    assert validate_finish(proposal, read_trace())[-1]["code"] == "CONFLICT_SOURCE_NOT_READ"
    proposal["conflicts"] = [{"doc_ids": ["D01", "D01"], "description": "期限冲突"}]
    assert validate_finish(proposal, read_trace())[-1]["code"] == "INVALID_CONFLICT"


def test_invalid_read_result_does_not_count_as_evidence():
    trace = read_trace()
    trace[0]["result"]["document"] = None
    assert validate_finish(valid_proposal(), trace)[0]["code"] == "SOURCE_NOT_READ"
