import json
from pathlib import Path

import pytest

from knowledge_agent import CorpusError, KnowledgeBase


CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "documents.json"
SCENARIOS_PATH = Path(__file__).resolve().parents[1] / "data" / "scenarios.json"
RETRIEVAL_CASES = [
    case
    for case in json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if case["evidence_kind"] == "real_corpus"
    and not case["history"]
    and case["required_sections"]
]


@pytest.fixture
def knowledge_base():
    return KnowledgeBase.from_json(CORPUS_PATH)


def section_ids(result):
    return {hit["section_id"] for hit in result["hits"]}


def test_loads_corpus_and_reads_exact_source(knowledge_base):
    result = knowledge_base.read_document("D03")
    assert result["status"] == "success"
    assert result["document"]["version_date"] == "2026-05-01"
    assert result["document"]["sections"][0]["text"] == "正式员工申请年假，应至少提前3个工作日提交申请。"


def test_conflicting_sources_are_both_returned(knowledge_base):
    result = knowledge_base.search_documents("员工费用报销期限是多久？")
    assert result["status"] == "success"
    assert {"D01-S1", "D02-S1"} <= section_ids(result)
    assert all(hit["priority"] is None for hit in result["hits"] if hit["doc_id"] in {"D01", "D02"})


@pytest.mark.parametrize(
    ("query", "required"),
    [
        ("正式员工申请年假需要提前多久？", "D03-S1"),
        ("正式员工年休假最晚何时提交申请？", "D03-S1"),
        ("会议室预约取消需要提前多久？", "D10-S2"),
        ("会议室预约要提前多久提交？", "D10-S1"),
        ("示例剪辑器桌面版V2能导出WebM吗？", "D09-S2"),
    ],
)
def test_retrieval_finds_required_sections(knowledge_base, query, required):
    result = knowledge_base.search_documents(query)
    assert result["status"] == "success"
    assert required in section_ids(result)


@pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=lambda case: case["id"])
def test_scenario_retrieval_recalls_required_evidence(knowledge_base, case):
    result = knowledge_base.search_documents(case["question"])
    assert result["status"] == "success"
    assert set(case["required_sections"]) <= section_ids(result)


def test_search_keeps_explicit_version_metadata(knowledge_base):
    result = knowledge_base.search_documents("示例剪辑器桌面版支持导出哪些视频格式？")
    by_id = {hit["doc_id"]: hit for hit in result["hits"]}
    assert {"D08", "D09"} <= by_id.keys()
    assert by_id["D09"]["supersedes"] == ["D08"]


def test_scope_filter_excludes_wrong_population(knowledge_base):
    result = knowledge_base.search_documents("出差申请要提前几天？", scope="正式员工")
    assert "D06-S1" in section_ids(result)
    assert "D07-S1" not in section_ids(result)


@pytest.mark.parametrize("query", ["出差住宿补贴标准是多少？", "月度绩效奖金如何计算？"])
def test_missing_topic_is_not_reported_as_a_hit(knowledge_base, query):
    result = knowledge_base.search_documents(query)
    assert result["status"] == "no_results"
    assert result["hits"] == []
    assert result["error"] is None


@pytest.mark.parametrize(
    ("query", "scope", "code"),
    [
        ("", None, "INVALID_QUERY"),
        (None, None, "INVALID_QUERY"),
        ("年假", "", "INVALID_SCOPE"),
    ],
)
def test_invalid_search_arguments_are_errors(knowledge_base, query, scope, code):
    result = knowledge_base.search_documents(query, scope)
    assert result["status"] == "error"
    assert result["error"]["code"] == code
    assert result["hits"] == []


def test_unknown_and_malformed_document_ids_are_errors(knowledge_base):
    assert knowledge_base.read_document("D99")["error"]["code"] == "DOC_NOT_FOUND"
    assert knowledge_base.read_document("../D03")["error"]["code"] == "INVALID_DOC_ID"


def test_backend_exception_is_not_disguised_as_no_results(knowledge_base, monkeypatch):
    def broken_search(query, scope):
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(knowledge_base, "_search_hits", broken_search)
    result = knowledge_base.search_documents("正式员工年假")
    assert result["status"] == "error"
    assert result["error"]["code"] == "TOOL_EXECUTION_ERROR"
    assert result["hits"] == []


def test_duplicate_doc_ids_are_rejected(tmp_path):
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus.append(corpus[0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CorpusError, match="duplicate doc_id"):
        KnowledgeBase.from_json(path)


def test_missing_or_invalid_corpus_is_rejected(tmp_path):
    with pytest.raises(CorpusError, match="cannot read"):
        KnowledgeBase.from_json(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('[{"doc_id": "D01"}]', encoding="utf-8")
    with pytest.raises(CorpusError, match="invalid document corpus"):
        KnowledgeBase.from_json(bad)
