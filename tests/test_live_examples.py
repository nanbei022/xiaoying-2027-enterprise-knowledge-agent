"""Read-only checks of all saved real-model scenarios; no API calls."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCUMENTS = {
    item["doc_id"]: item
    for item in json.loads((ROOT / "data" / "documents.json").read_text(encoding="utf-8"))
}
SCENARIOS = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[item["id"] for item in SCENARIOS])
def test_saved_trace_matches_actual_output(scenario):
    scenario_id = scenario["id"]
    expected_status = scenario["expected_status"]
    run_input = json.loads((EXAMPLES / f"{scenario_id}-input.json").read_text(encoding="utf-8"))
    output = json.loads((EXAMPLES / f"{scenario_id}-output.json").read_text(encoding="utf-8"))

    assert run_input["question"] == scenario["question"]
    assert run_input["history"] == scenario["history"]
    assert run_input["setup"] == scenario["setup"]
    assert run_input["model"] == (
        None if expected_status == "invalid_input" else "deepseek-v4.1-flash-1017"
    )
    assert output["status"] == expected_status
    assert output["decision_steps"] <= run_input["max_steps"]
    assert output["tool_calls"] <= run_input["max_tool_calls"]

    trace_path = EXAMPLES / f"{scenario_id}-trace.jsonl"
    if expected_status == "invalid_input":
        assert output["trace"] == []
        assert output["trace_path"] is None
        assert output["decision_steps"] == 0 and output["tool_calls"] == 0
        assert not trace_path.exists()
        return

    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert output["trace"] == trace
    assert Path(output["trace_path"]).resolve() == trace_path.resolve()
    assert trace[-1]["phase"] == "run_end"
    assert trace[-1]["status"] == output["status"]
    assert all(event["run_id"] == output["run_id"] for event in trace)
    assert all(event["model_kind"].startswith("real_model") for event in trace)
    assert output["answer"] == (output["proposal"] or {}).get("answer", "")
    assert output["citations"] == (output["proposal"] or {}).get("citations", [])
    assert output["conflicts"] == (output["proposal"] or {}).get("conflicts", [])
    assert output["missing_info"] == (output["proposal"] or {}).get("missing_info", [])

    for event in trace:
        if event["phase"] == "tool":
            assert "arguments" in event
            assert "result" in event
            assert event["next_action"]
    cited_sections = {item["section_id"] for item in output["citations"]}
    assert set(scenario["required_sections"]) <= cited_sections
    read_docs = {
        event["result"]["document"]["doc_id"]
        for event in trace
        if event["phase"] == "tool"
        and event["name"] == "read_document"
        and event["result"]["status"] == "success"
    }
    for citation in output["citations"]:
        assert citation["doc_id"] in read_docs
        sections = DOCUMENTS[citation["doc_id"]]["sections"]
        assert any(
            section["section_id"] == citation["section_id"] and citation["quote"] in section["text"]
            for section in sections
        )
    if expected_status == "completed":
        assert output["citations"]
        assert output["missing_info"] == []
        assert next(event for event in trace if event["phase"] == "finish")["validation"] == "passed"
    if expected_status in {"needs_clarification", "insufficient_evidence"}:
        assert output["answer"]
        assert output["missing_info"]


def test_saved_paths_and_failure_boundaries():
    s01 = json.loads((EXAMPLES / "S01-output.json").read_text(encoding="utf-8"))
    s02 = json.loads((EXAMPLES / "S02-output.json").read_text(encoding="utf-8"))
    s03 = json.loads((EXAMPLES / "S03-output.json").read_text(encoding="utf-8"))
    s06 = json.loads((EXAMPLES / "S06-output.json").read_text(encoding="utf-8"))
    s07 = json.loads((EXAMPLES / "S07-output.json").read_text(encoding="utf-8"))
    s08 = json.loads((EXAMPLES / "S08-output.json").read_text(encoding="utf-8"))
    s09 = json.loads((EXAMPLES / "S09-output.json").read_text(encoding="utf-8"))

    assert {c["section_id"] for c in s01["proposal"]["citations"]} == {"D03-S1"}
    assert {c["section_id"] for c in s02["proposal"]["citations"]} == {"D01-S1", "D02-S1"}
    assert s02["proposal"]["conflicts"][0]["doc_ids"] == ["D01", "D02"]
    assert s03["proposal"]["citations"] == []
    assert s03["trace"][1]["result"]["status"] == "no_results"
    assert s03["trace"][1]["next_action"] == "search_documents"
    s03_queries = [
        event["arguments"]["query"]
        for event in s03["trace"]
        if event["phase"] == "model" and event.get("action") == "search_documents"
    ]
    assert len(s03_queries) == 2 and s03_queries[0] != s03_queries[1]
    assert any(
        event["phase"] == "tool" and event["result"]["error"]["code"] == "SIMULATED_FAILURE"
        for event in s06["trace"]
        if event["phase"] == "tool" and event["result"]["status"] == "error"
    )
    assert s07["proposal"] is None and s08["proposal"] is None
    assert s07["trace"][-2]["next_action"] == "limit_reached"
    assert s08["trace"][-2]["next_action"] == "limit_reached"
    assert {c["section_id"] for c in s09["proposal"]["citations"]} == {"D05-S1"}
    first_tool = next(event for event in s09["trace"] if event["phase"] == "tool")
    assert first_tool["result"]["status"] == "no_results"
    assert first_tool["next_action"] == "search_documents"
    s09_queries = [
        event["arguments"]["query"]
        for event in s09["trace"]
        if event["phase"] == "model" and event.get("action") == "search_documents"
    ]
    assert len(s09_queries) >= 2 and s09_queries[0] != s09_queries[1]
