"""Graph and validation tests. The model here is a test double, not an LLM."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from knowledge_agent.agent import build_agent_graph, run_agent
from knowledge_agent.knowledge_base import KnowledgeBase


ROOT = Path(__file__).resolve().parents[1]


class DecisionDouble:
    """Select a call from the observed ToolMessage for repeatable flow tests."""

    def __init__(self, decide):
        self.decide = decide
        self.seen = []

    def bind_tools(self, tools, **kwargs):
        self.tool_names = {item.name for item in tools}
        assert kwargs == {"tool_choice": "required", "parallel_tool_calls": False}
        return self

    def invoke(self, messages):
        observation = next(
            (json.loads(item.content) for item in reversed(messages) if isinstance(item, ToolMessage)),
            None,
        )
        self.seen.append(observation)
        name, args = self.decide(observation, len(self.seen))
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call-{len(self.seen)}"}],
        )


def make_graph(decide, knowledge_base=None):
    model = DecisionDouble(decide)
    base = knowledge_base or KnowledgeBase.from_json(ROOT / "data" / "documents.json")
    graph = build_agent_graph(model, base)
    assert model.tool_names == {"search_documents", "read_document", "finish"}
    return graph, model


def proposed_answer():
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


def test_successful_search_read_finish_answer_and_trace(tmp_path):
    def decide(observation, _step):
        if observation is None:
            return "search_documents", {"query": "费用报销"}
        if observation["status"] == "success" and "hits" in observation:
            return "read_document", {"doc_id": observation["hits"][0]["doc_id"]}
        return "finish", proposed_answer()

    graph, model = make_graph(decide)
    trace_file = tmp_path / "trace.jsonl"
    result = run_agent(graph, "报销期限是多少？", model_kind="test_double", trace_path=trace_file)

    assert result["status"] == "completed"
    assert result["proposal"]["status"] == "completed"
    assert result["answer"] == proposed_answer()["answer"]
    assert result["citations"] == proposed_answer()["citations"]
    assert result["decision_steps"] == 3
    assert result["tool_calls"] == 2
    assert [event["action"] for event in result["trace"] if event["phase"] == "model"] == [
        "search_documents", "read_document", "finish"
    ]
    tool_events = [event for event in result["trace"] if event["phase"] == "tool"]
    assert tool_events[0]["result"]["hits"]
    assert tool_events[0]["next_action"] == "read_document"
    assert tool_events[1]["result"]["document"]["doc_id"]
    assert tool_events[1]["next_action"] == "finish"
    assert model.seen[0] is None
    assert model.seen[1]["status"] == "success"
    assert model.seen[2]["status"] == "success"
    lines = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    assert lines == result["trace"]
    assert all(line["model_kind"] == "test_double" for line in lines)


def test_no_results_causes_query_change():
    def decide(observation, _step):
        if observation is None:
            return "search_documents", {"query": "zzzznonexistent"}
        if observation["status"] == "no_results":
            return "search_documents", {"query": "费用报销"}
        return "finish", {
            "status": "insufficient_evidence",
            "answer": "尚需核实。",
            "citations": [],
            "conflicts": [],
            "missing_info": ["适用范围"],
        }

    graph, model = make_graph(decide)
    result = run_agent(graph, "报销期限？", model_kind="test_double")

    assert result["status"] == "insufficient_evidence"
    assert result["tool_calls"] == 2
    searches = [event for event in result["trace"] if event["phase"] == "tool"]
    assert [event["arguments"]["query"] for event in searches] == ["zzzznonexistent", "费用报销"]
    assert searches[0]["status"] == "no_results"
    assert searches[0]["next_action"] == "search_documents"
    assert model.seen[1]["status"] == "no_results"


def test_tool_error_is_observed_before_model_terminates():
    def decide(observation, _step):
        if observation is None:
            return "read_document", {"doc_id": "D999"}
        assert observation["error"]["code"] == "DOC_NOT_FOUND"
        return "finish", {
            "status": "failed",
            "answer": "资料读取失败。",
            "citations": [],
            "conflicts": [],
            "missing_info": [],
        }

    graph, model = make_graph(decide)
    result = run_agent(graph, "读取不存在的资料", model_kind="test_double")
    assert result["status"] == "failed"
    assert result["tool_calls"] == 1
    assert model.seen[1]["status"] == "error"
    assert [event["next_action"] for event in result["trace"] if event["phase"] == "tool"] == ["finish"]


def test_max_steps_returns_limit_reached():
    graph, _ = make_graph(lambda _observation, _step: ("search_documents", {"query": "报销期限"}))
    result = run_agent(graph, "报销期限？", max_steps=1, model_kind="test_double")
    assert result["status"] == "limit_reached"
    assert result["decision_steps"] == 1
    assert result["tool_calls"] == 1
    assert result["proposal"] is None
    assert result["trace"][-2]["next_action"] == "limit_reached"


def test_max_tool_calls_returns_limit_reached():
    graph, _ = make_graph(lambda _observation, _step: ("search_documents", {"query": "报销期限"}))
    result = run_agent(graph, "报销期限？", max_steps=4, max_tool_calls=1, model_kind="test_double")
    assert result["status"] == "limit_reached"
    assert result["tool_calls"] == 1
    assert result["decision_steps"] == 1
    last_tool = [event for event in result["trace"] if event["phase"] == "tool"][-1]
    assert last_tool["next_action"] == "limit_reached"
    assert last_tool["result"]["status"] == "no_results"


def test_finish_requires_a_source_read_and_model_can_correct():
    def decide(observation, _step):
        if observation is None:
            return "finish", proposed_answer()
        if observation.get("error") and observation["error"]["code"] == "MISSING_SOURCE_READ":
            return "read_document", {"doc_id": "D01"}
        return "finish", proposed_answer()

    graph, model = make_graph(decide)
    result = run_agent(graph, "报销期限？", model_kind="test_double")
    assert result["status"] == "completed"
    assert model.seen[1]["error"]["code"] == "MISSING_SOURCE_READ"
    assert [event["action"] for event in result["trace"] if event["phase"] == "model"] == [
        "finish", "read_document", "finish"
    ]


def test_bad_quote_is_observed_and_model_can_correct():
    def decide(observation, _step):
        if observation is None:
            return "read_document", {"doc_id": "D01"}
        if observation["status"] == "success":
            invalid = proposed_answer()
            invalid["citations"][0]["quote"] = "不存在的原文"
            return "finish", invalid
        assert observation["error"]["code"] == "INVALID_FINISH_PROPOSAL"
        assert observation["error"]["issues"][0]["code"] == "QUOTE_MISMATCH"
        return "finish", proposed_answer()

    graph, model = make_graph(decide)
    result = run_agent(graph, "报销期限？", model_kind="test_double")
    assert result["status"] == "completed"
    assert model.seen[2]["error"]["issues"][0]["code"] == "QUOTE_MISMATCH"
    rejected = [event for event in result["trace"] if event["phase"] == "tool" and event["name"] == "finish"]
    assert len(rejected) == 1
    assert rejected[0]["next_action"] == "finish"


def test_bad_quote_at_step_limit_never_completes():
    def decide(observation, _step):
        if observation is None:
            return "read_document", {"doc_id": "D01"}
        invalid = proposed_answer()
        invalid["citations"][0]["quote"] = "不存在的原文"
        return "finish", invalid

    graph, _ = make_graph(decide)
    result = run_agent(graph, "报销期限？", max_steps=2, model_kind="test_double")
    assert result["status"] == "limit_reached"
    assert result["proposal"] is None
    assert result["answer"] == ""
    assert result["trace"][-2]["next_action"] == "limit_reached"


def test_unknown_tool_is_rejected_then_model_can_end():
    def decide(observation, _step):
        if observation is None:
            return "delete_document", {"doc_id": "D01"}
        assert observation["error"]["code"] == "UNKNOWN_TOOL"
        return "finish", {
            "status": "failed",
            "answer": "动作无效。",
            "citations": [],
            "conflicts": [],
            "missing_info": [],
        }

    graph, _ = make_graph(decide)
    result = run_agent(graph, "报销期限？", model_kind="test_double")
    assert result["status"] == "failed"
    assert result["tool_calls"] == 1


def test_invalid_tool_arguments_are_observed():
    def decide(observation, _step):
        if observation is None:
            return "read_document", {"doc_id": 1}
        assert observation["error"]["code"] == "INVALID_TOOL_ARGS"
        return "finish", {
            "status": "failed",
            "answer": "参数不合法。",
            "citations": [],
            "conflicts": [],
            "missing_info": [],
        }

    graph, _ = make_graph(decide)
    result = run_agent(graph, "读取资料", model_kind="test_double")
    assert result["status"] == "failed"


def test_invalid_run_input_is_explicit():
    graph, model = make_graph(lambda _observation, _step: ("search_documents", {"query": "报销"}))
    result = run_agent(graph, "", model_kind="test_double")
    assert result["status"] == "invalid_input"
    assert model.seen == []


def test_model_failure_is_explicit():
    class BrokenModel:
        def bind_tools(self, _tools, **_kwargs):
            return self

        def invoke(self, _messages):
            raise RuntimeError("simulated model outage")

    base = KnowledgeBase.from_json(ROOT / "data" / "documents.json")
    graph = build_agent_graph(BrokenModel(), base)
    result = run_agent(graph, "报销期限？", model_kind="test_double")
    assert result["status"] == "failed"
    assert result["trace"][0]["error_code"] == "MODEL_ERROR"


def test_model_without_tool_call_fails_explicitly():
    class TextOnlyModel:
        def bind_tools(self, _tools, **_kwargs):
            return self

        def invoke(self, _messages):
            return AIMessage(content="An unsupported plain-text answer")

    base = KnowledgeBase.from_json(ROOT / "data" / "documents.json")
    graph = build_agent_graph(TextOnlyModel(), base)
    result = run_agent(graph, "报销期限？", model_kind="test_double")
    assert result["status"] == "failed"
    assert result["trace"][0]["error_code"] == "EXPECTED_ONE_ACTION"


def test_unexpected_tool_exception_is_returned_to_model():
    class BrokenKnowledgeBase(KnowledgeBase):
        def read_document(self, _doc_id):
            raise RuntimeError("simulated storage outage")

    base = BrokenKnowledgeBase.from_json(ROOT / "data" / "documents.json")

    def decide(observation, _step):
        if observation is None:
            return "read_document", {"doc_id": "D01"}
        assert observation["error"]["code"] == "TOOL_EXECUTION_ERROR"
        return "finish", {
            "status": "failed",
            "answer": "资料服务异常。",
            "citations": [],
            "conflicts": [],
            "missing_info": [],
        }

    graph, model = make_graph(decide, base)
    result = run_agent(graph, "读取资料", model_kind="test_double")
    assert result["status"] == "failed"
    assert model.seen[1]["error"]["code"] == "TOOL_EXECUTION_ERROR"


def test_invalid_tool_result_is_returned_to_model():
    class MalformedKnowledgeBase(KnowledgeBase):
        def read_document(self, _doc_id):
            return {"status": "success", "document": None, "error": None}

    base = MalformedKnowledgeBase.from_json(ROOT / "data" / "documents.json")

    def decide(observation, _step):
        if observation is None:
            return "read_document", {"doc_id": "D01"}
        assert observation["error"]["code"] == "INVALID_TOOL_RESULT"
        return "finish", {
            "status": "failed",
            "answer": "资料工具返回格式错误。",
            "citations": [],
            "conflicts": [],
            "missing_info": [],
        }

    graph, model = make_graph(decide, base)
    result = run_agent(graph, "读取资料", model_kind="test_double")
    assert result["status"] == "failed"
    assert model.seen[1]["error"]["code"] == "INVALID_TOOL_RESULT"


def test_trace_save_failure_does_not_report_success(tmp_path):
    def decide(observation, _step):
        if observation is None:
            return "read_document", {"doc_id": "D01"}
        return "finish", proposed_answer()

    graph, _ = make_graph(decide)
    result = run_agent(graph, "读取资料", model_kind="test_double", trace_path=tmp_path)
    assert result["status"] == "failed"
    assert result["trace_path"] is None
    assert result["trace"][-1]["status"] == "failed"
    assert result["trace"][-1]["error_code"] == "TRACE_WRITE_ERROR"
