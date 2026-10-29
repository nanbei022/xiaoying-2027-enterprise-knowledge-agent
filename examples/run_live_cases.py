"""Run selected synthetic scenarios with the configured real model.

This script saves the exact input, returned result and JSONL trace for review.
It never passes expected answers from data/scenarios.json to the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_agent.agent import build_agent_graph, run_agent
from knowledge_agent.knowledge_base import KnowledgeBase, ReadResult, SearchResult, ToolError
from knowledge_agent.model_client import create_chat_model


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent


class PersistentToolFailure(KnowledgeBase):
    """Controlled test backend: both tools always return explicit errors."""

    def search_documents(self, query, scope=None):
        return SearchResult(
            status="error",
            query=query,
            scope=scope,
            hits=[],
            error=ToolError(code="SIMULATED_FAILURE", message="controlled search failure"),
        ).model_dump(mode="json")

    def read_document(self, _doc_id):
        return ReadResult(
            status="error",
            document=None,
            error=ToolError(code="SIMULATED_FAILURE", message="controlled read failure"),
        ).model_dump(mode="json")


class FirstSearchEmpty(KnowledgeBase):
    """Controlled test backend: only the first search returns no_results."""

    def __init__(self, documents):
        super().__init__(documents)
        self.search_count = 0

    def search_documents(self, query, scope=None):
        self.search_count += 1
        if self.search_count == 1:
            return SearchResult(
                status="no_results",
                query=query,
                scope=scope,
                hits=[],
                error=None,
            ).model_dump(mode="json")
        return super().search_documents(query, scope)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_ids", nargs="*", help="Scenario IDs from data/scenarios.json")
    parser.add_argument("--all", action="store_true", help="Run all synthetic scenarios in file order")
    args = parser.parse_args()

    scenarios = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))
    by_id = {scenario["id"]: scenario for scenario in scenarios}
    if args.all and args.scenario_ids:
        parser.error("use either --all or scenario IDs")
    scenario_ids = list(by_id) if args.all else args.scenario_ids
    if not scenario_ids:
        parser.error("provide scenario IDs or --all")
    unknown = [scenario_id for scenario_id in scenario_ids if scenario_id not in by_id]
    if unknown:
        parser.error(f"unknown scenario IDs: {', '.join(unknown)}")

    model = None

    for scenario_id in scenario_ids:
        scenario = by_id[scenario_id]
        setup = scenario["setup"]
        backend_class = KnowledgeBase
        model_kind = "real_model"
        if "fault_injection" in setup:
            if setup["fault_injection"] != {
                "tools": ["search_documents", "read_document"],
                "mode": "persistent_error",
            }:
                parser.error(f"unsupported fault injection for {scenario_id}")
            backend_class = PersistentToolFailure
            model_kind = "real_model_controlled_tool_failure"
        elif "first_search_override" in setup:
            if setup["first_search_override"] != {"status": "no_results", "hits": []}:
                parser.error(f"unsupported first search override for {scenario_id}")
            backend_class = FirstSearchEmpty
            model_kind = "real_model_controlled_first_empty"
        elif set(setup) - {"max_steps", "max_tool_calls"}:
            parser.error(f"unsupported setup for {scenario_id}")

        if scenario["question"].strip():
            if model is None:
                model = create_chat_model()
            knowledge_base = backend_class.from_json(ROOT / "data" / "documents.json")
            graph = build_agent_graph(model, knowledge_base)
            model_name = model.model_name
        else:
            graph = None
            model_name = None
        run_input = {
            "question": scenario["question"],
            "history": scenario["history"],
            "max_steps": setup.get("max_steps", 8),
            "max_tool_calls": setup.get("max_tool_calls", 7),
            "model": model_name,
            "setup": setup,
        }
        input_path = EXAMPLES / f"{scenario_id}-input.json"
        output_path = EXAMPLES / f"{scenario_id}-output.json"
        trace_path = EXAMPLES / f"{scenario_id}-trace.jsonl"
        input_path.write_text(json.dumps(run_input, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = run_agent(
            graph,
            run_input["question"],
            history=run_input["history"],
            max_steps=run_input["max_steps"],
            max_tool_calls=run_input["max_tool_calls"],
            model_kind=model_kind,
            trace_path=trace_path,
        )
        if result["trace_path"] is None and trace_path.exists():
            trace_path.unlink()
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{scenario_id}: {result['status']}, steps={result['decision_steps']}, tools={result['tool_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
