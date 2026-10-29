"""Run one question through the local corpus and a configured chat model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from .agent import RunInput, build_agent_graph, run_agent
from .knowledge_base import CorpusError, KnowledgeBase
from .model_client import ModelConfigurationError, create_chat_model


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise knowledge QA Agent")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--documents", type=Path, default=ROOT / "data" / "documents.json")
    parser.add_argument("--history-file", type=Path, help="Optional JSON array of prior user/assistant turns")
    parser.add_argument("--trace", type=Path, help="Save JSONL trace to this path")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    args = parser.parse_args(argv)

    history = None
    if args.history_file is not None:
        try:
            history = json.loads(args.history_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            print(json.dumps({"status": "invalid_input", "error": "cannot read history JSON"}, ensure_ascii=False))
            return 2

    try:
        RunInput.model_validate(
            {
                "question": args.question,
                "history": [] if history is None else history,
                "max_steps": args.max_steps,
                "max_tool_calls": args.max_tool_calls,
            }
        )
    except ValidationError:
        print(json.dumps({"status": "invalid_input", "error": "invalid run input"}, ensure_ascii=False))
        return 2

    try:
        knowledge_base = KnowledgeBase.from_json(args.documents)
        model = create_chat_model()
    except (CorpusError, ModelConfigurationError) as exc:
        print(json.dumps({"status": "configuration_error", "error": str(exc)}, ensure_ascii=False))
        return 2

    graph = build_agent_graph(model, knowledge_base)
    result = run_agent(
        graph,
        args.question,
        history=history,
        max_steps=args.max_steps,
        max_tool_calls=args.max_tool_calls,
        model_kind="real_model",
        trace_path=args.trace,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
