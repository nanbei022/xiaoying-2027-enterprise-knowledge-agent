"""CLI rejects malformed input before any model setup or network request."""

import json

from knowledge_agent.cli import main


def test_empty_question_precedes_missing_credential(monkeypatch, capsys):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    exit_code = main([""])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["status"] == "invalid_input"


def test_invalid_limit_precedes_missing_credential(monkeypatch, capsys):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    exit_code = main(["question", "--max-steps", "0"])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["status"] == "invalid_input"
