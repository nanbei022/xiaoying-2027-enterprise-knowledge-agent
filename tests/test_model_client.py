"""Model configuration tests never contact a provider."""

import pytest

from knowledge_agent import model_client
from knowledge_agent.model_client import ModelConfigurationError, create_chat_model


def test_missing_model_settings_fail_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("DASHSCOPE_MODEL", "")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "")
    with pytest.raises(ModelConfigurationError):
        create_chat_model(env_file=tmp_path / "missing.env")


def test_local_endpoint_and_model_are_used_without_network(monkeypatch, tmp_path):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(model_client, "ChatOpenAI", fake_client)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("DASHSCOPE_MODEL", "")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/compatible-mode/v1")

    create_chat_model(env_file=tmp_path / "missing.env")
    assert captured["model"] == "deepseek-v4.1-flash-1017"
    assert captured["base_url"] == "https://example.invalid/compatible-mode/v1"
    assert captured["api_key"] == "test-only-placeholder"
    assert captured["max_retries"] == 0
    assert captured["temperature"] == 0
