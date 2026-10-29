"""Model configuration. Credentials stay outside the repository."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


class ModelConfigurationError(ValueError):
    """Required model settings are absent."""


DEFAULT_MODEL = "deepseek-v4.1-flash-1017"


def create_chat_model(env_file: str | Path | None = None) -> ChatOpenAI:
    """Create a tool-capable chat client after local credentials are supplied.

    The selected endpoint must be checked with an actual tool-call smoke test.
    Creating the client alone does not prove provider compatibility.
    """
    if env_file is not None:
        load_dotenv(dotenv_path=Path(env_file), override=False)
    else:
        project_env = Path(__file__).resolve().parents[2] / ".env"
        if project_env.exists():
            load_dotenv(dotenv_path=project_env, override=False)

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model_name = os.getenv("DASHSCOPE_MODEL", "").strip() or DEFAULT_MODEL
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise ModelConfigurationError(
            "Set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL in the local environment or .env file"
        )
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        timeout=30,
        temperature=0,
    )
