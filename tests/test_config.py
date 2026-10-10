from pathlib import Path

import pytest

from stratai.core.config import Settings

MODEL_FIELDS = [
    "SCOPING",
    "ISSUE_TREE",
    "RESEARCH_FAST",
    "RESEARCH_SMART",
    "VERIFY",
    "SYNTHESIS",
    "EMBEDDING",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)


def test_settings_defaults(clean_env: None) -> None:
    s = Settings(_env_file=None)

    assert s.OLLAMA_BASE_URL == "http://localhost:11434/v1"
    assert s.MAX_CONTEXT == 8192
    assert s.MAX_CONCURRENCY == 1
    assert s.DATA_DIR == Path("data")
    assert s.EMBEDDING_MODEL == "nomic-embed-text"
    for prefix in MODEL_FIELDS:
        assert getattr(s, f"{prefix}_MODEL")
        assert getattr(s, f"{prefix}_PROVIDER") == "ollama"


def test_settings_env_override(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CONTEXT", "4096")
    monkeypatch.setenv("VERIFY_PROVIDER", "openai")

    s = Settings(_env_file=None)

    assert s.MAX_CONTEXT == 4096
    assert s.VERIFY_PROVIDER == "openai"
