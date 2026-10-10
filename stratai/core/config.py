"""Application settings, loaded from environment variables and `.env`."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"

    SCOPING_MODEL: str = "qwen3:14b"
    SCOPING_PROVIDER: str = "ollama"
    ISSUE_TREE_MODEL: str = "qwen3:14b"
    ISSUE_TREE_PROVIDER: str = "ollama"
    RESEARCH_FAST_MODEL: str = "qwen3:8b"
    RESEARCH_FAST_PROVIDER: str = "ollama"
    RESEARCH_SMART_MODEL: str = "qwen3:14b"
    RESEARCH_SMART_PROVIDER: str = "ollama"
    VERIFY_MODEL: str = "qwen3:14b"
    VERIFY_PROVIDER: str = "ollama"
    SYNTHESIS_MODEL: str = "qwen3:14b"
    SYNTHESIS_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_PROVIDER: str = "ollama"

    MAX_CONTEXT: int = 8192
    MAX_CONCURRENCY: int = 1

    SEARX_URL: str = "http://localhost:8888"
    DATA_DIR: Path = Path("data")


def get_settings() -> Settings:
    return Settings()
