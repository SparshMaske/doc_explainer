import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from tenacity import wait_none

from stratai.core.config import Settings
from stratai.core.llm import LLMClient, StructuredOutputError, split_thinking
from stratai.core.run_logger import RunLogger
from stratai.core.schemas import Source

VALID_SOURCE = json.dumps(
    {
        "url": "https://example.com",
        "title": "Example",
        "retrieved_at": "2026-10-10T12:00:00Z",
        "snippet": "An example page.",
    }
)


class FakeOpenAI:
    """Stands in for openai.OpenAI: returns (or raises) scripted replies and records requests."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply, model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, DATA_DIR=tmp_path, SCOPING_MODEL="qwen3:14b")


def make_client(settings: Settings, replies: list[Any]) -> tuple[LLMClient, FakeOpenAI]:
    client = LLMClient(settings=settings, run_logger=RunLogger(data_dir=settings.DATA_DIR))
    fake = FakeOpenAI(replies)
    client._client_for = lambda provider: fake  # type: ignore[method-assign]
    return client, fake


def read_logs(client: LLMClient) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((client.run_logger.run_dir / "llm").glob("*.json"))]


def test_structured_retries_after_invalid_json(settings: Settings) -> None:
    client, fake = make_client(settings, ['{"url": "not a url", "title": 3', VALID_SOURCE])

    result = client.structured(agent="scoping", system="Extract the source.", prompt="...", schema=Source)

    assert isinstance(result, Source)
    assert str(result.url).startswith("https://example.com")
    assert len(fake.requests) == 2
    feedback = fake.requests[1]["messages"][-1]["content"]
    assert "failed validation" in feedback and "Fix the JSON" in feedback
    assert fake.requests[0]["response_format"] == {"type": "json_object"}


def test_structured_raises_after_three_failures(settings: Settings) -> None:
    client, fake = make_client(settings, ["nope", "still nope", "{}"])

    with pytest.raises(StructuredOutputError) as exc_info:
        client.structured(agent="scoping", system="s", prompt="p", schema=Source)

    assert exc_info.value.raw_output == "{}"
    assert len(fake.requests) == 3


def test_think_block_is_stripped_and_logged(settings: Settings) -> None:
    client, _ = make_client(settings, ["<think>\nsecret reasoning\n</think>\n\nThe answer is 42."])

    out = client.complete(agent="scoping", system="s", prompt="p", think=True)

    assert out == "The answer is 42."
    assert "secret" not in out
    log = read_logs(client)[-1]
    assert log["thinking"] == "secret reasoning"
    assert log["success"] is True
    assert {"agent", "model", "latency_ms", "prompt_chars", "response_chars"} <= log.keys()


def test_no_think_appended_for_qwen3_only(settings: Settings) -> None:
    client, fake = make_client(settings, ["a", "b"])
    client.complete(agent="scoping", system="s", prompt="p")
    client.complete(agent="scoping", system="s", prompt="p", think=True)

    assert fake.requests[0]["messages"][-1]["content"].endswith("/no_think")
    assert "/no_think" not in fake.requests[1]["messages"][-1]["content"]

    other = Settings(_env_file=None, DATA_DIR=settings.DATA_DIR, SCOPING_MODEL="gpt-4o-mini")
    client2, fake2 = make_client(other, ["c"])
    client2.complete(agent="scoping", system="s", prompt="p")
    assert "/no_think" not in fake2.requests[0]["messages"][-1]["content"]


@pytest.mark.parametrize(
    ("raw", "answer", "thinking"),
    [
        ("plain", "plain", ""),
        ("<think></think>\n\nhi", "hi", ""),
        ("dangling thought</think>hi", "hi", "dangling thought"),
        ("hi <think>truncated", "hi", "truncated"),
    ],
)
def test_split_thinking_edge_cases(raw: str, answer: str, thinking: str) -> None:
    assert split_thinking(raw) == (answer, thinking)


def test_network_errors_are_retried(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LLMClient._send.retry, "wait", wait_none())
    request = httpx.Request("POST", "http://localhost")
    client, fake = make_client(
        settings, [openai.APIConnectionError(request=request), httpx.ConnectError("down"), "ok"]
    )

    assert client.complete(agent="scoping", system="s", prompt="p") == "ok"
    assert len(fake.requests) == 3


def test_unknown_agent_raises(settings: Settings) -> None:
    client, _ = make_client(settings, [])
    with pytest.raises(ValueError, match="Unknown agent"):
        client.complete(agent="nobody", system="s", prompt="p")


async def test_acomplete(settings: Settings) -> None:
    client, _ = make_client(settings, ["<think>x</think>async ok"])
    assert await client.acomplete(agent="scoping", system="s", prompt="p") == "async ok"


@pytest.mark.live
def test_live_qwen3_8b_returns_source(tmp_path) -> None:
    settings = Settings(DATA_DIR=tmp_path, RESEARCH_FAST_MODEL="qwen3:8b", RESEARCH_FAST_PROVIDER="ollama")
    client = LLMClient(settings=settings, run_logger=RunLogger(data_dir=tmp_path))
    try:
        models = client.list_models("research_fast")
    except Exception as exc:  # noqa: BLE001 - any failure here means Ollama is not usable
        pytest.skip(f"Ollama unreachable: {exc}")
    if not any(m.startswith("qwen3:8b") for m in models):
        pytest.skip("qwen3:8b not pulled in Ollama")

    result = client.structured(
        agent="research_fast",
        system="You extract citation metadata.",
        prompt=(
            "Create a Source for the Python homepage: url https://www.python.org, "
            "title 'Welcome to Python.org', snippet 'The official home of the Python Programming Language.'"
        ),
        schema=Source,
    )

    assert isinstance(result, Source)
    assert "python.org" in str(result.url)
