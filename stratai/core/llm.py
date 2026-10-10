"""The single LLM gateway. Every LLM call in StratAI goes through LLMClient.

- The agent name picks model and provider from Settings: agent="scoping" -> SCOPING_MODEL / SCOPING_PROVIDER.
- Qwen3 thinking is disabled with "/no_think" unless think=True; <think> blocks are stripped from
  responses and saved to the run log, never returned.
- At most MAX_CONCURRENCY calls run at once, process-wide, across sync and async callers.
- Every model call is logged via RunLogger under stage "llm".
"""

import asyncio
import json
import os
import re
import threading
import time
import weakref
from typing import Any, TypeVar

import httpx
import openai
from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stratai.core.config import Settings, get_settings
from stratai.core.run_logger import RunLogger

T = TypeVar("T", bound=BaseModel)

MAX_STRUCTURED_ATTEMPTS = 3
REQUEST_TIMEOUT_S = 600.0  # a 14B model thinking on an M2 can take minutes
NO_THINK = "/no_think"

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

# OpenAI's APITimeoutError subclasses APIConnectionError; httpx errors cover transport-level failures.
_NETWORK_ERRORS = (openai.APIConnectionError, httpx.ConnectError, httpx.TimeoutException)

# Process-wide gate: the memory limit applies to the machine, not to one client instance.
_call_gate: threading.BoundedSemaphore | None = None
_call_gate_lock = threading.Lock()


def _get_call_gate(limit: int) -> threading.BoundedSemaphore:
    global _call_gate
    with _call_gate_lock:
        if _call_gate is None:
            _call_gate = threading.BoundedSemaphore(max(1, limit))
        return _call_gate


class StructuredOutputError(Exception):
    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def split_thinking(text: str) -> tuple[str, str]:
    """Return (answer, thinking) with all <think>...</think> content removed from the answer."""
    thoughts = [t.strip() for t in _THINK_BLOCK.findall(text)]
    answer = _THINK_BLOCK.sub("", text)
    # Unbalanced tags: a dangling "</think>" (opening tag was in the template) or a truncated "<think>".
    if "</think>" in answer:
        head, _, answer = answer.rpartition("</think>")
        thoughts.append(head.replace("<think>", "").strip())
    if "<think>" in answer:
        answer, _, tail = answer.partition("<think>")
        thoughts.append(tail.strip())
    return answer.strip(), "\n\n".join(t for t in thoughts if t)


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply that may be fenced or padded with prose."""
    text = text.strip()
    fenced = _JSON_FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


class LLMClient:
    def __init__(self, settings: Settings | None = None, run_logger: RunLogger | None = None) -> None:
        self.settings = settings or get_settings()
        self.run_logger = run_logger or RunLogger(data_dir=self.settings.DATA_DIR)
        self._gate = _get_call_gate(self.settings.MAX_CONCURRENCY)
        # asyncio primitives bind to one event loop, so keep one semaphore per loop (Streamlit reruns,
        # repeated asyncio.run calls). The thread gate below still enforces the process-wide limit.
        self._async_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self._clients: dict[str, OpenAI] = {}
        self._seq = 0
        self._seq_lock = threading.Lock()

    # ---- public API -------------------------------------------------------------------------

    def complete(
        self, agent: str, system: str, prompt: str, temperature: float = 0.2, think: bool = False
    ) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return self._call(agent, messages, temperature=temperature, think=think)

    def structured(
        self, agent: str, system: str, prompt: str, schema: type[T], think: bool = False
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_msg = (
            f"{system}\n\nRespond with a single JSON object only, no prose and no markdown. "
            f"It must validate against this JSON schema:\n{schema_json}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        raw = ""
        error = ""
        for attempt in range(1, MAX_STRUCTURED_ATTEMPTS + 1):
            raw = self._call(
                agent,
                messages,
                temperature=0.0,
                think=think,
                # JSON mode constrains output from the first token, which blocks qwen3's <think> block;
                # with thinking on, rely on the schema prompt plus the validate-and-retry loop instead.
                json_mode=not think,
                attempt=attempt,
            )
            try:
                return schema.model_validate_json(extract_json(raw))
            except ValidationError as exc:
                error = str(exc)
                self._log_validation_failure(agent, attempt, raw, error)
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your JSON failed validation:\n{error}\n\n"
                            "Fix the JSON so it validates against the schema. "
                            "Return only the corrected JSON object."
                        ),
                    },
                ]
        raise StructuredOutputError(
            f"{agent}: no valid {schema.__name__} after {MAX_STRUCTURED_ATTEMPTS} attempts: {error}",
            raw_output=raw,
        )

    async def acomplete(
        self, agent: str, system: str, prompt: str, temperature: float = 0.2, think: bool = False
    ) -> str:
        async with self._async_gate():
            return await asyncio.to_thread(self.complete, agent, system, prompt, temperature, think)

    async def astructured(
        self, agent: str, system: str, prompt: str, schema: type[T], think: bool = False
    ) -> T:
        async with self._async_gate():
            return await asyncio.to_thread(self.structured, agent, system, prompt, schema, think)

    def list_models(self, agent: str) -> list[str]:
        """Model IDs the agent's provider serves. Raises on network errors (useful as a health check)."""
        _, provider = self.resolve(agent)
        return [m.id for m in self._client_for(provider).models.list().data]

    def resolve(self, agent: str) -> tuple[str, str]:
        """Map an agent name to (model, provider) via Settings."""
        key = agent.strip().upper()
        model = getattr(self.settings, f"{key}_MODEL", None)
        provider = getattr(self.settings, f"{key}_PROVIDER", None)
        if model is None or provider is None:
            raise ValueError(f"Unknown agent {agent!r}: Settings has no {key}_MODEL / {key}_PROVIDER")
        return model, provider.lower()

    # ---- internals --------------------------------------------------------------------------

    def _async_gate(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        gate = self._async_gates.get(loop)
        if gate is None:
            gate = self._async_gates[loop] = asyncio.Semaphore(max(1, self.settings.MAX_CONCURRENCY))
        return gate

    def _client_for(self, provider: str) -> OpenAI:
        if provider not in self._clients:
            if provider == "ollama":
                base_url = self.settings.OLLAMA_BASE_URL.rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url += "/v1"
                api_key: str | None = "ollama"
            else:
                # Any OpenAI-compatible hosted API: <PROVIDER>_API_KEY and optional <PROVIDER>_BASE_URL.
                env = {**dotenv_values(".env"), **os.environ}
                prefix = provider.upper()
                base_url = env.get(f"{prefix}_BASE_URL") or None
                api_key = env.get(f"{prefix}_API_KEY")
            # max_retries=0: tenacity is the only retry layer.
            self._clients[provider] = OpenAI(
                base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=0
            )
        return self._clients[provider]

    @retry(
        retry=retry_if_exception_type(_NETWORK_ERRORS),
        stop=stop_after_attempt(4),  # 1 try + 3 retries
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _send(self, client: OpenAI, **kwargs: Any) -> Any:
        return client.chat.completions.create(**kwargs)

    def _call(
        self,
        agent: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        think: bool,
        json_mode: bool = False,
        attempt: int = 1,
    ) -> str:
        model, provider = self.resolve(agent)
        messages = [dict(m) for m in messages]
        if not think and model.lower().startswith("qwen3"):
            last_user = next(m for m in reversed(messages) if m["role"] == "user")
            last_user["content"] = f"{last_user['content']}\n{NO_THINK}"

        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        prompt_chars = sum(len(m["content"]) for m in messages)
        record: dict[str, Any] = {
            "agent": agent,
            "model": model,
            "provider": provider,
            "attempt": attempt,
            "json_mode": json_mode,
            "think": think,
            "prompt_chars": prompt_chars,
        }
        started = time.perf_counter()
        try:
            with self._gate:
                response = self._send(self._client_for(provider), **kwargs)
        except Exception as exc:
            record.update(
                success=False,
                latency_ms=round((time.perf_counter() - started) * 1000),
                response_chars=0,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._log(agent, record)
            raise

        message = response.choices[0].message
        answer, thinking = split_thinking(message.content or "")
        # Newer Ollama versions return thinking in a separate "reasoning" field.
        extra = getattr(message, "model_extra", None) or {}
        separate = extra.get("reasoning") or extra.get("reasoning_content") or ""
        if separate:
            thinking = f"{separate}\n\n{thinking}".strip()

        record.update(
            success=True,
            latency_ms=round((time.perf_counter() - started) * 1000),
            response_chars=len(answer),
            thinking=thinking,
            response=answer,
        )
        self._log(agent, record)
        return answer

    def _log_validation_failure(self, agent: str, attempt: int, raw: str, error: str) -> None:
        self._log(
            agent,
            {
                "agent": agent,
                "attempt": attempt,
                "success": False,
                "event": "structured_validation_failed",
                "error": error,
                "raw_output": raw,
            },
        )

    def _log(self, agent: str, payload: dict[str, Any]) -> None:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        self.run_logger.save("llm", f"{seq:04d}_{agent}", payload)
