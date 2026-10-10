# StratAI

Local demo of an AI consulting research assistant that mimics a McKinsey/BCG engagement.
Runs entirely on a MacBook M2 (16GB) via Ollama: `qwen3:14b` (reasoning), `qwen3:8b` (bulk), `nomic-embed-text` (embeddings).

## Pipeline (5 stages)

1. **Scope**: turn the client's question into a precise, bounded problem statement.
2. **Issue tree**: build a MECE issue tree with a hypothesis on each branch.
3. **Research**: answer every leaf question with cited web sources (SearXNG + Crawl4AI / GPT Researcher).
4. **Verify**: re-check key facts and numbers against their sources.
5. **Synthesize**: answer-first report (Pyramid Principle, SCQ) and slide deck.

## Folders

| Path | Purpose |
|------|---------|
| `stratai/core/` | Config, schemas, LLM client, run logger: shared infrastructure |
| `stratai/agents/` | One agent per pipeline stage |
| `stratai/tools/` | Search, crawl, and calculation helpers the agents call |
| `stratai/graph/` | LangGraph wiring of the pipeline stages |
| `stratai/outputs/` | Report and slide-deck renderers |
| `stratai/app/` | User interface (Streamlit) |
| `tests/` | pytest suite |
| `data/cache/` | Cached search/crawl results (gitignored) |
| `data/runs/` | Per-run JSON artifacts written by `RunLogger` (gitignored) |

## Permanent rules

- Every agent input and output is a pydantic model defined in `stratai/core/schemas.py`.
- The LLM never does arithmetic; all calculations happen in Python.
- Every factual claim carries at least one source URL.
- All LLM calls go through `stratai/core/llm.py`.
- Every agent saves its output via `run_logger` (`RunLogger.save(stage, name, payload)`).

## Constraints

- Only ONE LLM call may run at a time (`MAX_CONCURRENCY=1`) because of the 16GB memory limit.
- Configuration comes from `.env` via `stratai/core/config.py`; never hard-code model names.

## Commands

- `uv sync`: install dependencies
- `uv run pytest`: run tests
