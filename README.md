# StratAI

A local AI consulting research assistant: scope → MECE issue tree → cited research → fact verification → answer-first report and deck.

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
# 1. Install dependencies
uv sync

# 2. Pull the local models
ollama pull qwen3:14b
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 3. Configure environment
cp .env.example .env

# 4. Run tests
uv run pytest
```

See `CLAUDE.md` for the pipeline design and project rules.
