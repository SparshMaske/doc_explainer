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

## Search setup

StratAI searches the web through a self-hosted [SearXNG](https://docs.searxng.org) and scrapes pages with [Crawl4AI](https://docs.crawl4ai.com).

```bash
# 1. Start SearXNG (Docker Desktop must be running); serves http://localhost:8888
docker compose -f docker/searxng/docker-compose.yml up -d

# 2. Point StratAI at it: in .env set
#    SEARX_URL=http://localhost:8888

# 3. Check SearXNG works: should print JSON with a non-empty "results" list
curl -s "http://localhost:8888/search?q=test&format=json" | head -c 500
#    HTTP 403 means the json format is not enabled; check docker/searxng/settings.yml
#    and restart: docker compose -f docker/searxng/docker-compose.yml restart

# 4. Install the browser Crawl4AI uses for scraping (one time)
uv run crawl4ai-setup

# 5. Run the live search + scrape test
uv run pytest -m live tests/test_tools.py
```

Search and scrape results are cached for 7 days in `data/cache/`; delete that folder to force fresh fetches.
Stop SearXNG with `docker compose -f docker/searxng/docker-compose.yml down`.

See `CLAUDE.md` for the pipeline design and project rules.
