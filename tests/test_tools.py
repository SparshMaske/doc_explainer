from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from stratai.core.config import get_settings
from stratai.core.schemas import PageContent, SearchResult
from stratai.tools import scrape as scrape_mod
from stratai.tools import search as search_mod
from stratai.tools.cache import Cache
from stratai.tools.scrape import scrape, truncate_words
from stratai.tools.search import SearchError, parse_results, search

SEARX_PAYLOAD = {
    "query": "india pet food market",
    "results": [
        {"url": "https://www.example.com/report/", "title": "Market report", "content": "Size: X", "engine": "duckduckgo"},
        {"url": "https://example.com/report?utm_source=x", "title": "Dup", "content": "", "engines": ["bing"]},
        {"url": "https://example.org/file.pdf", "title": "PDF", "content": ""},
        {"url": "https://www.youtube.com/watch?v=abc", "title": "Video", "content": ""},
        {"url": "https://news.example.net/a", "title": "News", "content": "Snippet", "engines": ["brave", "bing"]},
        {"url": "not a url", "title": "Junk"},
    ],
}


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEARX_URL", "http://searx.test")
    return tmp_path


def mock_searx(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    calls: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    monkeypatch.setattr(
        search_mod, "_http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(recording))
    )
    return calls


def test_parse_results_dedupes_and_skips() -> None:
    results = parse_results(SEARX_PAYLOAD)

    assert [str(r.url) for r in results] == ["https://www.example.com/report/", "https://news.example.net/a"]
    assert results[0].engine == "duckduckgo"
    assert results[1].engine == "brave"
    assert results[0].snippet == "Size: X"


async def test_search_parses_mocked_searxng(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = mock_searx(monkeypatch, lambda req: httpx.Response(200, json=SEARX_PAYLOAD))

    results = await search("India  pet food market", max_results=1)

    assert len(results) == 1 and isinstance(results[0], SearchResult)
    assert calls[0].url.params["format"] == "json"
    assert calls[0].url.params["q"] == "India pet food market"
    assert str(calls[0].url).startswith("http://searx.test/search")


async def test_search_cache_hit_skips_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = mock_searx(monkeypatch, lambda req: httpx.Response(200, json=SEARX_PAYLOAD))
    first = await search("india pet food market")

    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network called on cache hit")

    mock_searx(monkeypatch, no_network)
    second = await search("India pet food market")

    assert second == first
    assert len(calls) == 1


async def test_search_raises_search_error_on_403(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_searx(monkeypatch, lambda req: httpx.Response(403))
    with pytest.raises(SearchError, match="json format"):
        await search("anything")


async def test_scrape_cache_hit_skips_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    page = PageContent(url="https://example.com/a", title="A", markdown="hello world", word_count=2)
    Cache("scrape").set("https://example.com/a", page.model_dump(mode="json"))

    async def no_crawl(url: str) -> None:
        raise AssertionError("browser called on cache hit")

    monkeypatch.setattr(scrape_mod, "_crawl", no_crawl)

    assert await scrape("https://www.example.com/a/") == page


class FakeCrawler:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result, self.error = result, error

    def __call__(self, config: Any = None) -> "FakeCrawler":
        return self

    async def __aenter__(self) -> "FakeCrawler":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def arun(self, url: str, config: Any = None) -> Any:
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize(
    "crawler",
    [
        FakeCrawler(result=SimpleNamespace(success=False, error_message="net::ERR_NAME_NOT_RESOLVED", status_code=None)),
        FakeCrawler(error=RuntimeError("browser crashed")),
        FakeCrawler(result=SimpleNamespace(success=True, markdown="# Error response\n\n404 Not Found", metadata={}, status_code=404)),
    ],
)
async def test_scrape_failure_returns_none(monkeypatch: pytest.MonkeyPatch, crawler: FakeCrawler, caplog) -> None:
    monkeypatch.setattr(scrape_mod, "AsyncWebCrawler", crawler)

    assert await scrape("https://unreachable.example") is None
    assert "scrape failed for https://unreachable.example" in caplog.text
    assert Cache("scrape").get("https://unreachable.example") is None


async def test_scrape_success_truncates_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    md = SimpleNamespace(fit_markdown="", raw_markdown="# Title\n\n" + "word " * 7000)
    result = SimpleNamespace(success=True, markdown=md, metadata={"title": "Big page"}, status_code=200)
    monkeypatch.setattr(scrape_mod, "AsyncWebCrawler", FakeCrawler(result=result))

    page = await scrape("https://example.com/big")

    assert page is not None and page.word_count == 6000 and page.title == "Big page"
    assert page.markdown.startswith("# Title")
    assert Cache("scrape").get("https://example.com/big") is not None


async def test_scrape_skips_pdf() -> None:
    assert await scrape("https://example.com/report.pdf") is None


def test_truncate_words_keeps_short_text() -> None:
    assert truncate_words("a  b\nc", max_words=5) == ("a  b\nc", 3)


@pytest.mark.live
async def test_live_search_and_scrape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # use the real .env / environment, not the fixture's fake SEARX_URL
    try:
        results = await search("India pet food market size 2025")
    except SearchError as exc:
        pytest.skip(f"SearXNG unavailable at {get_settings().SEARX_URL}: {exc}")
    assert results, "SearXNG returned no results"

    for result in results[:3]:
        page = await scrape(str(result.url))
        if page is not None:
            assert page.word_count > 0 and page.markdown
            return
    pytest.fail("none of the top 3 results could be scraped (is the Crawl4AI browser installed? run crawl4ai-setup)")
