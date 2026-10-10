"""Fetch a web page as clean markdown via Crawl4AI, cached by URL. Never raises."""

import logging
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilterLXML

from stratai.core.schemas import PageContent
from stratai.tools.cache import Cache
from stratai.tools.search import is_skipped, normalize_url

logger = logging.getLogger(__name__)

MAX_WORDS = 6000
MIN_FIT_WORDS = 50  # below this the pruning filter probably removed the article; fall back to raw
PAGE_TIMEOUT_MS = 30_000

_WORD = re.compile(r"\S+")


def truncate_words(text: str, max_words: int = MAX_WORDS) -> tuple[str, int]:
    """Cut text after `max_words` words, keeping the markdown formatting before the cut."""
    words = list(_WORD.finditer(text))
    if len(words) <= max_words:
        return text, len(words)
    return text[: words[max_words - 1].end()], max_words


def _pick_markdown(md: object) -> str:
    fit = getattr(md, "fit_markdown", None) or ""
    raw = getattr(md, "raw_markdown", None) or str(md or "")
    return fit if len(_WORD.findall(fit)) >= MIN_FIT_WORDS else raw


async def _crawl(url: str) -> PageContent | None:
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # our own diskcache handles caching
        page_timeout=PAGE_TIMEOUT_MS,
        excluded_tags=["nav", "header", "footer", "aside", "form", "script", "style"],
        markdown_generator=DefaultMarkdownGenerator(content_filter=PruningContentFilterLXML()),
        verbose=False,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    if not result.success:
        logger.warning("scrape failed for %s: %s (status %s)", url, result.error_message, result.status_code)
        return None
    if (result.status_code or 0) >= 400:  # Crawl4AI reports error pages as successful crawls
        logger.warning("scrape failed for %s: HTTP %s", url, result.status_code)
        return None
    markdown, word_count = truncate_words(_pick_markdown(result.markdown).strip())
    if word_count == 0:
        logger.warning("scrape failed for %s: page has no text content", url)
        return None
    title = ((result.metadata or {}).get("title") or url).strip()
    return PageContent(url=url, title=title, markdown=markdown, word_count=word_count)


async def scrape(url: str) -> PageContent | None:
    """Return the page as markdown (max 6000 words), or None on any failure. Never raises."""
    try:
        if is_skipped(url):
            logger.info("scrape skipped for %s: PDFs and videos are out of scope", url)
            return None
        cache = Cache("scrape")
        key = normalize_url(url)
        cached = cache.get(key)
        if cached is not None:
            return PageContent.model_validate(cached)
        page = await _crawl(url)
        if page is not None:
            cache.set(key, page.model_dump(mode="json"))
        return page
    except Exception as exc:  # noqa: BLE001 - contract: scraping never raises
        logger.warning("scrape failed for %s: %s: %s", url, type(exc).__name__, exc)
        return None
