"""Fetch a web page as clean markdown via Crawl4AI, cached by URL. Never raises."""

import logging
import re
from contextlib import AsyncExitStack

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


def _run_config() -> CrawlerRunConfig:
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # our own diskcache handles caching
        page_timeout=PAGE_TIMEOUT_MS,
        excluded_tags=["nav", "header", "footer", "aside", "form", "script", "style"],
        markdown_generator=DefaultMarkdownGenerator(content_filter=PruningContentFilterLXML()),
        verbose=False,
    )


async def _crawl(url: str, crawler: AsyncWebCrawler) -> PageContent | None:
    result = await crawler.arun(url=url, config=_run_config())

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


async def scrape_many(urls: list[str]) -> list[PageContent | None]:
    """Scrape URLs one at a time through a single shared browser; results align with `urls`.

    The browser starts only on the first cache miss. Never raises: failures are logged and give None.
    """
    pages: list[PageContent | None] = []
    cache = Cache("scrape")
    async with AsyncExitStack() as stack:
        crawler: AsyncWebCrawler | None = None
        browser_failed = False
        for url in urls:
            page = None
            try:
                if is_skipped(url):
                    logger.info("scrape skipped for %s: PDFs and videos are out of scope", url)
                elif (cached := cache.get(normalize_url(url))) is not None:
                    page = PageContent.model_validate(cached)
                elif browser_failed:
                    logger.warning("scrape failed for %s: browser could not be started", url)
                else:
                    if crawler is None:
                        try:
                            crawler = await stack.enter_async_context(
                                AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False))
                            )
                        except Exception:
                            browser_failed = True
                            raise
                    page = await _crawl(url, crawler)
                    if page is not None:
                        cache.set(normalize_url(url), page.model_dump(mode="json"))
            except Exception as exc:  # noqa: BLE001 - contract: scraping never raises
                logger.warning("scrape failed for %s: %s: %s", url, type(exc).__name__, exc)
            pages.append(page)
    return pages


async def scrape(url: str) -> PageContent | None:
    """Return the page as markdown (max 6000 words), or None on any failure. Never raises."""
    try:
        return (await scrape_many([url]))[0]
    except Exception as exc:  # noqa: BLE001 - e.g. the browser failing to shut down cleanly
        logger.warning("scrape failed for %s: %s: %s", url, type(exc).__name__, exc)
        return None
