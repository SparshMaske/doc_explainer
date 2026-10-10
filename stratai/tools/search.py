"""Web search via self-hosted SearXNG (JSON API), cached by query."""

import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from stratai.core.config import get_settings
from stratai.core.schemas import SearchResult
from stratai.tools.cache import Cache

logger = logging.getLogger(__name__)

TIMEOUT_S = 20.0

_SKIP_EXTENSIONS = (".pdf", ".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v")
_VIDEO_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "tiktok.com", "twitch.tv")
_TRACKING_PARAMS = re.compile(r"^(utm_\w+|fbclid|gclid|ref|ref_src)$")


class SearchError(Exception):
    """SearXNG could not be reached or returned an unusable response."""


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT_S)


def normalize_url(url: str) -> str:
    """Canonical form used only for deduplication; the original URL is what we keep."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)])
    return urlunsplit(("https" if parts.scheme in ("http", "https") else parts.scheme, host, parts.path.rstrip("/"), query, ""))


def is_skipped(url: str, category: str = "") -> bool:
    """PDFs and video links are out of scope for this demo."""
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.").removeprefix("m.")
    return (
        category in ("videos", "files")
        or parts.path.lower().endswith(_SKIP_EXTENSIONS)
        or any(host == h or host.endswith("." + h) for h in _VIDEO_HOSTS)
    )


def parse_results(payload: dict[str, Any]) -> list[SearchResult]:
    """Turn a SearXNG JSON payload into deduplicated SearchResults, preserving rank order."""
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in payload.get("results", []):
        url = (item.get("url") or "").strip()
        if not url or is_skipped(url, item.get("category", "")):
            continue
        key = normalize_url(url)
        if key in seen:
            continue
        engine = item.get("engine") or next(iter(item.get("engines") or []), "")
        try:
            result = SearchResult(
                url=url,
                title=(item.get("title") or url).strip(),
                snippet=(item.get("content") or "").strip(),
                engine=engine,
            )
        except ValidationError:
            logger.debug("Skipping result with invalid URL: %s", url)
            continue
        seen.add(key)
        results.append(result)
    return results


async def search(query: str, max_results: int = 8) -> list[SearchResult]:
    """Search SearXNG. Raises SearchError if SearXNG is unreachable or misconfigured."""
    query = " ".join(query.split())
    cache = Cache("search")
    cache_key = query.lower()
    cached = cache.get(cache_key)
    if cached is not None:
        return [SearchResult.model_validate(r) for r in cached][:max_results]

    url = get_settings().SEARX_URL.rstrip("/") + "/search"
    try:
        async with _http_client() as client:
            response = await client.get(url, params={"q": query, "format": "json"})
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        hint = " (is the json format enabled in settings.yml?)" if exc.response.status_code == 403 else ""
        raise SearchError(f"SearXNG returned HTTP {exc.response.status_code} for {url}{hint}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchError(f"SearXNG request to {url} failed: {type(exc).__name__}: {exc}") from exc

    results = parse_results(payload)
    if results:  # don't cache empty answers; they are often transient engine failures
        cache.set(cache_key, [r.model_dump(mode="json") for r in results])
    else:
        logger.warning("SearXNG returned no usable results for %r", query)
    return results[:max_results]
