"""Live evidence collectors for academic, news and general-web research."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import requests


USER_AGENT = "InsightForge/1.0 (research assistant; contact: local-app)"


def _clean(value: Any, limit: int = 900) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _rebuild_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(words))


def search_openalex(query: str, days: int, limit: int) -> list[dict[str, Any]]:
    """Fetch recent scholarly works from the public OpenAlex API."""
    start = (date.today() - timedelta(days=days)).isoformat()
    response = requests.get(
        "https://api.openalex.org/works",
        params={
            "search": query,
            "filter": f"from_publication_date:{start}",
            # The date filter guarantees recency; relevance-first avoids a newly
            # published but only tangential paper outranking a strong match.
            "sort": "relevance_score:desc",
            "per-page": min(limit, 25),
        },
        headers={"User-Agent": USER_AGENT},
        timeout=18,
    )
    response.raise_for_status()
    records = []
    for work in response.json().get("results", []):
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        authors = [
            authorship.get("author", {}).get("display_name", "")
            for authorship in work.get("authorships", [])[:5]
        ]
        url = work.get("doi") or location.get("landing_page_url") or work.get("id")
        records.append(
            {
                "kind": "Academic",
                "title": _clean(work.get("display_name"), 240),
                "url": url,
                "published": work.get("publication_date") or str(work.get("publication_year", "")),
                "publisher": _clean(source.get("display_name") or "OpenAlex"),
                "authors": _clean(", ".join(filter(None, authors)), 300),
                "summary": _clean(_rebuild_abstract(work.get("abstract_inverted_index")), 500),
                "cited_by": int(work.get("cited_by_count") or 0),
            }
        )
    return records


def search_news(query: str, days: int, limit: int) -> list[dict[str, Any]]:
    """Fetch recent reporting from Google News RSS without requiring an API key."""
    period = max(1, min(days, 3650))
    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": f"{query} when:{period}d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
        headers={"User-Agent": USER_AGENT},
        timeout=18,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    records = []
    for item in root.findall(".//item")[:limit]:
        source = item.find("source")
        records.append(
            {
                "kind": "News",
                "title": _clean(item.findtext("title"), 240),
                "url": _clean(item.findtext("link"), 800),
                "published": _clean(item.findtext("pubDate"), 80),
                "publisher": _clean(source.text if source is not None else "Google News"),
                "authors": "",
                "summary": _clean(item.findtext("description"), 500),
                "cited_by": 0,
            }
        )
    return records


def search_tavily(query: str, limit: int) -> list[dict[str, Any]]:
    """Use Tavily as an optional general-web layer when its key is present."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": min(limit, 10),
            "search_depth": "advanced",
            "include_answer": False,
        },
        timeout=22,
    )
    response.raise_for_status()
    return [
        {
            "kind": "Web",
            "title": _clean(item.get("title"), 240),
            "url": _clean(item.get("url"), 800),
            "published": _clean(item.get("published_date"), 80),
            "publisher": "Web source",
            "authors": "",
            "summary": _clean(item.get("content"), 500),
            "cited_by": 0,
        }
        for item in response.json().get("results", [])
    ]


def collect_evidence(
    query: str,
    days: int,
    per_source: int,
    use_academic: bool = True,
    use_news: bool = True,
    use_web: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run selected collectors concurrently and return records plus warnings."""
    jobs = []
    if use_academic:
        jobs.append(("OpenAlex", search_openalex, (query, days, per_source)))
    if use_news:
        jobs.append(("Google News", search_news, (query, days, per_source)))
    if use_web and os.getenv("TAVILY_API_KEY"):
        jobs.append(("Tavily", search_tavily, (query, per_source)))

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = {pool.submit(func, *args): name for name, func, args in jobs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                records.extend(future.result())
            except (requests.RequestException, ET.ParseError, ValueError) as exc:
                warnings.append(f"{name} could not be reached ({exc.__class__.__name__}).")

    seen: set[str] = set()
    unique = []
    for item in records:
        identity = (item.get("url") or item.get("title") or "").lower()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        item["id"] = f"S{len(unique) + 1}"
        unique.append(item)
    return unique, warnings


def format_evidence(records: list[dict[str, Any]]) -> str:
    """Serialize source cards into a bounded, citation-friendly model context."""
    if not records:
        return (
            "NO LIVE EVIDENCE WAS RETRIEVED. Do not make factual trend claims. "
            "Explain that the search returned no records and propose a better query."
        )
    blocks = []
    for item in records:
        details = [
            f"[{item['id']}] {item['title']}",
            f"Type: {item['kind']}",
            f"Published: {item.get('published') or 'Unknown'}",
            f"Publisher: {item.get('publisher') or 'Unknown'}",
            f"Authors: {item.get('authors') or 'Not listed'}",
            f"URL: {item.get('url') or 'Unavailable'}",
        ]
        if item.get("cited_by"):
            details.append(f"OpenAlex cited-by count: {item['cited_by']}")
        details.append(f"Excerpt: {item.get('summary') or 'No abstract/snippet available.'}")
        blocks.append("\n".join(details))
    return "\n\n---\n\n".join(blocks)
