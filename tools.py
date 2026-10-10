"""Live evidence collectors for academic, news and general-web research."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from html import unescape
from typing import Any

import requests


USER_AGENT = "ResearchMind/1.0 (research assistant; contact: local-app)"


def clean_text(value: Any, limit: int = 900) -> str:
    """Remove HTML tags, decode entities, collapse whitespace, and limit length."""
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def rebuild_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Turn OpenAlex's word-to-position mapping back into a readable abstract."""
    if not inverted:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for position in positions:
            words.append((position, word))
    return " ".join(word for _, word in sorted(words))


def search_openalex(query: str, days: int, limit: int) -> list[dict[str, Any]]:
    """Fetch recent scholarly works from the public OpenAlex API."""
    start = (date.today() - timedelta(days=days)).isoformat()
    headers = {"User-Agent": USER_AGENT}
    if os.getenv("OPENALEX_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENALEX_API_KEY']}"
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
        headers=headers,
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
                "title": clean_text(work.get("display_name"), 240),
                "url": url,
                "published": work.get("publication_date") or str(work.get("publication_year", "")),
                "publisher": clean_text(source.get("display_name") or "OpenAlex"),
                "authors": clean_text(", ".join(filter(None, authors)), 300),
                "summary": clean_text(rebuild_abstract(work.get("abstract_inverted_index")), 500),
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
                "title": clean_text(item.findtext("title"), 240),
                "url": clean_text(item.findtext("link"), 800),
                "published": clean_text(item.findtext("pubDate"), 80),
                "publisher": clean_text(source.text if source is not None else "Google News"),
                "authors": "",
                "summary": clean_text(item.findtext("description"), 500),
                "cited_by": 0,
            }
        )
    return records


def search_tavily(query: str, days: int, limit: int) -> list[dict[str, Any]]:
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
            "start_date": (date.today() - timedelta(days=days)).isoformat(),
            "end_date": date.today().isoformat(),
        },
        timeout=22,
    )
    response.raise_for_status()
    return [
        {
            "kind": "Web",
            "title": clean_text(item.get("title"), 240),
            "url": clean_text(item.get("url"), 800),
            "published": clean_text(item.get("published_date"), 80),
            "publisher": "Web source",
            "authors": "",
            "summary": clean_text(item.get("content"), 500),
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
    """Run the selected search tools and return records plus warnings."""
    providers = []
    if use_academic:
        providers.append(("OpenAlex", search_openalex))
    if use_news:
        providers.append(("Google News", search_news))

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    if use_web:
        if os.getenv("TAVILY_API_KEY"):
            providers.append(("Tavily", search_tavily))
        else:
            warnings.append("Tavily was skipped: add TAVILY_API_KEY to enable web search.")

    for name, search_function in providers:
        try:
            results = search_function(query, days, per_source)
            records.extend(results)
        except (requests.RequestException, ET.ParseError, ValueError):
            warnings.append(f"{name} could not be reached.")

    seen: set[str] = set()
    unique = []
    for item in records:
        # URL paths can be case-sensitive. Only lowercase the title fallback.
        identity = item.get("url") or (item.get("title") or "").lower()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        item = item.copy()
        item["id"] = f"S{len(unique) + 1}"
        unique.append(item)
    return unique, warnings


def format_evidence(records: list[dict[str, Any]]) -> str:
    """Turn source dictionaries into text the model can cite as [S1], [S2], etc."""
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
