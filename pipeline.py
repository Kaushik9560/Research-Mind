"""Orchestrates InsightForge's strategy, evidence and synthesis agents."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Callable

from agents import (
    analyze_trends,
    build_research_strategy,
    challenge_analysis,
    synthesize_report,
)
from tools import collect_evidence, format_evidence


@dataclass(frozen=True)
class ResearchRequest:
    question: str
    domain: str = "General"
    days: int = 365
    period_label: str = "Past year"
    region: str = "Global"
    audience: str = "General reader"
    depth: str = "Standard"
    language: str = "English"
    use_academic: bool = True
    use_news: bool = True
    use_web: bool = False

    def to_prompt(self) -> str:
        return "\n".join(
            [
                f"Today's date: {date.today().isoformat()}",
                f"Core question: {self.question}",
                f"Domain: {self.domain}",
                f"Evidence window: {self.period_label} ({self.days} days)",
                f"Geographic focus: {self.region}",
                f"Audience: {self.audience}",
                f"Requested depth: {self.depth}",
                f"Output language: {self.language}",
            ]
        )


ProgressCallback = Callable[[str, str], None]


def _notify(callback: ProgressCallback | None, step: str, status: str) -> None:
    if callback:
        callback(step, status)


def _source_limit(depth: str) -> int:
    return {"Quick scan": 4, "Standard": 6, "Deep dive": 8}.get(depth, 6)


def _build_search_query(question: str, domain: str, region: str) -> str:
    """Convert prose questions into terms that source indexes match reliably."""
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "latest", "of", "on", "recent", "the",
        "to", "trend", "trends", "what", "which", "with", "changing", "advances",
    }
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", question.lower()):
        if token not in stopwords and (len(token) > 2 or token == "ai") and token not in terms:
            terms.append(token)
    if len(terms) < 3 and domain.lower() != "general":
        for token in re.findall(r"[A-Za-z0-9]+", domain.lower()):
            if token not in stopwords and token not in terms:
                terms.append(token)
    if region.lower() != "global":
        terms.extend(token for token in re.findall(r"[A-Za-z0-9]+", region.lower()) if token not in terms)
    return " ".join(terms[:12]) or question


def run_research_pipeline(
    request: ResearchRequest,
    on_progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the complete, inspectable multi-agent research workflow."""
    state: dict[str, object] = {"request": asdict(request)}
    profile = request.to_prompt()

    _notify(on_progress, "strategist", "running")
    state["strategy"] = build_research_strategy(profile)
    _notify(on_progress, "strategist", "done")

    _notify(on_progress, "collector", "running")
    query = _build_search_query(request.question, request.domain, request.region)
    state["search_query"] = query
    evidence, warnings = collect_evidence(
        query=query,
        days=request.days,
        per_source=_source_limit(request.depth),
        use_academic=request.use_academic,
        use_news=request.use_news,
        use_web=request.use_web,
    )
    state["evidence"] = evidence
    state["warnings"] = warnings
    evidence_text = format_evidence(evidence)
    _notify(on_progress, "collector", "done")

    _notify(on_progress, "analyst", "running")
    state["trends"] = analyze_trends(
        profile,
        str(state["strategy"]),
        evidence_text,
    )
    _notify(on_progress, "analyst", "done")

    _notify(on_progress, "skeptic", "running")
    state["challenge"] = challenge_analysis(
        profile,
        str(state["trends"]),
        evidence_text,
    )
    _notify(on_progress, "skeptic", "done")

    _notify(on_progress, "writer", "running")
    state["report"] = synthesize_report(
        profile,
        str(state["strategy"]),
        str(state["trends"]),
        str(state["challenge"]),
        evidence_text,
    )
    _notify(on_progress, "writer", "done")
    return state


def export_markdown(state: dict[str, object]) -> str:
    """Build a portable report that retains its source mapping and audit trail."""
    request = state.get("request", {})
    evidence = state.get("evidence", [])
    lines = [str(state.get("report", "")), "", "---", "", "# Evidence library", ""]
    for item in evidence if isinstance(evidence, list) else []:
        lines.extend(
            [
                f"## [{item['id']}] {item['title']}",
                f"- Type: {item['kind']}",
                f"- Published: {item.get('published') or 'Unknown'}",
                f"- Publisher: {item.get('publisher') or 'Unknown'}",
                f"- URL: {item.get('url') or 'Unavailable'}",
                f"- Excerpt: {item.get('summary') or 'Not available'}",
                "",
            ]
        )
    lines.extend(
        [
            "---",
            "",
            "# Research audit trail",
            "",
            "## Strategy",
            str(state.get("strategy", "")),
            "",
            "## Trend analysis",
            str(state.get("trends", "")),
            "",
            "## Adversarial review",
            str(state.get("challenge", "")),
            "",
            f"_Generated for: {request.get('question', 'Research request') if isinstance(request, dict) else 'Research request'}_",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    question = input("Research question: ").strip()
    result = run_research_pipeline(ResearchRequest(question=question))
    print("\n", result["report"])
