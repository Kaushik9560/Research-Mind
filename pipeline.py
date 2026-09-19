"""Run the tutorial's four-step research flow with production upgrades."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Callable

from agents import (
    build_critic_chain,
    build_reader_agent,
    build_search_agent,
    build_writer_chain,
    invoke_chain,
    research_provider_plan,
)
from tools import collect_evidence, format_evidence


@dataclass(frozen=True)
class ResearchRequest:
    """Everything the four stages need to understand the user's request."""

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
    """Turn a natural-language question into terms source APIs match well."""
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
        terms.extend(
            token
            for token in re.findall(r"[A-Za-z0-9]+", region.lower())
            if token not in terms
        )
    return " ".join(terms[:12]) or question


def run_research_pipeline(
    request: ResearchRequest | str,
    on_progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Execute Search Agent -> Reader Agent -> Writer Chain -> Critic Chain.

    A plain topic string is accepted for compatibility with the original video.
    The Streamlit app passes a richer ``ResearchRequest`` instead.
    """
    if isinstance(request, str):
        request = ResearchRequest(question=request)

    state: dict[str, object] = {"request": asdict(request)}
    profile = request.to_prompt()
    model_plan = research_provider_plan(request.depth)
    state["model_plan"] = model_plan

    # Step 1 - Search Agent: create the search plan, then run live source tools.
    _notify(on_progress, "search", "running")
    search_agent = build_search_agent(model_plan["primary"])
    state["search_plan"] = invoke_chain(search_agent, {"profile": profile})
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
    # This familiar key makes comparison with the tutorial pipeline immediate.
    state["search_results"] = evidence_text
    _notify(on_progress, "search", "done")

    # Step 2 - Reader Agent: read all records instead of one fragile scraped URL.
    _notify(on_progress, "reader", "running")
    reader_agent = build_reader_agent(model_plan["primary"])
    state["reader_notes"] = invoke_chain(
        reader_agent,
        {
            "profile": profile,
            "search_plan": str(state["search_plan"])[:3000],
            "evidence": evidence_text[:12000],
        },
    )
    state["scraped_content"] = state["reader_notes"]
    _notify(on_progress, "reader", "done")

    # Step 3 - Writer Chain: same prompt | llm | parser pattern as the tutorial.
    _notify(on_progress, "writer", "running")
    writer_chain = build_writer_chain(model_plan["primary"])
    state["report"] = invoke_chain(
        writer_chain,
        {
            "profile": profile,
            "research": str(state["reader_notes"])[:6000],
            "evidence": evidence_text[:9000],
            "language": request.language,
        },
    )
    _notify(on_progress, "writer", "done")

    # Step 4 - Critic Chain: use another provider when both keys are available.
    _notify(on_progress, "critic", "running")
    critic_chain = build_critic_chain(model_plan["reviewer"])
    state["feedback"] = invoke_chain(
        critic_chain,
        {
            "report": str(state["report"])[:7000],
            "evidence": evidence_text[:7000],
        },
    )
    _notify(on_progress, "critic", "done")
    return state


def export_markdown(state: dict[str, object]) -> str:
    """Export the report, source library, and four-stage learning trail."""
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
            "# Four-stage audit trail",
            "",
            "## 1. Search Agent plan",
            str(state.get("search_plan", "")),
            "",
            "## 2. Reader Agent notes",
            str(state.get("reader_notes", "")),
            "",
            "## 3. Writer Chain report",
            str(state.get("report", "")),
            "",
            "## 4. Critic Chain feedback",
            str(state.get("feedback", "")),
            "",
            f"_Generated for: {request.get('question', 'Research request') if isinstance(request, dict) else 'Research request'}_",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    topic = input("Research topic: ").strip()
    result = run_research_pipeline(topic)
    print("\n", result["report"])
    print("\nCRITIC FEEDBACK\n", result["feedback"])
