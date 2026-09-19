"""Search Agent -> Reader Agent -> Writer Chain -> Critic Chain."""

import re
from datetime import date

from agents import (
    build_critic_chain,
    build_reader_agent,
    build_search_agent,
    build_writer_chain,
    create_models,
)
from tools import collect_evidence, format_evidence


def make_search_query(topic: str, domain: str, region: str) -> str:
    """Remove common question words before calling source APIs."""
    ignored_words = {
        "a", "an", "and", "are", "for", "from", "how", "in", "is", "latest",
        "of", "on", "recent", "the", "to", "trend", "trends", "what", "which",
        "with",
    }
    words = re.findall(r"[A-Za-z0-9]+", topic.lower())
    useful_words = [word for word in words if word not in ignored_words]

    if domain.lower() != "general":
        useful_words.extend(re.findall(r"[A-Za-z0-9]+", domain.lower()))
    if region.lower() != "global":
        useful_words.extend(re.findall(r"[A-Za-z0-9]+", region.lower()))

    return " ".join(dict.fromkeys(useful_words[:12])) or topic


def run_research_pipeline(
    topic: str,
    days: int = 365,
    period: str = "1 year",
    domain: str = "General",
    region: str = "Global",
    audience: str = "General reader",
    depth: str = "Standard",
    language: str = "English",
    use_academic: bool = True,
    use_news: bool = True,
    use_web: bool = False,
    on_progress=None,
) -> dict:
    """Run the four stages in the same order as the original project."""
    state = {}

    request = {
        "question": topic,
        "period": period,
        "domain": domain,
        "region": region,
        "audience": audience,
        "depth": depth,
        "language": language,
    }
    request_text = "\n".join(
        [
            f"Date: {date.today().isoformat()}",
            f"Question: {topic}",
            f"Time range: {period}",
            f"Domain: {domain}",
            f"Region: {region}",
            f"Audience: {audience}",
            f"Depth: {depth}",
        ]
    )
    state["request"] = request

    main_model, critic_model = create_models()
    state["models"] = {
        "research": "Gemini",
        "critic": "Groq" if critic_model is not main_model else "Gemini",
    }

    # Step 1: Search Agent
    if on_progress:
        on_progress("search", "running")

    search_agent = build_search_agent(main_model)
    state["search_plan"] = search_agent.invoke({"request": request_text})

    query = make_search_query(topic, domain, region)
    source_limit = {"Quick scan": 4, "Standard": 6, "Deep dive": 8}[depth]
    evidence, warnings = collect_evidence(
        query=query,
        days=days,
        per_source=source_limit,
        use_academic=use_academic,
        use_news=use_news,
        use_web=use_web,
    )
    sources_text = format_evidence(evidence)
    state["search_results"] = sources_text
    state["evidence"] = evidence
    state["warnings"] = warnings

    if on_progress:
        on_progress("search", "done")

    # Step 2: Reader Agent
    if on_progress:
        on_progress("reader", "running")

    reader_agent = build_reader_agent(main_model)
    state["scraped_content"] = reader_agent.invoke(
        {
            "request": request_text,
            "search_plan": state["search_plan"],
            "sources": sources_text[:12000],
        }
    )

    if on_progress:
        on_progress("reader", "done")

    # Step 3: Writer Chain
    if on_progress:
        on_progress("writer", "running")

    writer_chain = build_writer_chain(main_model)
    state["report"] = writer_chain.invoke(
        {
            "request": request_text,
            "notes": state["scraped_content"][:6000],
            "sources": sources_text[:9000],
            "language": language,
        }
    )

    if on_progress:
        on_progress("writer", "done")

    # Step 4: Critic Chain
    if on_progress:
        on_progress("critic", "running")

    critic_chain = build_critic_chain(critic_model)
    state["feedback"] = critic_chain.invoke(
        {
            "report": state["report"][:7000],
            "sources": sources_text[:7000],
        }
    )

    if on_progress:
        on_progress("critic", "done")

    return state


def export_markdown(state: dict) -> str:
    """Create a downloadable report with sources and critic feedback."""
    lines = [state.get("report", ""), "", "---", "", "# Sources", ""]

    for source in state.get("evidence", []):
        lines.extend(
            [
                f"## [{source['id']}] {source['title']}",
                f"- Type: {source['kind']}",
                f"- Published: {source.get('published') or 'Unknown'}",
                f"- Publisher: {source.get('publisher') or 'Unknown'}",
                f"- URL: {source.get('url') or 'Unavailable'}",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "# Search plan",
            state.get("search_plan", ""),
            "",
            "# Reader notes",
            state.get("scraped_content", ""),
            "",
            "# Critic feedback",
            state.get("feedback", ""),
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    research_topic = input("Enter a research topic: ").strip()
    result = run_research_pipeline(research_topic)
    print("\nREPORT\n", result["report"])
    print("\nCRITIC FEEDBACK\n", result["feedback"])
