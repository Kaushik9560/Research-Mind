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


SOURCE_LIMITS = {"Quick scan": 4, "Standard": 6, "Deep dive": 8}


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

    unique_words = []
    for word in useful_words:
        if word not in unique_words:
            unique_words.append(word)
    return " ".join(unique_words[:12]) or topic


def notify_progress(callback, stage: str, status: str) -> None:
    """Tell the UI (or terminal) a stage started/finished, if a callback exists."""
    if callback:
        callback(stage, status)


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
    """Run Search -> Reader -> Writer -> Critic and return one result dictionary."""
    topic = topic.strip()
    if not topic:
        raise ValueError("Enter a research question first.")
    if days < 1:
        raise ValueError("The time range must be at least one day.")
    if depth not in SOURCE_LIMITS:
        raise ValueError("Choose Quick scan, Standard, or Deep dive.")
    if not any((use_academic, use_news, use_web)):
        raise ValueError("Select at least one source.")

    # One dictionary carries outputs from each stage to the next.
    request = {
        "question": topic,
        "period": period,
        "domain": domain,
        "region": region,
        "audience": audience,
        "depth": depth,
        "language": language,
        "days": days,
        "sources": {"academic": use_academic, "news": use_news, "web": use_web},
    }
    request_text = (
        f"Date: {date.today().isoformat()}\nQuestion: {topic}\n"
        f"Time range: {period} ({days} days)\nDomain: {domain}\nRegion: {region}\n"
        f"Audience: {audience}\nDepth: {depth}\nLanguage: {language}"
    )
    state = {"request": request}

    main_model, critic_model = create_models()
    state["models"] = {
        "research": "Gemini",
        "critic": "Groq" if critic_model is not main_model else "Gemini",
    }

    # Step 1: Search Agent
    notify_progress(on_progress, "search", "running")

    search_agent = build_search_agent(main_model)
    state["search_plan"] = search_agent.invoke({"request": request_text})

    query = make_search_query(topic, domain, region)
    state["search_query"] = query
    evidence, warnings = collect_evidence(
        query=query,
        days=days,
        per_source=SOURCE_LIMITS[depth],
        use_academic=use_academic,
        use_news=use_news,
        use_web=use_web,
    )
    sources_text = format_evidence(evidence)
    state["search_results"] = sources_text
    state["evidence"] = evidence
    state["warnings"] = warnings

    notify_progress(on_progress, "search", "done")

    # Step 2: Reader Agent
    notify_progress(on_progress, "reader", "running")

    reader_agent = build_reader_agent(main_model)
    state["reader_notes"] = reader_agent.invoke(
        {
            "request": request_text,
            "search_plan": state["search_plan"],
            "sources": sources_text[:12000],
        }
    )

    notify_progress(on_progress, "reader", "done")

    # Step 3: Writer Chain
    notify_progress(on_progress, "writer", "running")

    writer_chain = build_writer_chain(main_model)
    state["report"] = writer_chain.invoke(
        {
            "request": request_text,
            "notes": state["reader_notes"][:6000],
            "sources": sources_text[:9000],
            "language": language,
        }
    )

    notify_progress(on_progress, "writer", "done")

    # Step 4: Critic Chain
    notify_progress(on_progress, "critic", "running")

    critic_chain = build_critic_chain(critic_model)
    state["feedback"] = critic_chain.invoke(
        {
            "report": state["report"][:7000],
            "sources": sources_text[:7000],
        }
    )

    notify_progress(on_progress, "critic", "done")

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
                f"- Authors: {source.get('authors') or 'Not listed'}",
                f"- URL: {source.get('url') or 'Unavailable'}",
                f"- OpenAlex cited-by count: {source.get('cited_by', 0)}",
                "",
                source.get("summary") or "No abstract/snippet available.",
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
            "# Search query",
            state.get("search_query", ""),
            "",
            "# Reader notes",
            state.get("reader_notes", ""),
            "",
            "# Critic feedback",
            state.get("feedback", ""),
        ]
    )
    if state.get("warnings"):
        lines.extend(["", "# Retrieval warnings", *state["warnings"]])
    return "\n".join(lines)


if __name__ == "__main__":
    research_topic = input("Enter a research topic: ").strip()
    result = run_research_pipeline(research_topic)
    print("\nREPORT\n", result["report"])
    print("\nCRITIC FEEDBACK\n", result["feedback"])
