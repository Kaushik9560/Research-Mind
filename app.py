"""Streamlit UI: collect options, run the pipeline, and display its results."""

import os
from collections import Counter

import streamlit as st

from agents import ModelConfigurationError, answer_follow_up, provider_name
from pipeline import export_markdown, run_research_pipeline
from tools import format_evidence


PERIODS = {"30 days": 30, "6 months": 183, "1 year": 365, "3 years": 1095, "5 years": 1825}
STAGES = ["search", "reader", "writer", "critic"]
EXAMPLES = [
    "Small language models and on-device AI",
    "Green hydrogen storage trends",
    "Generative AI in Indian higher education",
]


def load_cloud_secrets() -> None:
    """Copy Streamlit Cloud settings to the environment without replacing local ones."""
    names = [
        "GOOGLE_API_KEY", "GOOGLE_MODEL", "GROQ_API_KEY", "GROQ_MODEL",
        "MAX_OUTPUT_TOKENS", "TAVILY_API_KEY", "OPENALEX_API_KEY",
    ]
    try:
        for name in names:
            if name in st.secrets:
                os.environ.setdefault(name, str(st.secrets[name]))
    except FileNotFoundError:
        pass  # Local development can use .env instead of Streamlit secrets.


def initialize_session() -> None:
    """Keep the report, conversation, and question when Streamlit reruns the script."""
    defaults = {"research_result": None, "followups": [], "question_input": ""}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_form() -> tuple[bool, dict]:
    """Return whether Start was clicked and the keyword arguments for the pipeline."""
    for column, example in zip(st.columns(3), EXAMPLES):
        if column.button(example, use_container_width=True):
            st.session_state.question_input = f"What are the latest trends in {example}?"

    with st.form("research_form"):
        topic = st.text_area("What do you want to research?", key="question_input", height=105)
        time_column, depth_column, language_column = st.columns(3)
        period = time_column.selectbox("Time range", list(PERIODS), index=2)
        depth = depth_column.selectbox("Depth", ["Quick scan", "Standard", "Deep dive"], index=1)
        language = language_column.selectbox("Language", ["English", "Hinglish", "Hindi"])

        with st.expander("More options"):
            domain_column, region_column, audience_column = st.columns(3)
            domain = domain_column.text_input("Domain", placeholder="e.g. Healthcare AI")
            region = region_column.text_input("Region", value="Global")
            audience = audience_column.selectbox(
                "Audience",
                ["Researcher", "General reader", "Student", "Founder / operator", "Policy professional"],
            )
            paper_column, news_column, web_column = st.columns(3)
            use_academic = paper_column.checkbox("Academic papers", value=True)
            use_news = news_column.checkbox("Recent news", value=True)
            tavily_ready = bool(os.getenv("TAVILY_API_KEY"))
            use_web = web_column.checkbox(
                "General web", value=tavily_ready, disabled=not tavily_ready,
                help="Add TAVILY_API_KEY to enable this source.",
            )
        submitted = st.form_submit_button("Start research →", use_container_width=True)

    options = {
        "topic": topic.strip(), "days": PERIODS[period], "period": period,
        "domain": domain.strip() or "General", "region": region.strip() or "Global",
        "audience": audience, "depth": depth, "language": language,
        "use_academic": use_academic, "use_news": use_news, "use_web": use_web,
    }
    return submitted, options


def start_research(options: dict) -> None:
    """Validate input, show stage progress, and save a successful research result."""
    if not options["topic"]:
        st.warning("Enter a research question first.")
        return
    if not any((options["use_academic"], options["use_news"], options["use_web"])):
        st.warning("Select at least one source under More options.")
        return

    with st.status("Starting research…", expanded=True) as status:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def show_progress(stage: str, stage_status: str) -> None:
            """Translate the pipeline callback into a stage label and progress bar."""
            position = STAGES.index(stage)
            progress_text.write(f"{stage.title()}: {stage_status}")
            finished = position + (1 if stage_status == "done" else 0)
            progress_bar.progress(finished / len(STAGES))

        try:
            result = run_research_pipeline(**options, on_progress=show_progress)
        except (ModelConfigurationError, ValueError) as exc:
            status.update(label="Check your configuration", state="error")
            st.error(str(exc))
        except Exception as exc:
            status.update(label="Research stopped", state="error")
            error_text = str(exc).lower()
            if "503" in error_text or "high demand" in error_text:
                st.error("The model provider is busy. Retry shortly or change GOOGLE_MODEL in your settings.")
            elif "429" in error_text or "quota" in error_text:
                st.error("The model quota or rate limit was reached. Check your API account and retry later.")
            else:
                st.error(
                    f"Research could not finish ({type(exc).__name__}). "
                    "Check your network, model name, and API quota, then try again."
                )
        else:
            st.session_state.research_result = result
            st.session_state.followups = []
            status.update(label="Research complete", state="complete", expanded=False)


def render_sources(evidence: list[dict]) -> None:
    """Filter source records and display their excerpts and original links."""
    if not evidence:
        st.info("No sources matched. Try broader wording or a longer time range.")
        return
    kinds = ["All"] + sorted({source["kind"] for source in evidence})
    selected = st.selectbox("Show", kinds, key="evidence_filter")
    for source in evidence:
        if selected != "All" and source["kind"] != selected:
            continue
        with st.container(border=True):
            # st.text keeps external source titles/excerpts as plain text.
            st.text(f"[{source['id']}] {source['title']}")
            st.caption(
                f"{source['kind']} · {source.get('publisher') or 'Unknown'} · "
                f"{source.get('published') or 'Unknown date'}"
            )
            st.text(source.get("summary") or "No excerpt available.")
            url = source.get("url") or ""
            if url.startswith(("http://", "https://")):
                st.link_button("Open source ↗", url)


def render_results(result: dict) -> None:
    """Display the report, reader notes, sources, critic, and search audit trail."""
    evidence = result["evidence"]
    counts = Counter(source["kind"] for source in evidence)
    st.subheader("Your intelligence brief")
    for column, label, value in zip(
        st.columns(4),
        ["Total sources", "Academic papers", "Current signals", "Pipeline stages"],
        [len(evidence), counts["Academic"], counts["News"] + counts["Web"], 4],
    ):
        column.metric(label, value)
    for warning in result["warnings"]:
        st.warning(warning)
    if not evidence:
        st.warning("No live evidence was retrieved. This report cannot establish factual trends.")

    report_tab, reader_tab, sources_tab, critic_tab, search_tab = st.tabs(
        ["Report", "Reader notes", "Sources", "Critic feedback", "Search plan"]
    )
    with report_tab:
        st.markdown(result["report"])
        st.download_button(
            "Download report", export_markdown(result),
            file_name="researchmind_report.md", mime="text/markdown",
        )
    with reader_tab:
        st.markdown(result["reader_notes"])
    with sources_tab:
        render_sources(evidence)
    with critic_tab:
        st.markdown(result["feedback"])
    with search_tab:
        st.markdown(result["search_plan"])
        st.caption(f"Search query: {result['search_query']}")
        st.caption(f"Research: {result['models']['research']} · Critic: {result['models']['critic']}")
        with st.expander("Run configuration"):
            st.json(result["request"])


def render_followups(result: dict) -> None:
    """Show chat history and answer questions from the existing research context."""
    st.subheader("Ask the evidence")
    for message in st.session_state.followups:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about a source, contradiction, trend, or research gap…")
    if not question:
        return
    st.session_state.followups.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    context = "\n\n".join([result["report"], result["feedback"], format_evidence(result["evidence"])])
    with st.chat_message("assistant"):
        with st.spinner("Checking the evidence…"):
            try:
                answer = answer_follow_up(context, question, result["request"]["language"])
            except Exception:
                answer = "I could not answer that follow-up. Check your connection and API quota."
            st.markdown(answer)
    st.session_state.followups.append({"role": "assistant", "content": answer})


def main() -> None:
    """Run the page from top to bottom each time Streamlit reruns the script."""
    st.set_page_config(page_title="ResearchMind", page_icon="◈", layout="wide")
    load_cloud_secrets()
    initialize_session()
    st.title("◈ ResearchMind")
    st.caption(f"Configured models: {provider_name()}")
    st.write("Search live sources, compare evidence, write a cited report, and review its claims.")
    submitted, options = render_form()
    st.caption("Search → Reader → Writer → Critic")
    if submitted:
        start_research(options)
    result = st.session_state.research_result
    if result:
        render_results(result)
        render_followups(result)
    st.caption("Sources: OpenAlex, Google News, optional Tavily. Verify important claims using the source links.")


if __name__ == "__main__":
    main()
