"""Simple, responsive Streamlit interface for InsightForge."""

from __future__ import annotations

import html
import os
import time
from collections import Counter

import streamlit as st


st.set_page_config(
    page_title="InsightForge — Research Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Streamlit Community Cloud stores configuration in st.secrets instead of .env.
# Local environment variables still take precedence.
for secret_name in (
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "MAX_OUTPUT_TOKENS",
    "GOOGLE_API_KEY",
    "GOOGLE_MODEL",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
):
    try:
        if secret_name in st.secrets:
            os.environ.setdefault(secret_name, str(st.secrets[secret_name]))
    except FileNotFoundError:
        break

from agents import ModelConfigurationError, answer_follow_up, provider_name  # noqa: E402
from pipeline import ResearchRequest, export_markdown, run_research_pipeline  # noqa: E402
from tools import format_evidence  # noqa: E402


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --bg: #f7f8f5;
    --card: #ffffff;
    --ink: #18211c;
    --muted: #69756e;
    --brand: #155e46;
    --brand-dark: #104936;
    --soft: #edf5f0;
    --accent: #d7f57a;
    --line: #dfe6e1;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer, header, [data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1080px; padding: 1.35rem 1.5rem 4rem; }

.nav { display:flex; align-items:center; justify-content:space-between; padding:.25rem 0 1.8rem; }
.brand { display:flex; align-items:center; gap:.65rem; font:800 1.02rem 'Manrope', sans-serif; color:var(--ink); }
.brand-icon { width:32px; height:32px; display:grid; place-items:center; border-radius:9px; background:var(--brand); color:var(--accent); }
.connection { display:flex; align-items:center; gap:.45rem; padding:.38rem .68rem; background:var(--card); border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:.72rem; }
.connection-dot { width:7px; height:7px; border-radius:50%; background:#23a46b; }

.hero { max-width:800px; margin:1.5rem auto 2rem; text-align:center; }
.eyebrow { color:var(--brand); font:700 .72rem 'Manrope', sans-serif; letter-spacing:.13em; text-transform:uppercase; }
.hero h1 { margin:.7rem 0 .8rem; color:var(--ink); font:800 clamp(2.25rem,5vw,4rem)/1.04 'Manrope', sans-serif; letter-spacing:-.05em; }
.hero h1 span { color:var(--brand); }
.hero p { max-width:650px; margin:auto; color:var(--muted); font-size:1rem; line-height:1.65; }

.section-kicker { margin-top:2.25rem; color:var(--brand); font:700 .7rem 'Manrope',sans-serif; letter-spacing:.12em; text-transform:uppercase; }
.section-heading { margin:.35rem 0 1rem; font:800 1.55rem 'Manrope',sans-serif; letter-spacing:-.025em; }

[data-testid="stForm"] { padding:1.35rem 1.4rem 1.5rem; background:var(--card); border:1px solid var(--line); border-radius:20px; box-shadow:0 14px 40px rgba(29,56,41,.055); }
.stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div {
    background:#fbfcfa !important; border-color:var(--line) !important; border-radius:11px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus { border-color:var(--brand) !important; box-shadow:0 0 0 3px rgba(21,94,70,.08) !important; }
[data-testid="stExpander"] { background:#fafbf9; border:1px solid var(--line); border-radius:12px; }

.stButton button, .stFormSubmitButton button, .stDownloadButton button, .stLinkButton a {
    border-radius:10px !important; min-height:2.65rem; font-weight:700 !important;
}
.stButton button { background:var(--card); border-color:var(--line); color:var(--ink); }
.stButton button:hover { border-color:var(--brand); color:var(--brand); }
.stFormSubmitButton button { width:100%; background:var(--brand) !important; border:1px solid var(--brand) !important; color:white !important; }
.stFormSubmitButton button:hover { background:var(--brand-dark) !important; border-color:var(--brand-dark) !important; }
.stDownloadButton button { background:var(--accent) !important; border-color:var(--accent) !important; color:var(--ink) !important; }

.agent-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.65rem; margin:1.1rem 0 0; }
.agent-card { padding:.9rem .8rem; background:var(--card); border:1px solid var(--line); border-radius:13px; }
.agent-number { color:var(--brand); font:800 .68rem 'Manrope',sans-serif; }
.agent-name { margin:.3rem 0 .12rem; color:var(--ink); font-weight:700; font-size:.82rem; }
.agent-role { color:var(--muted); font-size:.68rem; line-height:1.3; }

.metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin:1rem 0 1.4rem; }
.metric { padding:1rem; background:var(--card); border:1px solid var(--line); border-radius:13px; }
.metric strong { display:block; color:var(--brand); font:800 1.4rem 'Manrope',sans-serif; }
.metric span { color:var(--muted); font-size:.72rem; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.25rem; overflow-x:auto; }
[data-testid="stTabs"] button { font-weight:700; white-space:nowrap; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top:1.2rem; }
.source-card { margin:.7rem 0 .35rem; padding:1rem 1.05rem; background:var(--card); border:1px solid var(--line); border-radius:13px; }
.source-tags { display:flex; gap:.4rem; margin-bottom:.5rem; }
.tag { padding:.2rem .48rem; background:var(--soft); border-radius:999px; color:var(--brand); font-size:.66rem; font-weight:700; }
.source-title { color:var(--ink); font:700 .96rem 'Manrope',sans-serif; }
.source-meta { margin:.2rem 0 .45rem; color:var(--muted); font-size:.72rem; }
.source-summary { color:#4f5d55; font-size:.84rem; line-height:1.55; }
[data-testid="stChatMessage"] { background:var(--card); border:1px solid var(--line); border-radius:13px; }
.fine-print { max-width:680px; margin:3.5rem auto 0; text-align:center; color:#8a958e; font-size:.7rem; line-height:1.5; }

@media (max-width: 760px) {
    .block-container { padding:1rem .85rem 3rem; }
    .nav { padding-bottom:.7rem; }
    .connection span:last-child { display:none; }
    .hero { margin:1.2rem auto 1.5rem; text-align:left; }
    .hero h1 { font-size:2.35rem; }
    .hero p { font-size:.92rem; }
    [data-testid="stForm"] { padding:1rem; border-radius:15px; }
    .agent-grid { grid-template-columns:repeat(2,1fr); }
    .metrics { grid-template-columns:repeat(2,1fr); }
}

@media (max-width: 420px) {
    .hero h1 { font-size:2.05rem; }
    .agent-grid { grid-template-columns:1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


PERIODS = {
    "30 days": 30,
    "6 months": 183,
    "1 year": 365,
    "3 years": 1095,
    "5 years": 1825,
}

AGENTS = {
    "search": ("01", "Search Agent", "Plans and finds live sources"),
    "reader": ("02", "Reader Agent", "Reads and compares evidence"),
    "writer": ("03", "Writer Chain", "Writes the cited report"),
    "critic": ("04", "Critic Chain", "Reviews weak claims"),
}

for state_key, default in {
    "research_result": None,
    "followups": [],
    "question_input": "",
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default


st.markdown(
    f"""
<div class="nav">
  <div class="brand"><span class="brand-icon">◈</span>InsightForge</div>
  <div class="connection"><span class="connection-dot"></span><span>{html.escape(provider_name())} connected</span></div>
</div>
<div class="hero">
  <div class="eyebrow">Multi-agent research intelligence</div>
  <h1>Research any domain.<br><span>See what is changing.</span></h1>
  <p>Search live sources, read the evidence, write a cited brief, and critically review every important claim.</p>
</div>
""",
    unsafe_allow_html=True,
)


examples = [
    "Small language models and on-device AI",
    "Green hydrogen storage trends",
    "Generative AI in Indian higher education",
]
example_columns = st.columns(3)
for column, example in zip(example_columns, examples):
    if column.button(example, use_container_width=True):
        st.session_state.question_input = f"What are the latest trends in {example}?"
        st.rerun()


with st.form("research_form"):
    question = st.text_area(
        "What do you want to research?",
        key="question_input",
        placeholder="Ask a focused research question…",
        height=105,
    )

    basic_one, basic_two, basic_three = st.columns(3)
    with basic_one:
        period_label = st.selectbox("Time range", list(PERIODS), index=2)
    with basic_two:
        depth = st.selectbox("Depth", ["Quick scan", "Standard", "Deep dive"], index=1)
    with basic_three:
        language = st.selectbox("Language", ["English", "Hinglish", "Hindi"])

    with st.expander("More options", expanded=False):
        advanced_one, advanced_two, advanced_three = st.columns(3)
        with advanced_one:
            domain = st.text_input("Domain", placeholder="e.g. Healthcare AI")
        with advanced_two:
            region = st.text_input("Region", value="Global")
        with advanced_three:
            audience = st.selectbox(
                "Audience",
                ["Researcher", "General reader", "Student", "Founder / operator", "Policy professional"],
            )

        st.caption("Sources")
        source_one, source_two, source_three = st.columns(3)
        use_academic = source_one.checkbox("Academic papers", value=True)
        use_news = source_two.checkbox("Recent news", value=True)
        tavily_ready = bool(os.getenv("TAVILY_API_KEY"))
        use_web = source_three.checkbox(
            "General web",
            value=tavily_ready,
            disabled=not tavily_ready,
            help="Add TAVILY_API_KEY to enable this source.",
        )

    submitted = st.form_submit_button("Start research  →", use_container_width=True)


agent_cards = "".join(
    f'<div class="agent-card"><div class="agent-number">{number}</div>'
    f'<div class="agent-name">{name}</div><div class="agent-role">{role}</div></div>'
    for number, name, role in AGENTS.values()
)
st.markdown(f'<div class="agent-grid">{agent_cards}</div>', unsafe_allow_html=True)


if submitted:
    if not question.strip():
        st.warning("Enter a research question first.")
    elif not any((use_academic, use_news, use_web)):
        st.warning("Select at least one source under More options.")
    else:
        research_request = ResearchRequest(
            question=question.strip(),
            domain=domain.strip() or "General",
            days=PERIODS[period_label],
            period_label=period_label,
            region=region.strip() or "Global",
            audience=audience,
            depth=depth,
            language=language,
            use_academic=use_academic,
            use_news=use_news,
            use_web=use_web,
        )
        st.session_state.followups = []

        with st.status("Starting the research team…", expanded=True) as status:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def show_progress(step: str, state: str) -> None:
                number, name, role = AGENTS[step]
                position = list(AGENTS).index(step)
                if state == "running":
                    progress_text.markdown(f"**{number} · {name}** — {role}")
                    progress_bar.progress(position / len(AGENTS))
                else:
                    progress_bar.progress((position + 1) / len(AGENTS))

            try:
                output = run_research_pipeline(research_request, on_progress=show_progress)
                st.session_state.research_result = output
                progress_text.markdown("**Done** — Your cited report is ready.")
                status.update(label="Research complete", state="complete", expanded=False)
            except ModelConfigurationError as exc:
                status.update(label="Model configuration needed", state="error")
                st.error(str(exc))
            except Exception as exc:
                status.update(label="Research stopped", state="error")
                error_text = str(exc).lower()
                if "request too large" in error_text or "error code: 413" in error_text:
                    st.error("The model context limit was reached. Try Quick scan or a shorter time range.")
                else:
                    st.error(f"The run could not finish ({exc.__class__.__name__}). Please try again.")


result = st.session_state.research_result
if result:
    evidence = result.get("evidence", [])
    counts = Counter(item.get("kind", "Other") for item in evidence)
    request_data = result.get("request", {})

    st.markdown('<div class="section-kicker">Research complete</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Your intelligence brief</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="metrics">
  <div class="metric"><strong>{len(evidence)}</strong><span>Total sources</span></div>
  <div class="metric"><strong>{counts.get('Academic', 0)}</strong><span>Academic papers</span></div>
  <div class="metric"><strong>{counts.get('News', 0) + counts.get('Web', 0)}</strong><span>Current signals</span></div>
  <div class="metric"><strong>4</strong><span>Pipeline stages</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    for warning in result.get("warnings", []):
        st.warning(warning)
    if not evidence:
        st.warning("No live sources matched. Try broader wording or a longer time range.")

    report_tab, reader_tab, evidence_tab, review_tab, method_tab = st.tabs(
        ["Report", "Reader notes", "Sources", "Critic feedback", "Search plan"]
    )

    with report_tab:
        st.markdown(result.get("report", ""))
        st.download_button(
            "Download report",
            data=export_markdown(result),
            file_name=f"insightforge_{int(time.time())}.md",
            mime="text/markdown",
        )

    with reader_tab:
        st.markdown(result.get("reader_notes", ""))

    with evidence_tab:
        if evidence:
            available_kinds = ["All"] + sorted({item.get("kind", "Other") for item in evidence})
            selected_kind = st.selectbox("Show", available_kinds, key="evidence_filter")
            visible_evidence = (
                evidence
                if selected_kind == "All"
                else [item for item in evidence if item.get("kind") == selected_kind]
            )
            for item in visible_evidence:
                title = html.escape(str(item.get("title") or "Untitled"))
                summary = html.escape(str(item.get("summary") or "No excerpt available."))
                source_meta = " · ".join(
                    filter(None, [str(item.get("publisher") or ""), str(item.get("published") or "")])
                )
                st.markdown(
                    f'<div class="source-card"><div class="source-tags">'
                    f'<span class="tag">{html.escape(str(item.get("id")))}</span>'
                    f'<span class="tag">{html.escape(str(item.get("kind")))}</span></div>'
                    f'<div class="source-title">{title}</div>'
                    f'<div class="source-meta">{html.escape(source_meta)}</div>'
                    f'<div class="source-summary">{summary}</div></div>',
                    unsafe_allow_html=True,
                )
                source_url = str(item.get("url") or "")
                if source_url.startswith(("http://", "https://")):
                    st.link_button("Open source ↗", source_url)
        else:
            st.info("No evidence records are available for this run.")

    with review_tab:
        st.markdown(result.get("feedback", ""))

    with method_tab:
        st.markdown("### Search Agent plan")
        st.markdown(result.get("search_plan", ""))
        model_plan = result.get("model_plan", {})
        if isinstance(model_plan, dict):
            st.caption(
                f"Automatic model routing: {model_plan.get('primary', 'Auto')} synthesis · "
                f"{model_plan.get('reviewer', 'Auto')} independent review"
            )
        with st.expander("Run configuration"):
            if isinstance(request_data, dict):
                st.json(request_data)

    st.markdown('<div class="section-kicker">Follow-up</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Ask the evidence</div>', unsafe_allow_html=True)
    for message in st.session_state.followups:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    follow_up = st.chat_input("Ask about a source, contradiction, trend, or research gap…")
    if follow_up:
        st.session_state.followups.append({"role": "user", "content": follow_up})
        with st.chat_message("user"):
            st.markdown(follow_up)
        research_context = "\n\n".join(
            [
                str(result.get("report", "")),
                str(result.get("feedback", "")),
                format_evidence(evidence),
            ]
        )
        with st.chat_message("assistant"):
            with st.spinner("Checking the evidence…"):
                try:
                    answer = answer_follow_up(
                        research_context,
                        follow_up,
                        str(request_data.get("language", "English"))
                        if isinstance(request_data, dict)
                        else "English",
                        str(result.get("model_plan", {}).get("primary", "Auto"))
                        if isinstance(result.get("model_plan"), dict)
                        else "Auto",
                    )
                except Exception:
                    answer = "I could not answer that follow-up. Please try again."
                st.markdown(answer)
        st.session_state.followups.append({"role": "assistant", "content": answer})


st.markdown(
    '<div class="fine-print">InsightForge uses OpenAlex, Google News, and optional Tavily. '
    "AI synthesis can be wrong—verify important claims against the linked primary sources.</div>",
    unsafe_allow_html=True,
)
