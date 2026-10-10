"""An offline interview demo that runs the real pipeline with sample API/model replies."""

import os
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pipeline import export_markdown, run_research_pipeline


DEMO_TOPIC = "What are the latest trends in AI in education?"


def demo_http_get(url: str, **kwargs):
    """Supply small fictional API responses; the real collectors still parse them."""
    print(f"\nHTTP GET (sample response): {url}")
    print("Query parameters:", kwargs.get("params", {}))
    response = Mock()
    response.raise_for_status.return_value = None
    if "api.openalex.org" in url:
        response.json.return_value = {
            "results": [{
                "display_name": "DEMO: AI tutoring pilot",
                "doi": "https://example.com/demo-paper",
                "publication_date": "2026-09-01",
                "primary_location": {"source": {"display_name": "Demo Journal"}},
                "authorships": [{"author": {"display_name": "Demo Author"}}],
                "abstract_inverted_index": {
                    "AI": [0], "tutoring": [1], "may": [2],
                    "support": [3], "practice.": [4],
                },
                "cited_by_count": 2,
            }],
        }
    elif "news.google.com" in url:
        response.content = b"""<rss><channel><item>
            <title>DEMO: Schools evaluate AI tutors</title>
            <link>https://example.com/demo-news</link>
            <pubDate>Tue, 01 Sep 2026 12:00:00 GMT</pubDate>
            <source>Demo Newsroom</source>
            <description>Schools are testing AI tools; privacy remains a concern.</description>
        </item></channel></rss>"""
    else:
        raise AssertionError(f"Unexpected URL in offline demo: {url}")
    return response


def demo_model_reply(prompt):
    """Print the actual formatted prompt and return a fictional reply for its role."""
    messages = prompt.to_messages()
    role = messages[0].content
    print("\nCHAIN CALL: prompt -> offline sample model -> StrOutputParser")
    for message in messages:
        print(f"\n{message.type.upper()} INPUT:\n{message.content}")

    if "search agent" in role:
        reply = "Search for AI tutoring pilots, classroom adoption, and privacy concerns."
    elif "research reader" in role:
        reply = "[S1] suggests tutoring may support practice. [S2] mentions trials and privacy. "
        reply += "These two sample records do not prove better learning outcomes."
    elif "research writer" in role:
        reply = (
            "# Offline sample report\n\n"
            "## Executive answer\nAI tutoring is being explored in these fictional records [S1][S2].\n\n"
            "## Key findings and trends\nTutoring may support practice [S1]. "
            "Schools are testing tools while considering privacy [S2].\n\n"
            "## Limitations\nTwo sample snippets cannot establish effectiveness.\n\n"
            "## Implications and conclusion\nEvaluate outcomes and privacy before adoption."
        )
    elif "research critic" in role:
        reply = (
            "Score: 8/10 (sample).\nStrengths: cited claims and explicit uncertainty.\n"
            "Areas to improve: obtain real studies and measurable outcomes.\n"
            "One-line verdict: a demonstration, not a verified research conclusion."
        )
    else:
        raise AssertionError(f"Unexpected model role in offline demo: {role}")
    print(f"\nMODEL OUTPUT (sample):\n{reply}")
    return AIMessage(content=reply)


def show_progress(stage: str, status: str) -> None:
    """Print the same stage notifications used by the Streamlit progress bar."""
    print(f"\n=== {stage.upper()}: {status} ===")


def run_dry_run() -> dict:
    """Run real prompts, collectors, ID assignment, and orchestration without network calls."""
    print("OFFLINE DRY RUN: all records and model answers are fictional samples.\n")
    sample_model = RunnableLambda(demo_model_reply)
    # Patch only external boundaries; run_research_pipeline itself is unchanged.
    with (
        patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}),
        patch("pipeline.create_models", return_value=(sample_model, sample_model)),
        patch("tools.requests.get", side_effect=demo_http_get),
    ):
        result = run_research_pipeline(
            DEMO_TOPIC, depth="Quick scan", on_progress=show_progress,
        )
    result["models"] = {"research": "Offline sample", "critic": "Offline sample"}
    print("\nFINAL RESULT DICTIONARY:")
    for key, value in result.items():
        # Some LangChain versions return a str subclass; show its useful Python type.
        value_type = "str" if isinstance(value, str) else type(value).__name__
        print(f"  {key}: {value_type}")
    print("\nEXPORTED MARKDOWN:\n")
    print(export_markdown(result))
    return result


if __name__ == "__main__":
    run_dry_run()
