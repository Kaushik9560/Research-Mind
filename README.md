# ResearchMind

A multi-agent research assistant built with Python, LangChain, and Streamlit,
with a simple, sequential workflow:

**Search → Reader → Writer → Critic**

It collects academic abstracts, news snippets, and optional web results, compares
them, writes a cited report, and reviews the report against the collected evidence.
Gemini runs research. Groq runs the critic when configured; otherwise Gemini
performs the review too.

## Understand the code

| File | Job |
| --- | --- |
| `tools.py` | Retrieve and normalize sources; assign citation IDs |
| `agents.py` | Configure models and build role-specific prompt/model/parser chains |
| `pipeline.py` | Run the four stages in order and export Markdown |
| `app.py` | Streamlit form, progress, result tabs, and follow-up chat |
| `dry_run.py` | Execute the real flow with fictional responses, offline |

The Search model plans the research. Python creates the keyword query and calls
OpenAlex, Google News RSS, and optional Tavily. The Reader compares the returned
abstracts/snippets. Source IDs such as `[S1]` map report citations to source links.
The critic produces feedback; it does not automatically rewrite the report.

See [the current interview guide](docs/INTERVIEW_GUIDE.md) for every function's
inputs, output, purpose, a worked dry run, and interview questions. Earlier
handbooks describe previous versions.

## Tech stack

Python, LangChain, Streamlit, Google Gemini, Groq, OpenAlex, Google News RSS,
and optional Tavily.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set GOOGLE_API_KEY in .env. GROQ_API_KEY is optional.
streamlit run app.py
```

A `.env` file in a parent directory is also discovered. Model names and output
token limits can be configured there (default output budget: 4096 tokens). Streamlit Cloud settings use `st.secrets`.
OpenAlex supports basic keyless queries and an optional `OPENALEX_API_KEY` for a
larger budget; see [its authentication docs](https://help.openalex.org/api/authentication/).
Google News RSS needs no key. Tavily requires `TAVILY_API_KEY`.

The UI includes question examples, time/depth/language/domain/region/audience
options, stage progress, five result tabs, source filtering, Markdown download,
and follow-up answers using the existing research context.

## Offline interview demo

```bash
python dry_run.py
```

No keys or network access required. The demo prints actual formatted prompts,
search requests, sample responses, stage progress, result dictionary keys, and
exported Markdown. It uses the real pipeline and source parsers. All model
answers and source records are fictional and labelled as samples.

## Check the project

```bash
python -m unittest discover -s tests -v
```

Tests use sample responses to verify retrieval parsing, failures, citations,
model fallback, pipeline wiring, export, and Streamlit interactions.

To run live research from a terminal:

```bash
python pipeline.py
```

## Deployment and limits

See [DEPLOYMENT.md](DEPLOYMENT.md) for Streamlit Cloud and Docker instructions.
Live runs depend on your API credentials, quota, and network. The Reader uses
abstracts/snippets and model inputs have character limits; full sources remain
available in the UI/export. The critique and citations help inspect claims but
do not guarantee factual accuracy.
