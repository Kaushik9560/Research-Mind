# InsightForge

InsightForge is a tutorial-shaped multi-agent research workspace for researchers,
students, founders, policy teams, and anyone exploring a fast-moving domain. It
keeps the original project's simple Search → Reader → Writer → Critic flow while
adding live academic/news evidence, citations, a second-model critic, and deployment.

## Research workflow

1. **Search Agent** scopes the question and searches OpenAlex, Google News, and
   optional Tavily web results.
2. **Reader Agent** reads all retrieved abstracts/snippets and compares findings,
   recent signals, disagreements, and evidence gaps.
3. **Writer Chain** uses the familiar `prompt | llm | output parser` pattern to
   create the cited report.
4. **Critic Chain** scores the report and highlights unsupported or overstated
   claims.

Gemini performs the Search, Reader, and Writer stages. Groq performs the Critic
stage when its key is present; otherwise Gemini reviews its own report. This is
the only model fallback in the project.

Every collected source receives a stable ID such as `[S3]`. The report, Reader
notes, Critic feedback, evidence library, and complete four-stage audit trail
remain visible and can be exported as Markdown.

## How this extends the tutorial

| Tutorial concept | Production upgrade |
|---|---|
| Tavily search results | OpenAlex + Google News + optional Tavily |
| Reader scrapes one URL | Reader compares every retrieved record |
| One OpenAI model | Gemini research + optional Groq critic |
| Writer report | Source-ID-grounded report with uncertainty |
| Critic score | Citation and bias review using an independent provider |
| Terminal output | Responsive Streamlit UI and downloadable audit trail |

## Run locally

```bash
cd multi-agent-Project/Multi-agent-research-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GOOGLE_API_KEY; GROQ_API_KEY is optional
streamlit run app.py
```

The app also discovers a `.env` file in a parent directory, so an existing repo
configuration can be reused. OpenAlex and Google News do not require API keys.
Tavily is optional.

## Deploy

The app is ready for Streamlit Community Cloud and Docker-based platforms.
See [DEPLOYMENT.md](DEPLOYMENT.md) for the entrypoint, secrets, and commands.

## Notes

- Live-source availability depends on the upstream services and network access.
- News aggregation links may redirect through Google News.
- Model-generated synthesis can still be wrong. Verify high-stakes claims against
  the linked primary sources.
