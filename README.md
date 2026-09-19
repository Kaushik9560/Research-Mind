# InsightForge

InsightForge is a multi-agent research intelligence workspace for researchers,
students, founders, policy teams, and anyone exploring a fast-moving domain. It
collects recent evidence, separates established trends from early signals,
challenges weak claims, and produces a source-mapped report.

## Research workflow

1. **Research Strategist** scopes the question and defines evidence criteria.
2. **Evidence Collector** searches recent OpenAlex papers and Google News. It can
   also search the general web when a Tavily key is configured.
3. **Trend Analyst** identifies established trends, emerging signals, drivers,
   counter-signals, and gaps.
4. **Skeptic Agent** checks the analysis for bias, contradictions, and claims that
   exceed the evidence.
5. **Lead Researcher** writes the final cited report.

Every collected source receives a stable ID such as `[S3]`. The report, trend
radar, adversarial review, evidence library, and full audit trail remain visible
and can be exported as Markdown.

## Run locally

```bash
cd multi-agent-Project/Multi-agent-research-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add at least one supported model key to .env
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
