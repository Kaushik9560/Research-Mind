# Deploy ResearchMind

## Streamlit Community Cloud (recommended)

1. Push this project to a GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io), choose **Create app**,
   and select the repository.
3. Set the entrypoint to `app.py`. If this project remains inside a larger
   repository, use:
   `multi-agent-Project/Multi-agent-research-system/app.py`.
4. In **App settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your-key"
   GROQ_MODEL = "openai/gpt-oss-20b"
   GOOGLE_API_KEY = "your-key"
   GOOGLE_MODEL = "gemini-3.6-flash"
   MAX_OUTPUT_TOKENS = "4096"
   # Optional
   TAVILY_API_KEY = "your-key"
   OPENALEX_API_KEY = "your-key"
   ```

5. Deploy. Google News needs no key. OpenAlex supports basic keyless queries;
   an optional `OPENALEX_API_KEY` increases its budget.

Never commit `.env` or `.streamlit/secrets.toml`.

## Docker / Render / Railway

The included `Dockerfile` exposes port `8501` and includes a health check.
Configure these environment variables in the hosting dashboard:

- `GROQ_API_KEY` (optional; enables the independent Critic model)
- `GROQ_MODEL=openai/gpt-oss-20b`
- `GOOGLE_API_KEY` (required) and `GOOGLE_MODEL=gemini-3.6-flash`
- `MAX_OUTPUT_TOKENS=4096`
- `TAVILY_API_KEY` (optional)
- `OPENALEX_API_KEY` (optional; increases the academic search budget)

Build locally with:

```bash
docker build -t researchmind .
docker run --rm -p 8501:8501 --env-file .env researchmind
```
