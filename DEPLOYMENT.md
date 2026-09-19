# Deploy InsightForge

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
   MAX_OUTPUT_TOKENS = "1400"
   # Optional
   TAVILY_API_KEY = "your-key"
   ```

5. Deploy. OpenAlex and Google News need no additional secrets.

Never commit `.env` or `.streamlit/secrets.toml`.

## Docker / Render / Railway

The included `Dockerfile` exposes port `8501` and includes a health check.
Configure these environment variables in the hosting dashboard:

- `GROQ_API_KEY` (required unless using another supported model provider)
- `GROQ_MODEL=openai/gpt-oss-20b`
- `MAX_OUTPUT_TOKENS=1400`
- `TAVILY_API_KEY` (optional)

Build locally with:

```bash
docker build -t insightforge .
docker run --rm -p 8501:8501 --env-file .env insightforge
```
