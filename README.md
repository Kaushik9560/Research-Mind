# ResearchMind

ResearchMind is a multi-agent AI research system that helps users research a topic using live academic, news, and web sources.

The project follows a simple workflow:

Search → Reader → Writer → Critic

The main idea is to divide the research process into separate stages so that each stage has a clear responsibility.

## Research Workflow

1. **Search Agent**  
   It understands the user query and collects relevant information from OpenAlex, Google News, and optionally Tavily.

2. **Reader Agent**  
   It reads the retrieved abstracts and snippets, compares the information, and identifies important findings, recent updates, disagreements, and evidence gaps.

3. **Writer Chain**  
   It uses the `prompt | llm | output parser` flow to generate a structured research report using the collected sources.

4. **Critic Chain**  
   It reviews the generated report and checks for unsupported claims, bias, missing evidence, and overconfident statements.

Gemini is used for the Search, Reader, and Writer stages.

If a Groq API key is available, Groq is used for the Critic stage. Otherwise, Gemini is used for the review as well.

Each retrieved source is assigned a stable source ID such as `[S1]`, `[S2]`, or `[S3]`.

These source IDs are used in the generated report so that the information can be linked back to the retrieved evidence.

The application also shows the Reader notes, Critic feedback, source list, and the complete research flow. The final report can be downloaded as a Markdown file.

## Tech Stack

- Python
- LangChain
- Streamlit
- Google Gemini
- Groq
- OpenAlex
- Google News RSS
- Tavily

## Run Locally

```bash
cd multi-agent-Project/Multi-agent-research-system

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

streamlit run app.py
```

Add your `GOOGLE_API_KEY` in the `.env` file.

`GROQ_API_KEY` is optional.

OpenAlex and Google News do not require API keys.

Tavily is also optional and is used only when its API key is available.

The project can also detect a `.env` file from a parent directory, so an existing configuration can be reused.

## Deployment

The application can be deployed using Streamlit Community Cloud or Docker-based platforms.

Deployment-related configuration and commands are available in `DEPLOYMENT.md`.

## Important Notes

- Live results depend on the availability of external services.
- Google News links may redirect through Google News.
- AI-generated reports can still contain mistakes.
- Important information should always be verified using the original sources.
