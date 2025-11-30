# Maestro AI Backend MVP

A local-first email assistant backend built with FastAPI, SQLite, and local Hugging Face models. Everything runs on your machine (CUDA preferred when available) with no remote AI services.

## Features
- Gmail sync with OAuth via the Google API client
- SQLite storage via SQLAlchemy
- HTML cleaning to plain text
- Keyword and FAISS-powered semantic search
- Local summarization and LLM-powered chat/drafting using `transformers`
- FastAPI HTTP API and Typer CLI

## Getting started
1. Install dependencies (Python 3.13+):
   ```bash
   pip install -e .
   ```
2. Prepare Gmail credentials in `./config/credentials.json` and run the first sync to generate a token:
   ```bash
   python -m maestro.cli.main sync-gmail --max-results 50
   ```
3. Start the API server:
   ```bash
   uvicorn maestro.api.server:app --reload
   ```
4. Try the CLI search or chat locally:
   ```bash
   python -m maestro.cli.main search "invoices" --mode semantic
   python -m maestro.cli.main chat
   ```

Notes:
- FAISS indices are stored locally (default `./data/faiss.index`).
- Model names and paths are configurable via environment variables in `maestro/core/config.py`.
- Services are intentionally modular for future extension.

