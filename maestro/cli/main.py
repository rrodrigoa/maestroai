"""Typer-based CLI entrypoint."""
from __future__ import annotations

import typer

from maestro.core.logging import configure_logging
from maestro.data.repository import SqlAlchemyEmailRepository
from maestro.gmail.client import GoogleGmailClient
from maestro.processing.html_cleaner import HTMLCleaner
from maestro.nlp.embeddings import FaissEmbeddingIndex, HFEmbeddingModel
from maestro.nlp.indexing import WordIndex
from maestro.nlp.llm import HFCausalLLM
from maestro.nlp.summarizer import HFSummarizer
from maestro.services.chat_service import ChatService
from maestro.services.drafting_service import DraftingService
from maestro.services.email_ingestion import EmailIngestionService
from maestro.services.search_service import SearchService

app = typer.Typer(help="Interact with Maestro locally")


def bootstrap_services():
    configure_logging()
    repo = SqlAlchemyEmailRepository()
    gmail = GoogleGmailClient()
    cleaner = HTMLCleaner()
    embedding_model = HFEmbeddingModel()
    dim = embedding_model.embed_texts(["bootstrap"]).shape[1]
    embedding_index = FaissEmbeddingIndex(dim=dim)
    word_index = WordIndex()
    summarizer = HFSummarizer()
    llm = HFCausalLLM()

    ingestion = EmailIngestionService(
        gmail_client=gmail,
        repository=repo,
        cleaner=cleaner,
        embedding_model=embedding_model,
        embedding_index=embedding_index,
        summarizer=summarizer,
        word_index=word_index,
    )
    search = SearchService(repo, embedding_model, embedding_index)
    chat = ChatService(search, llm)
    draft = DraftingService(llm, search)
    return ingestion, search, chat, draft


@app.command()
def sync_gmail(max_results: int = typer.Option(200, help="Max emails to fetch")):
    ingestion, *_ = bootstrap_services()
    imported = ingestion.sync_gmail(max_results=max_results)
    typer.echo(f"Imported {imported} emails")


@app.command()
def search(query: str, mode: str = typer.Option("semantic", help="keyword|semantic|hybrid")):
    _, search_service, _, _ = bootstrap_services()
    if mode == "keyword":
        emails = search_service.search_keyword(query)
    elif mode == "hybrid":
        emails = search_service.search_hybrid(query)
    else:
        emails = search_service.search_semantic(query)
    for email in emails:
        typer.echo(f"[{email.id}] {email.subject} - {email.summary or email.plain_text[:120]}")


@app.command()
def chat():
    _, _, chat_service, _ = bootstrap_services()
    history: list[dict] = []
    typer.echo("Starting Maestro chat. Type 'exit' to quit.")
    while True:
        message = typer.prompt("You")
        if message.strip().lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": message})
        reply = chat_service.chat_with_emails(history)
        history.append({"role": "assistant", "content": reply})
        typer.echo(f"Maestro: {reply}")


@app.command()
def draft(instruction: str, related_query: str = typer.Option(None, help="Optional related search query")):
    _, _, _, drafting = bootstrap_services()
    draft_text = drafting.draft_email(instruction, related_query)
    typer.echo(draft_text)


if __name__ == "__main__":
    app()

