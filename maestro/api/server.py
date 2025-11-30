"""FastAPI server exposing Maestro capabilities."""
from __future__ import annotations

import logging
from fastapi import FastAPI

from maestro.core.config import settings
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
from maestro.api.schemas import (
    ChatRequest,
    ChatResponse,
    DraftRequest,
    DraftResponse,
    EmailResponse,
    ImportRequest,
    ImportResponse,
    SearchRequest,
    SearchResponse,
)

configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="Maestro Email Assistant")

# Instantiate core services
repository = SqlAlchemyEmailRepository(settings.database_url)
gmail_client = GoogleGmailClient()
cleaner = HTMLCleaner()
embedding_model = HFEmbeddingModel()
# temporary model to get dimension
_sample_vec = embedding_model.embed_texts(["bootstrap"])
embedding_index = FaissEmbeddingIndex(dim=_sample_vec.shape[1])
word_index = WordIndex()
summarizer = HFSummarizer()
llm_client = HFCausalLLM()
ingestion_service = EmailIngestionService(
    gmail_client=gmail_client,
    repository=repository,
    cleaner=cleaner,
    embedding_model=embedding_model,
    embedding_index=embedding_index,
    summarizer=summarizer,
    word_index=word_index,
)
search_service = SearchService(repository, embedding_model, embedding_index)
chat_service = ChatService(search_service, llm_client)
drafting_service = DraftingService(llm_client, search_service)


@app.post("/emails/import/gmail", response_model=ImportResponse)
def import_gmail(payload: ImportRequest) -> ImportResponse:
    imported = ingestion_service.sync_gmail(max_results=payload.max_results or 200)
    return ImportResponse(imported=imported)


@app.post("/emails/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    if payload.mode == "keyword":
        emails = search_service.search_keyword(payload.query, limit=payload.limit)
    elif payload.mode == "hybrid":
        emails = search_service.search_hybrid(payload.query, limit=payload.limit)
    else:
        emails = search_service.search_semantic(payload.query, limit=payload.limit)
    return SearchResponse(
        results=[
            EmailResponse(
                id=email.id,
                subject=email.subject,
                from_address=email.from_address,
                to_addresses=email.to_addresses,
                date=email.date,
                summary=email.summary,
            )
            for email in emails
        ]
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    reply = chat_service.chat_with_emails(payload.messages, top_k=payload.top_k)
    return ChatResponse(reply=reply)


@app.post("/emails/draft", response_model=DraftResponse)
def draft_email(payload: DraftRequest) -> DraftResponse:
    draft = drafting_service.draft_email(payload.instruction, payload.related_query)
    return DraftResponse(draft=draft)


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

