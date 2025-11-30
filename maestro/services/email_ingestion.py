"""Email ingestion pipeline."""
from __future__ import annotations

import logging

from maestro.data.repository import EmailRepository
from maestro.gmail.client import GmailClient, GoogleGmailClient
from maestro.processing.html_cleaner import HTMLCleaner
from maestro.nlp.embeddings import EmbeddingIndex, EmbeddingModel
from maestro.nlp.indexing import IndexCoordinator, WordIndex
from maestro.nlp.summarizer import Summarizer

logger = logging.getLogger(__name__)


class EmailIngestionService:
    """Download, clean, store, summarize, and index emails."""

    def __init__(
        self,
        gmail_client: GmailClient,
        repository: EmailRepository,
        cleaner: HTMLCleaner,
        embedding_model: EmbeddingModel,
        embedding_index: EmbeddingIndex,
        summarizer: Summarizer,
        word_index: WordIndex,
    ) -> None:
        self.gmail_client = gmail_client
        self.repository = repository
        self.cleaner = cleaner
        self.summarizer = summarizer
        self.index_coordinator = IndexCoordinator(embedding_model, embedding_index, word_index)

    def sync_gmail(self, max_results: int = 200) -> int:
        raw_emails = self.gmail_client.fetch_emails(max_results=max_results)
        domain_emails = []
        for raw in raw_emails:
            plain = self.cleaner.to_plain_text(raw.raw_html)
            summary = self.summarizer.summarize(plain)
            email = GoogleGmailClient.to_email(raw, plain_text=plain, summary=summary)
            domain_emails.append(email)
        self.repository.save_emails(domain_emails)
        persisted = self.repository.list_recent(limit=len(domain_emails))
        self.index_coordinator.index_emails(persisted)
        logger.info("Synced %s emails", len(domain_emails))
        return len(domain_emails)

