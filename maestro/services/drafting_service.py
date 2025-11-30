"""Service for generating email drafts."""
from __future__ import annotations

from maestro.nlp.llm import LLMClient
from maestro.services.search_service import SearchService


class DraftingService:
    """Generate email drafts using an LLM and optional context search."""

    def __init__(self, llm: LLMClient, search: SearchService) -> None:
        self.llm = llm
        self.search = search

    def draft_email(self, instruction: str, related_query: str | None = None):
        context_emails = []
        if related_query:
            context_emails = self.search.search_semantic(related_query, limit=5)
        return self.llm.generate_email_draft(instruction, context_emails)

