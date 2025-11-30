"""Conversational interface over emails."""
from __future__ import annotations

from typing import List

from maestro.data.models import Email
from maestro.nlp.llm import LLMClient
from maestro.services.search_service import SearchService


class ChatService:
    """Chat over the email corpus."""

    def __init__(self, search_service: SearchService, llm: LLMClient) -> None:
        self.search_service = search_service
        self.llm = llm

    def chat_with_emails(self, history: List[dict], top_k: int = 5) -> str:
        user_message = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
        relevant_emails = self.search_service.search_semantic(user_message, limit=top_k)
        context_snippets = self._format_context(relevant_emails)
        system_prompt = (
            "You are Maestro, an assistant that answers based on the user's email archive. "
            "Use the provided snippets to ground your answers."
        )
        augmented_history = history + [{"role": "system", "content": f"Context:\n{context_snippets}"}]
        return self.llm.chat(system_prompt=system_prompt, messages=augmented_history)

    def _format_context(self, emails: List[Email]) -> str:
        return "\n\n".join(
            f"Subject: {email.subject}\nFrom: {email.from_address}\nSummary: {email.summary or email.plain_text[:280]}"
            for email in emails
        )

