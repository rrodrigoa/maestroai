"""Local LLM utilities for chat and drafting."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from transformers import pipeline

from maestro.core.config import settings
from maestro.data.models import Email


class LLMClient(ABC):
    """Abstract interface for conversational and generation abilities."""

    @abstractmethod
    def chat(self, system_prompt: str, messages: List[dict]) -> str:
        """Chat given a prompt and message history."""

    @abstractmethod
    def generate_email_draft(self, instruction: str, context_emails: List[Email]) -> str:
        """Generate an email draft using provided instruction and context."""


class HFCausalLLM(LLMClient):
    """Hugging Face causal LM wrapper using local models."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.llm_model_name
        device_name = device or settings.device
        self.device = 0 if device_name == "cuda" else -1
        self.generator = pipeline("text-generation", model=self.model_name, device=self.device)

    def chat(self, system_prompt: str, messages: List[dict]) -> str:
        prompt = self._build_chat_prompt(system_prompt, messages)
        response = self.generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
        return response[0]["generated_text"][len(prompt) :].strip()

    def generate_email_draft(self, instruction: str, context_emails: List[Email]) -> str:
        context = "\n\n".join(
            f"From: {email.from_address}\nSubject: {email.subject}\nSummary: {email.summary or email.plain_text[:280]}"
            for email in context_emails
        )
        prompt = (
            f"You are Maestro, an email drafting assistant. Use the context below to craft a helpful response.\n"
            f"Context:\n{context}\n\nInstruction: {instruction}\nDraft:"
        )
        response = self.generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
        return response[0]["generated_text"][len(prompt) :].strip()

    def _build_chat_prompt(self, system_prompt: str, messages: List[dict]) -> str:
        serialized = system_prompt + "\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            serialized += f"{role}: {content}\n"
        serialized += "assistant:"
        return serialized

