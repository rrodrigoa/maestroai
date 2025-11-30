"""Summarization utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod

from transformers import pipeline

from maestro.core.config import settings


class Summarizer(ABC):
    """Abstract summarization service."""

    @abstractmethod
    def summarize(self, text: str, max_length: int = 128) -> str:
        """Generate a summary of the text."""


class HFSummarizer(Summarizer):
    """HuggingFace pipeline-based summarizer."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.summarizer_model_name
        self.device = 0 if (device or settings.device) == "cuda" else -1
        self.pipeline = pipeline("summarization", model=self.model_name, device=self.device)

    def summarize(self, text: str, max_length: int = 128) -> str:
        if not text.strip():
            return ""
        result = self.pipeline(text, max_length=max_length, min_length=max_length // 2, do_sample=False)
        return result[0]["summary_text"]

