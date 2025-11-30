"""Utilities for converting HTML email bodies to plain text."""
from __future__ import annotations

from bs4 import BeautifulSoup
import html2text


class HTMLCleaner:
    """Convert HTML to plain text for downstream processing."""

    def __init__(self) -> None:
        self._html2text = html2text.HTML2Text()
        self._html2text.ignore_links = False
        self._html2text.ignore_images = True

    def to_plain_text(self, html: str) -> str:
        """Convert HTML content to cleaned plain text."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        stripped = soup.get_text("\n", strip=True)
        markdown_like = self._html2text.handle(stripped)
        return markdown_like.strip()

