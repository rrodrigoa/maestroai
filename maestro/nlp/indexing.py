"""Index orchestration for keyword and semantic search."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from maestro.data.models import Email
from maestro.nlp.embeddings import EmbeddingIndex, EmbeddingModel

logger = logging.getLogger(__name__)


class WordIndex:
    """In-memory inverted index for keyword search."""

    def __init__(self) -> None:
        self.index: Dict[str, set[int]] = defaultdict(set)

    def build(self, emails: Iterable[Email]) -> None:
        for email in emails:
            tokens = self._tokenize(email.subject + " " + email.plain_text)
            for token in tokens:
                self.index[token].add(email.id)
        logger.info("Keyword index built for %s terms", len(self.index))

    def search(self, query: str, limit: int = 20) -> List[int]:
        tokens = self._tokenize(query)
        results: set[int] = set()
        for token in tokens:
            results |= self.index.get(token, set())
        return list(results)[:limit]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\b\w+\b", text)]


class IndexCoordinator:
    """Coordinates semantic and keyword index updates."""

    def __init__(self, embedding_model: EmbeddingModel, embedding_index: EmbeddingIndex, word_index: WordIndex) -> None:
        self.embedding_model = embedding_model
        self.embedding_index = embedding_index
        self.word_index = word_index

    def index_emails(self, emails: List[Email]) -> None:
        if not emails:
            return
        vectors = self.embedding_model.embed_texts([email.plain_text for email in emails])
        ids = [email.id for email in emails]
        self.embedding_index.add_items(ids, vectors)
        self.word_index.build(emails)

