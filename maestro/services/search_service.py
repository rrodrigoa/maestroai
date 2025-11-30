"""Search service for Maestro."""
from __future__ import annotations

from maestro.data.repository import EmailRepository
from maestro.nlp.embeddings import EmbeddingIndex, EmbeddingModel
from maestro.nlp.retrieval import semantic_retrieve


class SearchService:
    """Provide keyword and semantic search over emails."""

    def __init__(self, repository: EmailRepository, embedding_model: EmbeddingModel, embedding_index: EmbeddingIndex) -> None:
        self.repository = repository
        self.embedding_model = embedding_model
        self.embedding_index = embedding_index

    def search_keyword(self, query: str, limit: int = 20):
        return self.repository.search_by_keyword(query, limit=limit)

    def search_semantic(self, query: str, limit: int = 20):
        return semantic_retrieve(query, self.repository, self.embedding_model, self.embedding_index, k=limit)

    def search_hybrid(self, query: str, limit: int = 20):
        semantic_results = self.search_semantic(query, limit=limit)
        keyword_results = self.search_keyword(query, limit=limit)
        seen = {email.id for email in semantic_results}
        merged = semantic_results + [email for email in keyword_results if email.id not in seen]
        return merged[:limit]

