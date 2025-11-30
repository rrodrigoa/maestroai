"""Retrieval helpers combining semantic search with storage."""
from __future__ import annotations

from typing import List

from maestro.data.models import Email
from maestro.data.repository import EmailRepository
from maestro.nlp.embeddings import EmbeddingIndex, EmbeddingModel


def semantic_retrieve(query: str, repo: EmailRepository, model: EmbeddingModel, index: EmbeddingIndex, k: int = 5) -> List[Email]:
    """Retrieve top-k emails using semantic similarity."""
    query_vec = model.embed_texts([query])
    results = index.search(query_vec, k=k)
    emails: List[Email] = []
    for email_id, _score in results:
        email = repo.get_email(email_id)
        if email:
            emails.append(email)
    return emails

