"""Pydantic schemas for API payloads."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class EmailResponse(BaseModel):
    id: int
    subject: str
    from_address: str
    to_addresses: str
    date: datetime
    summary: Optional[str]


class ImportRequest(BaseModel):
    max_results: Optional[int] = Field(default=200, ge=1, le=500)


class ImportResponse(BaseModel):
    imported: int


class SearchRequest(BaseModel):
    query: str
    mode: Literal["keyword", "semantic", "hybrid"] = "semantic"
    limit: int = 20


class SearchResponse(BaseModel):
    results: List[EmailResponse]


class ChatRequest(BaseModel):
    messages: List[dict]
    top_k: int = 5


class ChatResponse(BaseModel):
    reply: str


class DraftRequest(BaseModel):
    instruction: str
    related_query: Optional[str] = None


class DraftResponse(BaseModel):
    draft: str

