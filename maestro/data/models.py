"""SQLAlchemy ORM models for Maestro."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class."""


class Email(Base):
    """Email message stored locally."""

    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gmail_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    from_address: Mapped[str] = mapped_column(String(256))
    to_addresses: Mapped[str] = mapped_column(Text)
    cc_addresses: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bcc_addresses: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    subject: Mapped[str] = mapped_column(String(512))
    raw_html: Mapped[str] = mapped_column(Text)
    plain_text: Mapped[str] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:  # pragma: no cover - repr convenience
        return f"Email(id={self.id}, subject={self.subject!r})"

