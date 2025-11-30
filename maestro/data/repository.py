"""Repository abstractions for emails."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from maestro.core.config import settings
from maestro.data.models import Base, Email

logger = logging.getLogger(__name__)


class EmailRepository(ABC):
    """Abstract repository for storing and querying emails."""

    @abstractmethod
    def save_emails(self, emails: Iterable[Email]) -> None:
        """Persist a collection of emails."""

    @abstractmethod
    def get_email(self, id: int) -> Optional[Email]:
        """Retrieve an email by primary key."""

    @abstractmethod
    def get_by_gmail_id(self, gmail_id: str) -> Optional[Email]:
        """Retrieve an email by Gmail message id."""

    @abstractmethod
    def search_by_keyword(self, query: str, limit: int = 20) -> List[Email]:
        """Search for emails containing a keyword in subject or body."""

    @abstractmethod
    def list_recent(self, limit: int = 50) -> List[Email]:
        """List recent emails by date."""


class SqlAlchemyEmailRepository(EmailRepository):
    """SQLite-backed repository using SQLAlchemy."""

    def __init__(self, database_url: str | None = None) -> None:
        self.engine = create_engine(database_url or settings.database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def save_emails(self, emails: Iterable[Email]) -> None:
        email_list = list(emails)
        with self.SessionLocal() as session:
            for email in email_list:
                session.merge(email)
            session.commit()
            logger.info("Saved %s emails", len(email_list))

    def get_email(self, id: int) -> Optional[Email]:
        with self.SessionLocal() as session:
            return session.get(Email, id)

    def get_by_gmail_id(self, gmail_id: str) -> Optional[Email]:
        with self.SessionLocal() as session:
            stmt = select(Email).where(Email.gmail_id == gmail_id)
            return session.scalars(stmt).first()

    def search_by_keyword(self, query: str, limit: int = 20) -> List[Email]:
        pattern = f"%{query}%"
        with self.SessionLocal() as session:
            stmt = (
                select(Email)
                .where((Email.subject.ilike(pattern)) | (Email.plain_text.ilike(pattern)))
                .order_by(Email.date.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt))

    def list_recent(self, limit: int = 50) -> List[Email]:
        with self.SessionLocal() as session:
            stmt = select(Email).order_by(Email.date.desc()).limit(limit)
            return list(session.scalars(stmt))

