"""Gmail client integration."""
from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from maestro.core.config import settings
from maestro.data.models import Email

logger = logging.getLogger(__name__)
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@dataclass
class RawGmailEmail:
    """Lightweight representation of a Gmail message."""

    gmail_id: str
    thread_id: str
    raw_html: str
    subject: str
    from_address: str
    to_addresses: str
    cc_addresses: str | None
    bcc_addresses: str | None
    date: datetime


class GmailClient(ABC):
    """Abstract Gmail client."""

    @abstractmethod
    def fetch_emails(self, max_results: int = 100) -> List[RawGmailEmail]:
        """Fetch recent emails."""


class GoogleGmailClient(GmailClient):
    """Implementation backed by Google Gmail API."""

    def __init__(self) -> None:
        self.creds = self._load_credentials()
        self.service = build("gmail", "v1", credentials=self.creds)

    def _load_credentials(self) -> Credentials:
        creds: Credentials | None = None
        if settings.gmail_token_path.exists():
            creds = Credentials.from_authorized_user_file(str(settings.gmail_token_path), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(settings.gmail_credentials_path), SCOPES)
                creds = flow.run_local_server(port=0)
            settings.gmail_token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(settings.gmail_token_path, "w", encoding="utf-8") as token:
                token.write(creds.to_json())
        return creds

    def fetch_emails(self, max_results: int = 100) -> List[RawGmailEmail]:
        logger.info("Fetching up to %s emails from Gmail", max_results)
        results = self.service.users().messages().list(userId="me", maxResults=max_results).execute()
        messages = results.get("messages", [])
        fetched: List[RawGmailEmail] = []
        for msg in messages:
            full = self.service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
            payload = full.get("payload", {})
            headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
            snippet = full.get("snippet", "")
            parts = payload.get("parts", [])
            body = ""
            for part in parts:
                if part.get("mimeType", "") in {"text/html", "text/plain"}:
                    data = part.get("body", {}).get("data")
                    if data:
                        body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                        break
            fetched.append(
                RawGmailEmail(
                    gmail_id=full["id"],
                    thread_id=full.get("threadId", ""),
                    raw_html=body or snippet,
                    subject=headers.get("subject", "(no subject)"),
                    from_address=headers.get("from", ""),
                    to_addresses=headers.get("to", ""),
                    cc_addresses=headers.get("cc"),
                    bcc_addresses=headers.get("bcc"),
                    date=datetime.fromtimestamp(int(full.get("internalDate", 0)) / 1000),
                )
            )
        return fetched

    @staticmethod
    def to_email(raw: RawGmailEmail, plain_text: str, summary: str | None = None) -> Email:
        """Convert RawGmailEmail to domain Email."""

        return Email(
            gmail_id=raw.gmail_id,
            thread_id=raw.thread_id,
            from_address=raw.from_address,
            to_addresses=raw.to_addresses,
            cc_addresses=raw.cc_addresses,
            bcc_addresses=raw.bcc_addresses,
            subject=raw.subject,
            raw_html=raw.raw_html,
            plain_text=plain_text,
            summary=summary,
            date=raw.date,
        )

