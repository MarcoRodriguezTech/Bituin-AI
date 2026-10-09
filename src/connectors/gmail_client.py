"""Gmail connector via Google API (OAuth desktop flow) with mock fallback.

First live sync opens a browser for Google consent, then caches a refresh
token in token.json — subsequent syncs need no re-login.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import GMAIL_CREDENTIALS_PATH, GMAIL_SCOPES, GMAIL_TOKEN_PATH, MOCK_GMAIL_PATH

log = logging.getLogger(__name__)


def _load_mock() -> list[dict]:
    try:
        with open(MOCK_GMAIL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("Mock Gmail data not found at %s", MOCK_GMAIL_PATH)
        return []


def _header(headers: list[dict], name: str) -> str:
    return next((h["value"] for h in headers if h["name"].lower() == name.lower()), "")


class GmailClient:
    def __init__(self, credentials_path: str = GMAIL_CREDENTIALS_PATH):
        self.credentials_path = credentials_path
        self.offline = not Path(credentials_path).exists()

    def fetch(self) -> list[dict]:
        if self.offline:
            log.info("credentials.json missing — using mock Gmail data.")
            return _load_mock()
        try:
            return self._fetch_live()
        except Exception as e:  # noqa: BLE001 — any API failure -> offline mode
            log.warning("Gmail API failed (%s) — falling back to mock data.", e)
            return _load_mock()

    def _get_service(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if Path(GMAIL_TOKEN_PATH).exists():
            creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH, GMAIL_SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, GMAIL_SCOPES
                )
                creds = flow.run_local_server(port=0)
            Path(GMAIL_TOKEN_PATH).write_text(creds.to_json())

        return build("gmail", "v1", credentials=creds)

    def _fetch_live(self) -> list[dict]:
        service = self._get_service()
        result = (
            service.users()
            .messages()
            .list(userId="me", q="is:unread in:inbox category:primary", maxResults=50)
            .execute()
        )
        records = []
        for stub in result.get("messages", []):
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=stub["id"], format="metadata",
                     metadataHeaders=["From", "Subject", "Date"])
                .execute()
            )
            headers = msg.get("payload", {}).get("headers", [])
            ts_ms = int(msg.get("internalDate", 0)) / 1000
            records.append(
                {
                    "id": f"gmail-{msg['id']}",
                    "source": "gmail",
                    "sender": _header(headers, "From") or "unknown",
                    "subject": _header(headers, "Subject") or "(no subject)",
                    "content": msg.get("snippet", ""),
                    "timestamp": datetime.fromtimestamp(ts_ms, tz=timezone.utc).isoformat(),
                    "is_read": False,
                }
            )
        return records
