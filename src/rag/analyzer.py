"""LLM email analysis: urgency triage + action-item extraction + briefing."""
import json
import logging
import re
from datetime import datetime

import ollama
from dateutil import parser as _dateparser

from config.settings import LLM_MODEL, OLLAMA_HOST

log = logging.getLogger(__name__)

# Strict date patterns — a month name or a 4-digit year is required, so
# document codes like "MCS 7-9" don't read as dates.
_DATE_RE = re.compile(
    r"(?:\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*(?:,?\s+\d{4})?)",
    re.I,
)


# Notification noise the model occasionally calls "urgent" anyway —
# live-stream alerts are never time-critical.
_STREAM_RE = re.compile(r"\bis live\b|went live|livestream|live stream", re.I)


def _dates_in_text(text: str, ref_iso: str) -> set[str]:
    """ISO dates explicitly written in the email text (subject + snippet).

    Missing components (e.g. year in "September 15") default to the email's
    own date — "Sept 15" on a Sept 5 email means Sept 15 of that year.
    """
    try:
        base = datetime.fromisoformat(str(ref_iso).replace("Z", "+00:00"))
    except ValueError:
        base = datetime.now()
    found = set()
    for m in _DATE_RE.finditer(text):
        try:
            found.add(_dateparser.parse(m.group(0), default=base).date().isoformat())
        except (ValueError, OverflowError):
            continue
    return found

_ANALYZE_SYSTEM = (
    "You analyze one email and respond with ONLY a JSON object, no other text.\n"
    'Schema: {"urgency": "urgent"|"action"|"fyi", "reason": "one short sentence", '
    '"tasks": [{"description": "imperative action", "due_date": "YYYY-MM-DD or null"}]}\n'
    "- urgent: action needed within ~48h of TODAY'S DATE or explicitly "
    "time-critical. A deadline that already passed is NOT urgent.\n"
    "- action: asks the reader to do something, no tight deadline\n"
    "- fyi: informational, newsletters, receipts, no action needed. "
    "Live-stream/video notifications, promotions, and social alerts are ALWAYS fyi.\n"
    "- tasks: concrete action items for the reader; empty list if none. "
    "Only create a task if the email explicitly asks the reader to do "
    "something — a bare link or notification is not a task. "
    "due_date as ISO date ONLY if an explicit deadline date is written in "
    "the email text; otherwise null. Never guess or infer a date."
)

_DIGEST_SYSTEM = (
    "You are a concise local assistant writing a morning briefing from the "
    "user's inbox analysis below. Group it as: URGENT (now), ACTION ITEMS "
    "(with due dates), and FYI. Bullet points, one line each, plain text.\n"
    "IMPORTANT: cluster similar/repeated items into ONE bullet with a count "
    "(e.g. '12 emails about failed CI runs'). Max 12 bullets total. "
    "Prioritize real deadlines over notifications."
)


class EmailAnalyzer:
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.client = ollama.Client(host=OLLAMA_HOST)

    def analyze(self, email: dict) -> dict:
        prompt = (
            f"Today: {datetime.now().date().isoformat()}\n"
            f"From: {email['sender']}\n"
            f"Subject: {email['subject']}\n"
            f"Date: {email['timestamp']}\n\n"
            f"{email['content']}"
        )
        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _ANALYZE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0, "num_ctx": 2048},
            )
            data = json.loads(resp["message"]["content"])
        except Exception as e:  # noqa: BLE001 — never let one email break the batch
            log.warning("Analysis failed for %s (%s)", email["id"], e)
            data = {}
        urgency = data.get("urgency")
        if urgency not in ("urgent", "action", "fyi"):
            urgency = "fyi"
        if _STREAM_RE.search(f"{email['subject']} {email['sender']}"):
            urgency = "fyi"
        tasks = [
            {"description": str(t.get("description", ""))[:200], "due_date": t.get("due_date")}
            for t in data.get("tasks", [])
            if isinstance(t, dict) and t.get("description")
        ]
        # The model sometimes invents deadlines (e.g. reading a document code
        # like "SJS 9-10" as a date) — keep a due_date only if that exact date
        # is written in the email text, otherwise null it out.
        written = _dates_in_text(
            f"{email['subject']}\n{email['content']}", email["timestamp"]
        )
        for t in tasks:
            dd = str(t.get("due_date") or "")[:10]
            if dd and dd not in written:
                t["due_date"] = None
        return {
            "urgency": urgency,
            "reason": str(data.get("reason", ""))[:300],
            "tasks": tasks[:5],
        }

    def digest(self, emails: list[dict], analysis_map: dict, tasks: list[dict]) -> str:
        lines = []
        for e in emails:
            a = analysis_map.get(e["id"], {})
            lines.append(
                f"[{a.get('urgency', '?').upper()}] {e['subject']} — {e['sender']}"
                f" | {a.get('reason', '')}"
            )
        if tasks:
            lines.append("\nOPEN TASKS:")
            for t in tasks:
                due = f" (due {t['due_date']})" if t.get("due_date") else ""
                lines.append(f"- {t['description']}{due}")

        resp = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _DIGEST_SYSTEM},
                {"role": "user", "content": "\n".join(lines)},
            ],
            options={"temperature": 0.3, "num_ctx": 4096},
        )
        return resp["message"]["content"]
