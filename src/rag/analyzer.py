"""LLM email analysis: urgency triage + action-item extraction + briefing."""
import json
import logging

import ollama

from config.settings import LLM_MODEL, OLLAMA_HOST

log = logging.getLogger(__name__)

_ANALYZE_SYSTEM = (
    "You analyze one email and respond with ONLY a JSON object, no other text.\n"
    'Schema: {"urgency": "urgent"|"action"|"fyi", "reason": "one short sentence", '
    '"tasks": [{"description": "imperative action", "due_date": "YYYY-MM-DD or null"}]}\n'
    "- urgent: real-world deadline within ~48h or explicitly time-critical\n"
    "- action: asks the reader to do something, no tight deadline\n"
    "- fyi: informational, newsletters, receipts, no action needed. "
    "Live-stream/video notifications, promotions, and social alerts are ALWAYS fyi.\n"
    "- tasks: concrete action items for the reader; empty list if none. "
    "due_date as ISO date or null if none stated."
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
        tasks = [
            {"description": str(t.get("description", ""))[:200], "due_date": t.get("due_date")}
            for t in data.get("tasks", [])
            if isinstance(t, dict) and t.get("description")
        ]
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
