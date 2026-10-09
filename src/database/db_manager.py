"""SQLite raw-storage layer for normalized communications."""
import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone

from config.settings import DB_PATH


class DBManager:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS communications (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    sender TEXT,
                    subject TEXT,
                    content TEXT,
                    timestamp TEXT,
                    is_read INTEGER DEFAULT 0,
                    hash TEXT UNIQUE
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis (
                    email_id TEXT PRIMARY KEY,
                    urgency TEXT,
                    reason TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    email_id TEXT,
                    description TEXT,
                    due_date TEXT,
                    done INTEGER DEFAULT 0
                )
                """
            )

    def insert_record(self, record: dict) -> bool:
        """Insert a normalized record. Returns False if it's a duplicate."""
        content_hash = hashlib.sha256(
            f"{record['source']}_{record['sender']}_{record['subject']}_{record['content']}".encode()
        ).hexdigest()
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO communications VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record["id"],
                        record["source"],
                        record.get("sender", ""),
                        record.get("subject", ""),
                        record["content"],
                        record.get("timestamp")
                        or datetime.now(timezone.utc).isoformat(),
                        int(record.get("is_read", 0)),
                        content_hash,
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def insert_many(self, records: list[dict]) -> int:
        return sum(self.insert_record(r) for r in records)

    def fetch_all(self, unread_only: bool = False) -> list[dict]:
        query = "SELECT id, source, sender, subject, content, timestamp, is_read FROM communications"
        if unread_only:
            query += " WHERE is_read = 0"
        query += " ORDER BY timestamp DESC"
        rows = self.conn.execute(query).fetchall()
        return [
            {
                "id": r[0],
                "source": r[1],
                "sender": r[2],
                "subject": r[3],
                "content": r[4],
                "timestamp": r[5],
                "is_read": bool(r[6]),
            }
            for r in rows
        ]

    def fetch_recent(self, days: int = 2, limit: int = 20) -> list[dict]:
        """Most recent messages by received time — deterministic context for
        'summarize my inbox' style questions (bypasses vector search)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            """SELECT id, source, sender, subject, content, timestamp, is_read
               FROM communications WHERE timestamp >= ?
               ORDER BY timestamp DESC LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        if len(rows) < limit:
            have = {r[0] for r in rows}
            rows += [
                r
                for r in self.conn.execute(
                    """SELECT id, source, sender, subject, content, timestamp, is_read
                       FROM communications ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
                if r[0] not in have
            ]
        return [
            {
                "id": r[0],
                "source": r[1],
                "sender": r[2],
                "subject": r[3],
                "content": r[4],
                "timestamp": r[5],
                "is_read": bool(r[6]),
            }
            for r in rows[:limit]
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM communications").fetchone()[0]

    # ---------- Analysis ----------
    def unanalyzed_ids(self) -> list[str]:
        rows = self.conn.execute(
            """SELECT c.id FROM communications c
               LEFT JOIN analysis a ON c.id = a.email_id
               WHERE a.email_id IS NULL"""
        ).fetchall()
        return [r[0] for r in rows]

    def save_analysis(self, email_id: str, urgency: str, reason: str, tasks: list[dict]):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO analysis VALUES (?, ?, ?)",
                (email_id, urgency, reason),
            )
            self.conn.execute("DELETE FROM tasks WHERE email_id = ?", (email_id,))
            for i, t in enumerate(tasks):
                self.conn.execute(
                    "INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?, 0)",
                    (f"{email_id}::t{i}", email_id, t.get("description", ""), t.get("due_date")),
                )

    def fetch_analysis_map(self) -> dict:
        rows = self.conn.execute("SELECT email_id, urgency, reason FROM analysis").fetchall()
        return {r[0]: {"urgency": r[1], "reason": r[2]} for r in rows}

    # ---------- Tasks ----------
    def fetch_tasks(self, include_done: bool = True) -> list[dict]:
        query = """SELECT t.id, t.email_id, t.description, t.due_date, t.done,
                          c.subject, c.sender
                   FROM tasks t LEFT JOIN communications c ON t.email_id = c.id"""
        if not include_done:
            query += " WHERE t.done = 0"
        query += " ORDER BY t.done, COALESCE(t.due_date, '9999') ASC"
        return [
            {
                "id": r[0],
                "email_id": r[1],
                "description": r[2],
                "due_date": r[3],
                "done": bool(r[4]),
                "subject": r[5],
                "sender": r[6],
            }
            for r in self.conn.execute(query).fetchall()
        ]

    def set_task_done(self, task_id: str, done: bool):
        with self.conn:
            self.conn.execute("UPDATE tasks SET done = ? WHERE id = ?", (int(done), task_id))

    def delete_task(self, task_id: str):
        with self.conn:
            self.conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    def urgency_counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT urgency, COUNT(*) FROM analysis GROUP BY urgency"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def close(self):
        self.conn.close()
