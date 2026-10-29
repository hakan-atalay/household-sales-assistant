from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

REASONS = ("wrong_number", "wrong_scope", "wrong_analysis", "too_long_or_unclear", "off_topic", "other")
EXPORT_KINDS = ("sft", "dpo", "eval")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    tool_calls TEXT NOT NULL,
    model TEXT,
    grounded INTEGER NOT NULL,
    rating INTEGER,
    reason TEXT,
    correction TEXT
)
"""


class MessageNotFound(KeyError):
    pass


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def save_message(self, question: str, answer: str, tool_calls: list[dict], model: str | None, grounded: bool) -> int:
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO messages (created_at, question, answer, tool_calls, model, grounded) VALUES (?,?,?,?,?,?)",
                (time.time(), question, answer, json.dumps(tool_calls, ensure_ascii=False), model, int(grounded)),
            )
            return int(cursor.lastrowid)

    def set_feedback(self, message_id: int, rating: int, reason: str | None = None, correction: str | None = None) -> None:
        if rating not in (1, -1):
            raise ValueError("rating must be 1 (like) or -1 (dislike)")
        if reason and reason not in REASONS:
            raise ValueError(f"reason must be one of {REASONS}")
        with closing(self._connect()) as db, db:
            updated = db.execute(
                "UPDATE messages SET rating=?, reason=?, correction=? WHERE id=?",
                (rating, reason, (correction or "").strip() or None, message_id),
            ).rowcount
        if not updated:
            raise MessageNotFound(message_id)

    def liked_examples(self, limit: int = 2) -> list[tuple[str, str]]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT question, answer FROM messages WHERE rating=1 AND grounded=1 ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [(r["question"], r["answer"]) for r in rows]

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(rating=1),0) AS likes, COALESCE(SUM(rating=-1),0) AS dislikes, "
                "COALESCE(SUM(correction IS NOT NULL),0) AS corrections FROM messages"
            ).fetchone()
        return {"messages": row["n"], "likes": row["likes"], "dislikes": row["dislikes"], "corrections": row["corrections"]}

    def export(self, kind: str) -> str:
        if kind not in EXPORT_KINDS:
            raise ValueError(f"kind must be one of {EXPORT_KINDS}")
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM messages WHERE rating IS NOT NULL ORDER BY id").fetchall()
        records = [record for row in rows if (record := _record(kind, row))]
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


def _record(kind: str, row: sqlite3.Row) -> dict | None:
    tools = json.loads(row["tool_calls"] or "[]")
    if kind == "sft" and (row["rating"] == 1 or row["correction"]):
        return {
            "messages": [{"role": "user", "content": row["question"]},
                         {"role": "assistant", "content": row["correction"] or row["answer"]}],
            "tool_calls": tools,
        }
    if kind == "dpo" and row["rating"] == -1 and row["correction"]:
        return {"prompt": row["question"], "chosen": row["correction"], "rejected": row["answer"]}
    if kind == "eval" and row["rating"] == -1:
        return {"question": row["question"], "tools_used": tools, "reason": row["reason"], "correction": row["correction"]}
    return None
