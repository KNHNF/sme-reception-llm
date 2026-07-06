"""
Call record store (local, offline).

A dedicated SQLite log of calls handled by the assistant: the caller's transcript,
the action taken, and the booking outcome. This is the business's own record of a
call, kept locally with the caller's consent. It is deliberately separate from
metrics.db (which is ephemeral demo instrumentation and is wiped by /metrics/clear).

Privacy: transcripts are caller personal data. data/call_log.db is gitignored and
never leaves the machine. retention_purge() lets the business delete old records so
data is not kept longer than needed.

Usage:
    from src.call_log import log_call, list_calls
    log_call(session_id, transcript, action, spoken_reply, booking, consent, source)
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "call_log.db"


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            REAL NOT NULL,
            session_id    TEXT NOT NULL,
            source        TEXT,
            consent       INTEGER,
            transcript    TEXT,
            action        TEXT,
            booking       TEXT,
            spoken_reply  TEXT
        )
    """)
    con.commit()
    return con


def log_call(session_id: str, transcript: str, action, spoken_reply: str = "",
             booking=None, consent: bool = False, source: str = "text"):
    """Record one handled turn of a call. Consent is the caller's agreement to be recorded."""
    action_str = action.get("action") if isinstance(action, dict) else (
        str(action) if action is not None else None)
    booking_str = json.dumps(booking) if booking is not None else None
    with _conn() as con:
        con.execute(
            "INSERT INTO calls (ts, session_id, source, consent, transcript, action, "
            "booking, spoken_reply) VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), session_id, source, int(consent), transcript, action_str,
             booking_str, spoken_reply),
        )


def list_calls(limit: int = 50) -> list[dict]:
    """Most recent calls first, for a records view or the viva."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM calls ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def retention_purge(days: float) -> int:
    """Delete call records older than `days`. Returns rows removed. Supports a retention policy."""
    cutoff = time.time() - days * 86400
    with _conn() as con:
        cur = con.execute("DELETE FROM calls WHERE ts < ?", (cutoff,))
        return cur.rowcount


def clear_calls():
    with _conn() as con:
        con.execute("DELETE FROM calls")


if __name__ == "__main__":
    print(json.dumps(list_calls(), indent=2))
