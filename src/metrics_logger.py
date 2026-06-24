"""
Conversation Metrics Logger
Logs every pipeline turn to SQLite so results can be reviewed in the dashboard.

Usage:
    from src.metrics_logger import log_turn, get_metrics
    log_turn(session_id, utterance, action, validated, latency_ms)
    stats = get_metrics()

Dashboard:
    Start the API (uvicorn backend:app --port 5005) then open:
    http://localhost:5005/metrics-dashboard
"""

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "metrics.db"

ACTION_TYPES = [
    "book_appointment",
    "check_availability",
    "cancel_appointment",
    "clarify",
    "out_of_scope",
    "end_call",
]


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL NOT NULL,
            session_id  TEXT NOT NULL,
            utterance   TEXT,
            action      TEXT,
            validated   INTEGER,
            latency_ms  REAL
        )
    """)
    con.commit()
    return con


def log_turn(session_id: str, utterance: str, action, validated: bool, latency_ms: float):
    """Call this after every pipeline turn."""
    action_str = None
    if isinstance(action, dict):
        action_str = action.get("action")
    elif action is not None:
        action_str = str(action)

    with _conn() as con:
        con.execute(
            "INSERT INTO turns (ts, session_id, utterance, action, validated, latency_ms) "
            "VALUES (?,?,?,?,?,?)",
            (time.time(), session_id, utterance, action_str, int(validated), latency_ms),
        )


def get_metrics() -> dict:
    """
    Returns per-action counts and overall accuracy, latency P50/P95,
    and the last 20 turns for the dashboard.
    """
    with _conn() as con:
        rows = con.execute("SELECT * FROM turns ORDER BY ts DESC").fetchall()

    if not rows:
        return {"total": 0, "accuracy": 0, "latency_p50": 0, "latency_p95": 0,
                "per_action": {}, "recent": []}

    total     = len(rows)
    validated = sum(1 for r in rows if r["validated"])
    accuracy  = round(validated / total * 100, 1) if total else 0

    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    per_action = {}
    for a in ACTION_TYPES:
        a_rows = [r for r in rows if r["action"] == a]
        correct = sum(1 for r in a_rows if r["validated"])
        per_action[a] = {
            "count":    len(a_rows),
            "correct":  correct,
            "accuracy": round(correct / len(a_rows) * 100, 1) if a_rows else None,
        }

    recent = [
        {
            "ts":         r["ts"],
            "session_id": r["session_id"],
            "utterance":  r["utterance"],
            "action":     r["action"],
            "validated":  bool(r["validated"]),
            "latency_ms": r["latency_ms"],
        }
        for r in rows[:20]
    ]

    return {
        "total":       total,
        "validated":   validated,
        "accuracy":    accuracy,
        "latency_p50": round(p50, 1),
        "latency_p95": round(p95, 1),
        "per_action":  per_action,
        "recent":      recent,
    }


def clear_metrics():
    """Wipe all logged turns (useful for a fresh demo run)."""
    with _conn() as con:
        con.execute("DELETE FROM turns")
    print("[metrics] Cleared.")


if __name__ == "__main__":
    import json
    print(json.dumps(get_metrics(), indent=2))
