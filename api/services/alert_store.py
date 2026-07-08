"""
SQLite-backed alert log. Simple, dependency-free, laptop-friendly.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "models_store" / "alerts.db"

_lock = Lock()


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                src_ip TEXT,
                dst_ip TEXT,
                predicted_category TEXT,
                confidence REAL,
                severity TEXT,
                explanation TEXT
            )
            """
        )
        # Idempotent migration: add explanation column to pre-existing databases.
        try:
            conn.execute("ALTER TABLE alerts ADD COLUMN explanation TEXT")
        except Exception:
            pass  # Column already exists — safe to ignore.
        conn.commit()
        conn.close()


def log_alert(src_ip: str, dst_ip: str, predicted_category: str,
              confidence: float, severity: str):
    with _lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO alerts (timestamp, src_ip, dst_ip, predicted_category,
               confidence, severity) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                src_ip, dst_ip, predicted_category, confidence, severity,
            ),
        )
        conn.commit()
        conn.close()


def get_recent_alerts(limit: int = 100):
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def get_alert_stats():
    with _lock:
        conn = _get_conn()
        total = conn.execute("SELECT COUNT(*) as c FROM alerts").fetchone()["c"]
        by_category = conn.execute(
            "SELECT predicted_category, COUNT(*) as c FROM alerts GROUP BY predicted_category"
        ).fetchall()
        by_severity = conn.execute(
            "SELECT severity, COUNT(*) as c FROM alerts GROUP BY severity"
        ).fetchall()
        top_sources = conn.execute(
            """SELECT src_ip, COUNT(*) as c FROM alerts
               WHERE predicted_category != 'Normal'
               GROUP BY src_ip ORDER BY c DESC LIMIT 10"""
        ).fetchall()
        conn.close()
        return {
            "total_alerts": total,
            "by_category": {r["predicted_category"]: r["c"] for r in by_category},
            "by_severity": {r["severity"]: r["c"] for r in by_severity},
            "top_sources": [dict(r) for r in top_sources],
        }


def get_alert_by_id(alert_id: int) -> dict | None:
    """Return a single alert row by primary key, or None if not found."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None


def set_alert_explanation(alert_id: int, explanation: str) -> None:
    """Persist an LLM-generated explanation for a given alert (cache)."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE alerts SET explanation = ? WHERE id = ?",
            (explanation, alert_id),
        )
        conn.commit()
        conn.close()


def clear_alerts():
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM alerts")
        conn.commit()
        conn.close()
