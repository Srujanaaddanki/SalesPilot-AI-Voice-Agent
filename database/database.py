"""SQLite persistence layer for SalesPilot AI CRM, conversations, appointments, and summaries."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.getenv("SALESPILOT_DB_PATH", "database/salespilot.db")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class DatabaseManager:
    """Thread-safe SQLite manager for leads, appointments, conversations, and call summaries."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create required CRM and conversation tables if they do not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    name TEXT DEFAULT '',
                    email TEXT DEFAULT '',
                    phone TEXT DEFAULT '',
                    product_interest TEXT DEFAULT '',
                    budget TEXT DEFAULT '',
                    business_requirement TEXT DEFAULT '',
                    purchase_timeline TEXT DEFAULT '',
                    sentiment TEXT DEFAULT 'Neutral',
                    sentiment_score REAL DEFAULT 0.0,
                    lead_score INTEGER DEFAULT 0,
                    qualification_status TEXT DEFAULT 'New',
                    language TEXT DEFAULT 'English',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    customer_name TEXT NOT NULL,
                    contact_details TEXT NOT NULL,
                    preferred_date TEXT NOT NULL,
                    preferred_time TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    status TEXT DEFAULT 'Requested (Pending Confirmation)',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    language TEXT DEFAULT 'English',
                    sentiment TEXT DEFAULT 'Neutral',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS call_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    customer_requirements TEXT DEFAULT '',
                    product_interest TEXT DEFAULT '',
                    budget TEXT DEFAULT '',
                    purchase_timeline TEXT DEFAULT '',
                    sentiment TEXT DEFAULT 'Neutral',
                    qualification_status TEXT DEFAULT 'In Progress',
                    lead_score INTEGER DEFAULT 0,
                    recommended_next_action TEXT DEFAULT '',
                    summary_text TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )

    def upsert_lead(self, session_id: str, **fields: Any) -> Dict[str, Any]:
        """Insert or update a lead record for the given session_id."""
        now = _utc_now()
        allowed_cols = {
            "name",
            "email",
            "phone",
            "product_interest",
            "budget",
            "business_requirement",
            "purchase_timeline",
            "sentiment",
            "sentiment_score",
            "lead_score",
            "qualification_status",
            "language",
        }
        clean_fields = {k: v for k, v in fields.items() if k in allowed_cols and v is not None}

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM leads WHERE session_id = ?", (session_id,)
            ).fetchone()

            if existing is None:
                cols = ["session_id", "created_at", "updated_at"] + list(clean_fields.keys())
                vals = [session_id, now, now] + list(clean_fields.values())
                placeholders = ", ".join(["?"] * len(cols))
                conn.execute(
                    f"INSERT INTO leads ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
            elif clean_fields:
                assignments = ", ".join([f"{k} = ?" for k in clean_fields])
                vals = list(clean_fields.values()) + [now, session_id]
                conn.execute(
                    f"UPDATE leads SET {assignments}, updated_at = ? WHERE session_id = ?",
                    vals,
                )

            row = conn.execute(
                "SELECT * FROM leads WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else {}

    def get_lead_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM leads WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_leads(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM leads ORDER BY updated_at DESC, id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def save_message(
        self,
        session_id: str,
        role: str,
        message: str,
        language: str = "English",
        sentiment: str = "Neutral",
    ) -> int:
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO conversations (session_id, role, message, language, sentiment, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, message, language, sentiment, now),
            )
            return int(cur.lastrowid)

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_conversations(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_sessions(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id FROM conversations
                GROUP BY session_id
                ORDER BY MAX(id) DESC
                """
            ).fetchall()
            return [r["session_id"] for r in rows]

    def create_appointment(
        self,
        session_id: str,
        customer_name: str,
        contact_details: str,
        preferred_date: str,
        preferred_time: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        now = _utc_now()
        status = "Requested (Pending Confirmation)"
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO appointments (
                    session_id, customer_name, contact_details,
                    preferred_date, preferred_time, notes, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    customer_name.strip(),
                    contact_details.strip(),
                    preferred_date.strip(),
                    preferred_time.strip(),
                    notes.strip(),
                    status,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM appointments WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            return dict(row) if row else {}

    def get_all_appointments(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM appointments ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def save_call_summary(self, session_id: str, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO call_summaries (
                    session_id, customer_requirements, product_interest,
                    budget, purchase_timeline, sentiment, qualification_status,
                    lead_score, recommended_next_action, summary_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    customer_requirements = excluded.customer_requirements,
                    product_interest = excluded.product_interest,
                    budget = excluded.budget,
                    purchase_timeline = excluded.purchase_timeline,
                    sentiment = excluded.sentiment,
                    qualification_status = excluded.qualification_status,
                    lead_score = excluded.lead_score,
                    recommended_next_action = excluded.recommended_next_action,
                    summary_text = excluded.summary_text,
                    created_at = excluded.created_at
                """,
                (
                    session_id,
                    summary_data.get("customer_requirements", ""),
                    summary_data.get("product_interest", ""),
                    summary_data.get("budget", ""),
                    summary_data.get("purchase_timeline", ""),
                    summary_data.get("sentiment", "Neutral"),
                    summary_data.get("qualification_status", "In Progress"),
                    int(summary_data.get("lead_score", 0)),
                    summary_data.get("recommended_next_action", ""),
                    summary_data.get("summary_text", ""),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM call_summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else {}

    def get_call_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM call_summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM call_summaries ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
            qualified_leads = conn.execute(
                "SELECT COUNT(*) FROM leads WHERE qualification_status IN ('Qualified', 'Hot')"
            ).fetchone()[0]
            total_appointments = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
            total_summaries = conn.execute("SELECT COUNT(*) FROM call_summaries").fetchone()[0]
            total_messages = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            sentiment_rows = conn.execute(
                "SELECT sentiment, COUNT(*) AS cnt FROM leads GROUP BY sentiment"
            ).fetchall()
            status_rows = conn.execute(
                "SELECT qualification_status, COUNT(*) AS cnt FROM leads GROUP BY qualification_status"
            ).fetchall()

        return {
            "total_leads": total_leads,
            "qualified_leads": qualified_leads,
            "total_appointments": total_appointments,
            "total_summaries": total_summaries,
            "total_messages": total_messages,
            "sentiment_breakdown": {r["sentiment"]: r["cnt"] for r in sentiment_rows},
            "status_breakdown": {r["qualification_status"]: r["cnt"] for r in status_rows},
        }
