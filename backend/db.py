"""
db.py — Application SQLite Storage

Two types of SQLite are used:
  1) db.py: "Application DB" that stores learning sessions, representations, quiz results, reviews, and history.
  2) SqliteSaver in graph.py: "Checkpoint DB" for LangGraph state.

"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Dict, List, Optional

from .config import APP_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT,
    video_id     TEXT,
    video_title  TEXT,
    native_language TEXT,
    target_language TEXT,
    created_at   REAL
);
CREATE TABLE IF NOT EXISTS expressions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    user_id      TEXT,
    expression   TEXT,
    meaning      TEXT,
    example      TEXT,
    is_new       INTEGER DEFAULT 1,
    created_at   REAL
);
CREATE TABLE IF NOT EXISTS quiz_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    user_id      TEXT,
    score        INTEGER,
    total        INTEGER,
    created_at   REAL
);
CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT,
    expression   TEXT,
    meaning      TEXT,
    review_after_days INTEGER,
    due_at       REAL,
    done         INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT,
    event        TEXT,
    created_at   REAL
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(APP_DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)


# ── session ────────────────────────────────────────────────
def save_session(s: Dict) -> None:
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id,user_id,video_id,video_title,native_language,target_language,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (s["session_id"], s["user_id"], s.get("video_id"), s.get("video_title"),
             s.get("native_language"), s.get("target_language"), time.time()),
        )


# ── expressions ────────────────────────────────────────────────
def save_expressions(session_id: str, user_id: str, expressions: List[Dict]) -> None:
    with _conn() as c:
        for e in expressions:
            c.execute(
                """INSERT INTO expressions
                   (session_id,user_id,expression,meaning,example,is_new,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (session_id, user_id, e.get("expression"), e.get("meaning"),
                 e.get("example"), int(e.get("is_new", True)), time.time()),
            )


# ── quiz ────────────────────────────────────────────────
def save_quiz_result(session_id: str, user_id: str, score: int, total: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO quiz_results (session_id,user_id,score,total,created_at) VALUES (?,?,?,?,?)",
            (session_id, user_id, score, total, time.time()),
        )


# ── reviews ────────────────────────────────────────────────
def save_reviews(user_id: str, reviews: List[Dict]) -> None:
    now = time.time()
    with _conn() as c:
        for r in reviews:
            days = int(r.get("review_after_days", 1))
            c.execute(
                """INSERT INTO reviews
                   (user_id,expression,meaning,review_after_days,due_at,done)
                   VALUES (?,?,?,?,?,0)""",
                (user_id, r.get("expression"), r.get("meaning"), days,
                 now + days * 86400),
            )


def due_reviews(user_id: str) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM reviews WHERE user_id=? AND done=0 ORDER BY due_at ASC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── history ────────────────────────────────────────────────
def add_history(user_id: str, events: List[str]) -> None:
    with _conn() as c:
        for ev in events:
            c.execute("INSERT INTO history (user_id,event,created_at) VALUES (?,?,?)",
                      (user_id, ev, time.time()))


def get_history(user_id: str, limit: int = 50) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT event,created_at FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def user_stats(user_id: str) -> Dict:
    with _conn() as c:
        expr = c.execute("SELECT COUNT(*) n FROM expressions WHERE user_id=?", (user_id,)).fetchone()["n"]
        quiz = c.execute("SELECT COUNT(*) n FROM quiz_results WHERE user_id=?", (user_id,)).fetchone()["n"]
        rev = c.execute("SELECT COUNT(*) n FROM reviews WHERE user_id=? AND done=0", (user_id,)).fetchone()["n"]
    return {"expressions_learned": expr, "quizzes_taken": quiz, "reviews_due": rev}
