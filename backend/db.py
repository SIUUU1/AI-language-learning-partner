"""
db.py — 애플리케이션 SQLite 저장소
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
CREATE TABLE IF NOT EXISTS users (
    email              TEXT PRIMARY KEY,
    password_hash      TEXT NOT NULL,
    verified           INTEGER DEFAULT 0,
    verification_code  TEXT,
    code_expires_at    REAL,
    created_at         REAL
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(APP_DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)


# ── 세션 ────────────────────────────────────────────────
def save_session(s: Dict) -> None:
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id,user_id,video_id,video_title,native_language,target_language,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (s["session_id"], s["user_id"], s.get("video_id"), s.get("video_title"),
             s.get("native_language"), s.get("target_language"), time.time()),
        )


# ── 표현 ────────────────────────────────────────────────
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


# ── 퀴즈 ────────────────────────────────────────────────
def save_quiz_result(session_id: str, user_id: str, score: int, total: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO quiz_results (session_id,user_id,score,total,created_at) VALUES (?,?,?,?,?)",
            (session_id, user_id, score, total, time.time()),
        )


# ── 복습 ────────────────────────────────────────────────
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


def all_reviews(user_id: str, limit: int = 100) -> List[Dict]:
    """복습 예정 + 완료 이력을 함께 (사이드바 '이전 학습 복습' 용)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM reviews WHERE user_id=? ORDER BY done ASC, due_at ASC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_review_done(review_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE reviews SET done=1 WHERE id=?", (review_id,))


# ── 영상별 플래시카드 (사이드바 '플래시카드로 복습') ──────
def expressions_by_video(user_id: str, limit: int = 200) -> List[Dict]:
    """사용자가 학습한 표현을 어떤 유튜브 영상 검색(세션)에서 나왔는지와 함께 반환한다.
    프런트에서 '영상 제목 + 검색 날짜시간' 기준으로 묶어 접힌 플래시카드로 보여준다."""
    with _conn() as c:
        rows = c.execute(
            """SELECT e.expression, e.meaning, e.example, s.video_title, s.session_id,
                      s.created_at AS session_created_at
               FROM expressions e
               JOIN sessions s ON e.session_id = s.session_id
               WHERE e.user_id=?
               ORDER BY s.created_at DESC, e.created_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ── 이력 ────────────────────────────────────────────────
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


# ── 사용자 계정 (이메일 + 비밀번호 + 이메일 인증) ─────────
def upsert_pending_user(email: str, password_hash: str, code: str, expires_at: float) -> None:
    """회원가입/재발송 시 사용. 아직 인증 전이면 덮어써서 최신 코드로 갱신한다."""
    with _conn() as c:
        c.execute(
            """INSERT INTO users (email,password_hash,verified,verification_code,code_expires_at,created_at)
               VALUES (?,?,0,?,?,?)
               ON CONFLICT(email) DO UPDATE SET
                 password_hash=excluded.password_hash,
                 verification_code=excluded.verification_code,
                 code_expires_at=excluded.code_expires_at
               WHERE users.verified=0""",
            (email, password_hash, code, expires_at, time.time()),
        )


def get_user(email: str) -> Optional[Dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def mark_user_verified(email: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET verified=1, verification_code=NULL, code_expires_at=NULL WHERE email=?",
            (email,),
        )
