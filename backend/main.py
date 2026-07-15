"""
main.py — FastAPI Backend

The Streamlit UI interacts solely with this REST API. Each endpoint utilizes a specialized agent.

  POST /youtube/search     Search (YouTube Data API)
  POST /session/analyze    Select video → ContentAnalyzerAgent → Generate/save expressions & quizzes
  POST /quiz/grade         QuizMasterAgent grades → Save
  POST /roleplay/turn      RoleplayPartnerAgent executes a turn
  POST /session/feedback   FeedbackCoachAgent provides feedback → Save review items/flashcards
  GET  /user/{id}/history  Learning history (SQLite)
  GET  /user/{id}/reviews  Scheduled reviews (SQLite)
  GET  /user/{id}/stats    Statistics
  GET  /health             Mode/Status
"""
from __future__ import annotations

import base64
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from . import db
from .agents import analyzer, feedback_coach, quiz_master, roleplay_partner
from .config import mode_banner
from .state import PERSONAS
from . import tts
from . import stt
from .youtube_service import fetch_transcript, search_videos, video_metadata, extract_video_id
from langchain_core.messages import AIMessage, HumanMessage

app = FastAPI(title="LinguaLoop API", version="3.0.0")

# Session storage for UI interactions (lightweight; persistent data in SQLite/Chroma)
_SESSIONS: Dict[str, Dict] = {}

# The schema is guaranteed at import time (safe for both TestClient and uvicorn).
db.init_db()


# ── Schema ──────────────────────────────────────────────
class SearchReq(BaseModel):
    query: str
    max_results: int = 6


class AnalyzeReq(BaseModel):
    user_id: str
    url_or_id: str
    native_language: str = "한국어"
    target_language: str = "English"
    persona: str = "friend"


class GradeReq(BaseModel):
    session_id: str
    answers: List[str]


class RoleplayReq(BaseModel):
    session_id: str
    message: str
    speak: bool = True         # Whether partner voices are generated using gpt-4o-mini-tts


class TTSReq(BaseModel):
    text: str
    persona: str = "friend"
    speed: float = 1.0


class FeedbackReq(BaseModel):
    session_id: str
    mode: str = "roleplay"     # "roleplay" | "flashcards"


# ── Endpoint ──────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "mode": mode_banner(),
            "personas": list(PERSONAS.keys()),
            "tts_available": tts.available(),
            "stt_available": stt.available()}


@app.post("/youtube/search")
def youtube_search(req: SearchReq):
    return {"results": search_videos(req.query, req.max_results)}


@app.post("/session/analyze")
def analyze(req: AnalyzeReq):
    vid = extract_video_id(req.url_or_id)
    meta = video_metadata(vid)
    transcript = fetch_transcript(vid, req.target_language)
    session_id = uuid.uuid4().hex[:12]

    state = {
        "user_id": req.user_id, "session_id": session_id,
        "native_language": req.native_language, "target_language": req.target_language,
        "video_id": vid, "video_title": meta.get("title", ""),
        "transcript": transcript, "persona": req.persona,
        "messages": [], "turn_count": 0, "max_turns": 3,
        "key_expressions": [], "study_history": [], "stage": "start",
    }

    # 1) ContentAnalyzerAgent
    state.update(analyzer.run(state))
    # 2) QuizMasterAgent (Generation only; grading takes place after the user provides an answer)
    state.update(quiz_master.generate(state))

    # Persistent storage
    db.save_session(state)
    db.save_expressions(session_id, req.user_id, state["enriched_expressions"])
    db.add_history(req.user_id, state["study_history"])
    _SESSIONS[session_id] = state

    return {
        "session_id": session_id,
        "video_title": meta.get("title", ""),
        "transcript_preview": transcript[:300],
        "expressions": state["enriched_expressions"],
        "new_expression_count": state.get("new_expression_count", 0),
        "quiz": state["quiz"],
    }


@app.post("/quiz/grade")
def quiz_grade(req: GradeReq):
    state = _SESSIONS.get(req.session_id)
    if not state:
        return {"error": "session not found"}
    state["quiz_answers"] = req.answers
    state.update(quiz_master.grade(state))
    db.save_quiz_result(req.session_id, state["user_id"],
                        state["quiz_score"], len(state["quiz"]))
    db.add_history(state["user_id"], [f"📝 quiz {state['quiz_score']}/{len(state['quiz'])}"])
    return {"score": state["quiz_score"], "total": len(state["quiz"]),
            "quiz": state["quiz"]}


@app.post("/roleplay/turn")
def roleplay_turn(req: RoleplayReq):
    state = _SESSIONS.get(req.session_id)
    if not state:
        return {"error": "session not found"}
    upd = roleplay_partner.reply(state, req.message)
    # Accumulating session messages (manual accumulation, as `add_messages` is not a reducer)
    state.setdefault("messages", [])
    state["messages"].extend(upd["messages"])
    state["turn_count"] = upd["turn_count"]
    partner = upd["messages"][-1].content

    audio_b64 = None
    if req.speak:
        audio = tts.synthesize(partner, persona=state.get("persona", "friend"))
        if audio:
            audio_b64 = base64.b64encode(audio).decode()

    return {"partner_reply": partner, "turn": state["turn_count"],
            "max_turns": state.get("max_turns", 3),
            "audio_b64": audio_b64, "audio_ai_generated": True}


@app.post("/tts")
def synthesize_tts(req: TTSReq):
    """Arbitrary text → Speech (reusable, e.g., for listening to example sentences). If no key is provided, `audio_b64=None`."""
    audio = tts.synthesize(req.text, persona=req.persona, speed=req.speed)
    return {
        "audio_b64": base64.b64encode(audio).decode() if audio else None,
        "available": tts.available(),
        "audio_ai_generated": True,
    }


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...),
                         language: str = Form("English")):
    """Microphone recording (audio file) → Transcribed text. Returns `text=None` if no key is provided."""
    audio_bytes = await file.read()
    text = stt.transcribe(audio_bytes, filename=file.filename or "speech.wav",
                          language=language)
    return {"text": text, "available": stt.available()}


@app.post("/session/feedback")
def session_feedback(req: FeedbackReq):
    state = _SESSIONS.get(req.session_id)
    if not state:
        return {"error": "session not found"}
    state["practice_mode"] = req.mode
    upd = feedback_coach.run(state)
    state.update(upd)
    db.add_history(state["user_id"], upd["study_history"])
    if upd.get("review_list"):
        db.save_reviews(state["user_id"], upd["review_list"])
    return {
        "feedback": upd.get("feedback", ""),
        "review_list": upd.get("review_list", []),
        "flashcards": upd.get("flashcards", []),
    }


@app.get("/user/{user_id}/history")
def user_history(user_id: str):
    return {"history": db.get_history(user_id)}


@app.get("/user/{user_id}/reviews")
def user_reviews(user_id: str):
    return {"reviews": db.due_reviews(user_id)}


@app.get("/user/{user_id}/stats")
def user_stats(user_id: str):
    return db.user_stats(user_id)
