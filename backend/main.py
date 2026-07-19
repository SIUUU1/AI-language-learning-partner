"""
main.py — FastAPI

  POST /youtube/search     검색 (YouTube Data API)
  POST /session/analyze    영상 선택 → ContentAnalyzerAgent → 표현 + 퀴즈 생성/저장
  POST /quiz/grade         QuizMasterAgent 채점 → 저장
  POST /roleplay/turn      RoleplayPartnerAgent 한 턴
  POST /session/feedback   FeedbackCoachAgent 피드백 + 복습/플래시카드 저장
  GET  /user/{id}/history  학습 이력 (SQLite)
  GET  /user/{id}/reviews  복습 예정 (SQLite)
  GET  /user/{id}/stats    통계
  GET  /health             모드/상태
"""
from __future__ import annotations

import base64
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from . import auth
from . import db
from .agents import analyzer, feedback_coach, quiz_master, roleplay_partner
from .config import mode_banner
from .state import PERSONAS
from . import tts
from . import stt
from .youtube_service import (
    TranscriptUnavailableError, extract_video_id, fetch_transcript,
    search_videos, video_metadata,
)
from langchain_core.messages import AIMessage, HumanMessage

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = FastAPI(title="LinguaLoop API", version="3.0.0")

# UI 상호작용용 세션 저장소 (경량; 영속 데이터는 SQLite/Chroma)
_SESSIONS: Dict[str, Dict] = {}

# 스키마는 import 시점에 보장 (TestClient/uvicorn 어느 쪽이든 안전)
db.init_db()


# ── 스키마 ──────────────────────────────────────────────
class SearchReq(BaseModel):
    query: str
    max_results: int = 6


class AnalyzeReq(BaseModel):
    user_id: str
    url_or_id: str
    native_language: str = "한국어"
    target_language: str = "English"
    persona: str = "friend"
    max_turns: int = 3   # 학습자가 원하는 역할극 턴 수 (기술적 상한은 없지만 UX상 1~10 권장)


class GradeReq(BaseModel):
    session_id: str
    answers: List[str]


class RoleplayReq(BaseModel):
    session_id: str
    message: str
    speak: bool = True         # gpt-4o-mini-tts 로 파트너 음성 생성 여부


class RoleplayStartReq(BaseModel):
    session_id: str
    speak: bool = True


class TTSReq(BaseModel):
    text: str
    persona: str = "friend"
    speed: float = 1.0


class FeedbackReq(BaseModel):
    session_id: str
    mode: str = "roleplay"     # "roleplay" | "flashcards"


class SignupReq(BaseModel):
    email: str
    password: str


class VerifyReq(BaseModel):
    email: str
    code: str


class LoginReq(BaseModel):
    email: str
    password: str


class ResendCodeReq(BaseModel):
    email: str


# ── 엔드포인트 ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "LinguaLoop API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": mode_banner(),
        "tts_available": tts.available(),
        "stt_available": stt.available(),
        "email_available": auth.EMAIL_SENDING_CONFIGURED,
    }


# ── 인증: 이메일 + 비밀번호 회원가입/로그인 (이메일 인증 포함) ──
@app.post("/auth/signup")
def auth_signup(req: SignupReq):
    email = req.email.strip().lower()
    if not _EMAIL_RE.match(email):
        return {"ok": False, "message": "올바른 이메일 형식이 아니에요."}

    existing = db.get_user(email)
    if existing and existing.get("verified"):
        return {"ok": False, "message": "이미 가입된 이메일이에요. 로그인해 주세요."}

    strong, msg = auth.password_is_strong_enough(req.password)
    if not strong:
        return {"ok": False, "message": msg}

    code = auth.generate_verification_code()
    expires_at = time.time() + auth.CODE_TTL_SECONDS
    db.upsert_pending_user(email, auth.hash_password(req.password), code, expires_at)

    sent = auth.send_verification_email(email, code)
    resp = {"ok": True, "sent": sent,
           "message": "인증 코드를 이메일로 보냈어요." if sent
                      else "이메일 발송이 설정되지 않아, 데모용으로 코드를 화면에 보여드려요."}
    if not sent:
        resp["dev_code"] = code   # SMTP 미설정 시에만 — 데모 폴백 (실발송 시엔 절대 노출 안 함)
    return resp


@app.post("/auth/resend")
def auth_resend(req: ResendCodeReq):
    email = req.email.strip().lower()
    user = db.get_user(email)
    if not user or user.get("verified"):
        return {"ok": False, "message": "재발송할 가입 대기 계정을 찾을 수 없어요."}
    code = auth.generate_verification_code()
    expires_at = time.time() + auth.CODE_TTL_SECONDS
    db.upsert_pending_user(email, user["password_hash"], code, expires_at)
    sent = auth.send_verification_email(email, code)
    resp = {"ok": True, "sent": sent}
    if not sent:
        resp["dev_code"] = code
    return resp


@app.post("/auth/verify")
def auth_verify(req: VerifyReq):
    email = req.email.strip().lower()
    user = db.get_user(email)
    if not user:
        return {"ok": False, "message": "가입 정보를 찾을 수 없어요."}
    if user.get("verified"):
        return {"ok": True, "message": "이미 인증된 계정이에요."}
    if not user.get("verification_code") or user["verification_code"] != req.code.strip():
        return {"ok": False, "message": "인증 코드가 올바르지 않아요."}
    if user.get("code_expires_at") and time.time() > user["code_expires_at"]:
        return {"ok": False, "message": "인증 코드가 만료됐어요. 다시 받아 주세요."}
    db.mark_user_verified(email)
    return {"ok": True, "message": "이메일 인증이 완료됐어요!"}


@app.post("/auth/login")
def auth_login(req: LoginReq):
    email = req.email.strip().lower()
    user = db.get_user(email)
    if not user:
        return {"ok": False, "message": "가입되지 않은 이메일이에요."}
    if not user.get("verified"):
        return {"ok": False, "message": "이메일 인증이 아직 안 됐어요. 인증을 먼저 완료해 주세요.",
               "needs_verification": True}
    if not auth.verify_password(req.password, user["password_hash"]):
        return {"ok": False, "message": "비밀번호가 올바르지 않아요."}
    return {"ok": True, "email": email}


@app.post("/youtube/search")
def youtube_search(req: SearchReq):
    return {"results": search_videos(req.query, req.max_results)}


@app.post("/session/analyze")
def analyze(req: AnalyzeReq):
    vid = extract_video_id(req.url_or_id)
    meta = video_metadata(vid)

    # 1순위 공식 자막 → 2순위 Whisper STT → 3순위 완전 실패 시 다른 URL 안내
    try:
        result = fetch_transcript(vid, req.target_language)
    except TranscriptUnavailableError as e:
        return {
            "error": "video_unavailable",
            "message": (f"'{meta.get('title', vid)}' 영상에서 학습 콘텐츠를 만들 수 없었어요. "
                       "자막도 없고 음성 인식(Whisper)도 실패했어요. "
                       "다른 유튜브 영상 URL을 입력해 주세요."),
        }
    transcript = result["text"]
    transcript_source = result["source"]

    session_id = uuid.uuid4().hex[:12]

    state = {
        "user_id": req.user_id, "session_id": session_id,
        "native_language": req.native_language, "target_language": req.target_language,
        "video_id": vid, "video_title": meta.get("title", ""),
        "transcript": transcript, "persona": req.persona,
        "messages": [], "turn_count": 0,
        "max_turns": max(1, min(req.max_turns, 10)),   # 1~10 사이로 안전하게 clamp
        "key_expressions": [], "study_history": [], "stage": "start",
    }

    # 1) ContentAnalyzerAgent
    analyzer_out = analyzer.run(state)
    state.update(analyzer_out)
    # 2) QuizMasterAgent (생성만; 채점은 사용자가 답한 뒤)
    quiz_out = quiz_master.generate(state)
    state.update(quiz_out)

    # GPT 호출이 실패해 mock으로 대체된 경우, 절대 조용히 넘어가지 않고 사용자에게 노출한다
    llm_warnings = [w for w in (analyzer_out.get("llm_warning"), quiz_out.get("llm_warning")) if w]

    # 영속 저장
    db.save_session(state)
    db.save_expressions(session_id, req.user_id, state["enriched_expressions"])
    db.add_history(req.user_id, state["study_history"])
    _SESSIONS[session_id] = state

    return {
        "session_id": session_id,
        "video_id": vid,
        "video_title": meta.get("title", ""),
        "transcript_preview": transcript[:300],
        "transcript_source": transcript_source,   # "captions" | "whisper" | "sample"
        "expressions": state["enriched_expressions"],
        "new_expression_count": state.get("new_expression_count", 0),
        "quiz": state["quiz"],
        "llm_warnings": llm_warnings,             # 비어있지 않으면 UI가 반드시 표시해야 함
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


@app.post("/roleplay/start")
def roleplay_start(req: RoleplayStartReq):
    """역할극을 AI 파트너가 먼저 시작 (오프닝 인사). 학습자 턴 수는 늘리지 않는다."""
    state = _SESSIONS.get(req.session_id)
    if not state:
        return {"error": "session not found"}
    upd = roleplay_partner.start(state)
    state.setdefault("messages", [])
    state["messages"].extend(upd["messages"])
    partner = upd["messages"][-1].content if upd["messages"] else ""

    audio_b64 = None
    if req.speak and partner:
        audio = tts.synthesize(partner, persona=state.get("persona", "friend"))
        if audio:
            audio_b64 = base64.b64encode(audio).decode()

    return {"partner_reply": partner, "turn": state.get("turn_count", 0),
            "max_turns": state.get("max_turns", 3),
            "audio_b64": audio_b64, "audio_ai_generated": True,
            "llm_warning": upd.get("llm_warning")}


@app.post("/roleplay/turn")
def roleplay_turn(req: RoleplayReq):
    state = _SESSIONS.get(req.session_id)
    if not state:
        return {"error": "session not found"}
    upd = roleplay_partner.reply(state, req.message)
    # 세션 메시지 누적 (add_messages 리듀서가 아니므로 수동 누적)
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
            "audio_b64": audio_b64, "audio_ai_generated": True,
            "llm_warning": upd.get("llm_warning")}


@app.post("/tts")
def synthesize_tts(req: TTSReq):
    """임의 텍스트 → 음성 (예문 듣기 등 재사용). 키 없으면 audio_b64=None."""
    audio = tts.synthesize(req.text, persona=req.persona, speed=req.speed)
    return {
        "audio_b64": base64.b64encode(audio).decode() if audio else None,
        "available": tts.available(),
        "audio_ai_generated": True,
    }


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...),
                         language: str = Form("English"),
                         turn_index: int = Form(0)):
    """마이크 녹음(오디오 파일) → 받아쓴 텍스트. 키 없으면 데모용 예시 발화를 순서대로 반환."""
    audio_bytes = await file.read()
    text = stt.transcribe(audio_bytes, filename=file.filename or "speech.wav",
                          language=language, turn_index=turn_index)
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
        "corrections": upd.get("corrections", []),
        "llm_warning": upd.get("llm_warning"),
    }


@app.get("/user/{user_id}/history")
def user_history(user_id: str):
    return {"history": db.get_history(user_id)}


@app.get("/user/{user_id}/reviews")
def user_reviews(user_id: str):
    return {"reviews": db.due_reviews(user_id)}


@app.get("/user/{user_id}/reviews/all")
def user_reviews_all(user_id: str):
    """완료된 것 포함 전체 복습 이력 (사이드바 '이전 학습 복습하기' 용)."""
    return {"reviews": db.all_reviews(user_id)}


@app.post("/reviews/{review_id}/done")
def complete_review(review_id: int):
    db.mark_review_done(review_id)
    return {"ok": True}


@app.get("/user/{user_id}/stats")
def user_stats(user_id: str):
    return db.user_stats(user_id)


@app.get("/user/{user_id}/flashcards")
def user_flashcards(user_id: str):
    """지금까지 학습한 표현을 '영상 제목 + 검색 날짜시간'(세션) 별로 묶어서 반환한다.
    같은 영상을 다른 날 다시 검색했으면 별도 그룹으로 나뉜다.
    (사이드바 '🃏 플래시카드로 복습'에서 사용 — 세션이 끝난 뒤에도 계속 복습 가능)"""
    rows = db.expressions_by_video(user_id)
    grouped: Dict[str, Dict] = {}   # session_id -> {video_title, searched_at, cards}
    for r in rows:
        sid = r["session_id"]
        if sid not in grouped:
            ts = r.get("session_created_at")
            searched_at = (datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                          if ts else "")
            grouped[sid] = {
                "session_id": sid,
                "video_title": r.get("video_title") or "(제목 없음)",
                "searched_at": searched_at,
                "cards": [],
            }
        card = {"expression": r["expression"], "meaning": r["meaning"], "example": r["example"]}
        if card not in grouped[sid]["cards"]:   # 같은 세션 재분석 등으로 인한 중복 방지
            grouped[sid]["cards"].append(card)
    # 이미 세션 created_at DESC 로 정렬돼서 나왔으므로 dict 삽입 순서 그대로 최신순
    return {"flashcards_by_video": list(grouped.values())}
