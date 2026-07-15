"""
test_nodes.py — 전문 에이전트 & 그래프 & API 노드 테스트 (PyTest)

mock 모드(키 없음)에서도 전 과정이 결정적으로 통과하도록 작성했다.
실행:  pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.messages import HumanMessage

from backend.agents import analyzer, quiz_master, roleplay_partner, feedback_coach
from backend.graph import build_graph
from backend.youtube_service import SAMPLE_TRANSCRIPT, extract_video_id


BASE = {
    "user_id": "pytest-user", "session_id": "s",
    "native_language": "한국어", "target_language": "English",
    "transcript": SAMPLE_TRANSCRIPT, "persona": "barista",
    "messages": [], "turn_count": 0, "max_turns": 3,
    "key_expressions": [], "study_history": [], "stage": "start",
}


# ── 유틸 ────────────────────────────────────────────────
def test_extract_video_id_from_url():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


# ── 에이전트 1: ContentAnalyzer ─────────────────────────
def test_analyzer_extracts_and_enriches():
    out = analyzer.run(dict(BASE))
    assert len(out["key_expressions"]) >= 3
    assert len(out["enriched_expressions"]) == len(out["key_expressions"])
    first = out["enriched_expressions"][0]
    assert "definition" in first and "extra_examples" in first and "is_new" in first
    assert out["stage"] == "analyzed"


# ── 에이전트 2: QuizMaster ──────────────────────────────
def test_quiz_generate_and_grade():
    st = dict(BASE)
    st.update(analyzer.run(st))
    st.update(quiz_master.generate(st))
    assert len(st["quiz"]) == 3

    st["quiz_answers"] = [q["answer"] for q in st["quiz"]]  # 전부 정답
    graded = quiz_master.grade(st)
    assert graded["quiz_score"] == len(st["quiz"])

    st["quiz_answers"] = ["___wrong___"] * len(st["quiz"])
    assert quiz_master.grade(st)["quiz_score"] == 0


# ── 에이전트 3: RoleplayPartner ─────────────────────────
def test_roleplay_returns_partner_reply():
    st = dict(BASE)
    st.update(analyzer.run(st))
    out = roleplay_partner.reply(st, "Hi, I'll have a latte please.")
    assert out["turn_count"] == 1
    assert len(out["messages"]) == 2          # 학습자 + 파트너
    assert out["messages"][-1].content        # 파트너 응답 존재


# ── 에이전트 4: FeedbackCoach ───────────────────────────
def test_feedback_detects_used_expressions():
    st = dict(BASE)
    st.update(analyzer.run(st))
    st["messages"] = [HumanMessage(content="I'll have a latte, could you make it iced")]
    fb = feedback_coach.feedback(st)
    assert "사용한 표현" in fb["feedback"]
    assert isinstance(fb["review_list"], list)


def test_flashcards_path():
    st = dict(BASE)
    st.update(analyzer.run(st))
    cards = feedback_coach.flashcards(st)["flashcards"]
    assert len(cards) == len(st["enriched_expressions"])
    assert set(cards[0].keys()) == {"front", "back", "example"}


# ── 전체 그래프 (Supervisor 오케스트레이션) ──────────────
def test_full_graph_roleplay_path():
    app = build_graph()
    init = dict(BASE)
    init.update({
        "practice_mode": "roleplay",
        "quiz_answers": ["I'll have", "make", "to go"],
        "learner_queue": ["Hi, I'll have a latte", "Could you make it iced?", "To go thanks"],
    })
    res = app.invoke(init, {"configurable": {"thread_id": "pytest-roleplay"}})
    assert res["stage"] == "done"
    assert res["quiz_score"] == 3
    assert res["turn_count"] == 3
    assert res["feedback"]
    assert len(res["study_history"]) >= 3     # analyzer + quiz + feedback


def test_full_graph_flashcards_path():
    app = build_graph()
    init = dict(BASE)
    init.update({"practice_mode": "flashcards",
                 "quiz_answers": ["x", "y", "z"], "learner_queue": []})
    res = app.invoke(init, {"configurable": {"thread_id": "pytest-flash"}})
    assert res["stage"] == "done"
    assert len(res["flashcards"]) >= 3


# ── FastAPI 엔드포인트 ──────────────────────────────────
def test_api_end_to_end():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app

    c = TestClient(fastapi_app)
    assert c.get("/health").json()["status"] == "ok"

    a = c.post("/session/analyze", json={"user_id": "api-test", "url_or_id": "cafe_order_demo"}).json()
    sid = a["session_id"]
    assert len(a["expressions"]) >= 3 and len(a["quiz"]) == 3

    g = c.post("/quiz/grade", json={"session_id": sid, "answers": ["I'll have", "make", "to go"]}).json()
    assert g["score"] == 3

    r = c.post("/roleplay/turn", json={"session_id": sid, "message": "Hi there"}).json()
    assert r["partner_reply"]

    f = c.post("/session/feedback", json={"session_id": sid, "mode": "roleplay"}).json()
    assert "feedback" in f

    assert c.get("/user/api-test/stats").json()["expressions_learned"] >= 3


# ── TTS (gpt-4o-mini-tts) ───────────────────────────────
def test_tts_synthesize_mock_returns_none():
    # 키 없는 mock 모드에서는 None 을 돌려주고, 앱이 죽지 않아야 한다.
    from backend import tts
    assert tts.available() in (True, False)
    if not tts.available():
        assert tts.synthesize("hello", persona="friend") is None


def test_tts_persona_voice_map_covers_personas():
    from backend import tts
    from backend.state import PERSONAS
    for p in PERSONAS:
        assert p in tts.PERSONA_VOICE       # 모든 페르소나에 목소리 매핑 존재


def test_roleplay_turn_includes_audio_field():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    a = c.post("/session/analyze", json={"user_id": "tts-test", "url_or_id": "cafe_order_demo"}).json()
    r = c.post("/roleplay/turn", json={"session_id": a["session_id"],
                                       "message": "Hi", "speak": True}).json()
    assert "audio_b64" in r                 # 필드 존재 (mock 이면 None)
    assert r["audio_ai_generated"] is True


# ── STT (gpt-4o-mini-transcribe) ────────────────────────
def test_stt_available_and_mock_returns_none():
    from backend import stt
    assert stt.available() in (True, False)
    if not stt.available():
        assert stt.transcribe(b"\x00\x01", language="English") is None


def test_stt_iso_code_map():
    from backend import stt
    assert stt.iso_code("English") == "en"
    assert stt.iso_code("한국어") == "ko"
    assert stt.iso_code("日本語") == "ja"


def test_stt_endpoint_accepts_upload():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    files = {"file": ("speech.wav", b"RIFFxxxxWAVE", "audio/wav")}
    r = c.post("/stt", files=files, data={"language": "English"}).json()
    assert "text" in r and "available" in r   # mock 이면 text=None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
