"""
test_nodes.py — 전문 에이전트 & 그래프 & API 노드 테스트 (PyTest)

실행:  pytest -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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


# ── LLM JSON 파싱 견고성 (실제 버그 재현: markdown/prose 로 감싼 JSON) ──
def test_extract_json_handles_plain():
    from backend.config import _extract_json
    assert _extract_json('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_handles_markdown_fence():
    from backend.config import _extract_json
    raw = '```json\n[{"a": 1}, {"a": 2}]\n```'
    assert _extract_json(raw) == [{"a": 1}, {"a": 2}]


def test_extract_json_handles_prose_wrapped():
    from backend.config import _extract_json
    raw = 'Here are the expressions you asked for:\n```json\n[{"a": 1}]\n```\nLet me know if you need more!'
    assert _extract_json(raw) == [{"a": 1}]


def test_extract_json_raises_on_garbage():
    from backend.config import _extract_json
    import pytest as _pytest
    with _pytest.raises(ValueError):
        _extract_json("Sorry, I can't help with that.")


def test_llm_json_surfaces_warning_on_failure(monkeypatch):
    """실제 버그였던 부분: GPT 호출이 실패하면 절대 조용히 mock으로 넘어가면 안 되고,
    호출자가 반드시 알 수 있는 warning을 함께 돌려줘야 한다."""
    from backend import config as config_module

    class _BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("simulated API failure")

    monkeypatch.setattr(config_module, "USE_REAL_LLM", True)
    monkeypatch.setattr(config_module, "_llm", _BrokenLLM())

    data, warning = config_module.llm_json("sys", "user", mock=[{"x": 1}], context="테스트")
    assert data == [{"x": 1}]        # mock으로 대체되긴 하지만
    assert warning is not None       # 반드시 이유가 함께 와야 한다
    assert "테스트" in warning


def test_llm_json_no_warning_when_response_is_valid(monkeypatch):
    from backend import config as config_module

    class _WorkingLLM:
        def invoke(self, messages):
            class R:
                content = '```json\n[{"expression": "real content"}]\n```'
            return R()

    monkeypatch.setattr(config_module, "USE_REAL_LLM", True)
    monkeypatch.setattr(config_module, "_llm", _WorkingLLM())

    data, warning = config_module.llm_json("sys", "user", mock=[{"x": 1}])
    assert data == [{"expression": "real content"}]
    assert warning is None


def test_fetch_transcript_sample_mode_returns_dict_with_source():
    from backend.youtube_service import fetch_transcript
    result = fetch_transcript("cafe_order_demo", "English")
    assert result["source"] == "sample"
    assert result["text"] == SAMPLE_TRANSCRIPT


def test_analyze_endpoint_returns_transcript_source():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    a = c.post("/session/analyze", json={"user_id": "src-test", "url_or_id": "cafe_order_demo"}).json()
    assert a.get("transcript_source") == "sample"
    assert a.get("video_id") == "cafe_order_demo"   # 언어 변경 후 같은 영상 재분석에 필요


def test_analyze_endpoint_handles_total_transcript_failure(monkeypatch):
    """자막 + Whisper 둘 다 실패하면 500이 아니라 안내 메시지를 반환해야 한다."""
    from fastapi.testclient import TestClient
    from backend import main as main_module
    from backend.youtube_service import TranscriptUnavailableError

    def _boom(video_id, target_language="English"):
        raise TranscriptUnavailableError(video_id, "no captions, whisper failed")

    monkeypatch.setattr(main_module, "fetch_transcript", _boom)
    c = TestClient(main_module.app)
    r = c.post("/session/analyze", json={"user_id": "fail-test", "url_or_id": "totally_broken_id"})
    assert r.status_code == 200                   # 500이 아니라 정상 응답으로 안내
    body = r.json()
    assert body.get("error") == "video_unavailable"
    assert "다른" in body.get("message", "")       # 다른 URL 안내 문구 포함


# ── 에이전트 1: ContentAnalyzer ─────────────────────────
def test_analyzer_extracts_and_enriches():
    out = analyzer.run(dict(BASE))
    assert len(out["key_expressions"]) >= 3
    assert len(out["enriched_expressions"]) == len(out["key_expressions"])
    first = out["enriched_expressions"][0]
    assert "definition" in first and "extra_examples" in first and "is_new" in first
    assert out["stage"] == "analyzed"


def test_analyzer_includes_native_explanation():
    out = analyzer.run(dict(BASE))
    first = out["enriched_expressions"][0]
    assert first.get("explanation")            # 모국어 설명 존재
    assert len(first["explanation"]) > len(first["meaning"])  # meaning 보다 더 자세함


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


def test_quiz_is_multiple_choice_with_3_options():
    st = dict(BASE)
    st.update(analyzer.run(st))
    st.update(quiz_master.generate(st))
    for q in st["quiz"]:
        assert len(q["choices"]) == 3
        assert q["answer"] in q["choices"]        # 정답이 보기 안에 포함


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


def test_feedback_uses_semantic_llm_match_not_naive_substring(monkeypatch):
    """실제 버그 재현: 학습자가 표현을 '의역'해서 말해도(정확한 부분 문자열이 아니어도)
    LLM 기반 판정이면 '사용함'으로 정확히 인식해야 한다."""
    from backend import config as config_module

    st = dict(BASE)
    st.update(analyzer.run(st))
    expr_names = [e["expression"] for e in st["key_expressions"]]
    # 학습자가 표현을 그대로 말하지 않고 의역함 (naive substring 매칭이면 놓쳤을 문장)
    st["messages"] = [HumanMessage(content="I think I'd like to get a latte, if that's okay.")]

    class _WorkingLLM:
        def invoke(self, messages):
            class R:
                content = json.dumps({
                    "used": expr_names[:1],   # LLM이 의역을 보고도 첫 표현을 '사용함'으로 판정
                    "corrections": [{"original": "I think I'd like to get a latte, if that's okay.",
                                     "issue": "지나치게 장황함", "suggestion": "I'll have a latte, please."}],
                    "overall_comment": "잘했어요!",
                })
            return R()

    monkeypatch.setattr(config_module, "USE_REAL_LLM", True)
    monkeypatch.setattr(config_module, "_llm", _WorkingLLM())

    fb = feedback_coach.feedback(st)
    assert expr_names[0] in fb["feedback"]        # '사용한 표현'으로 반영됨
    assert fb["corrections"]                       # 어색한 표현 교정도 함께 반환
    assert fb["corrections"][0]["issue"] == "지나치게 장황함"


def test_feedback_no_llm_warning_key_when_using_mock_without_conversation():
    """대화가 비어있으면 LLM을 아예 호출하지 않고 결정적 mock으로 처리해야 한다."""
    st = dict(BASE)
    st.update(analyzer.run(st))
    st["messages"] = []
    fb = feedback_coach.feedback(st)
    assert fb.get("llm_warning") is None


def test_roleplay_system_prompt_forbids_language_mixing():
    """실제 버그 재현: 역할극 응답에 학습 언어가 아닌 다른 언어가 섞이던 문제.
    시스템 프롬프트가 명시적으로 언어 혼용을 금지해야 한다."""
    st = dict(BASE)
    st["target_language"] = "English"
    sys_msg = roleplay_partner._system(st)
    assert "ONLY in English" in sys_msg.content or "ENTIRELY in English" in sys_msg.content
    assert "never mix" in sys_msg.content.lower() or "code-switch" in sys_msg.content.lower()


def test_flashcards_path():
    st = dict(BASE)
    st.update(analyzer.run(st))
    cards = feedback_coach.flashcards(st)["flashcards"]
    assert len(cards) == len(st["enriched_expressions"])
    assert set(cards[0].keys()) == {"front", "back", "explanation", "example"}


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


def test_analyze_response_includes_llm_warnings_field():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    a = c.post("/session/analyze", json={"user_id": "warn-test", "url_or_id": "cafe_order_demo"}).json()
    assert "llm_warnings" in a
    assert a["llm_warnings"] == []   # mock 모드에서는 GPT를 아예 안 부르므로 경고 없음


def test_max_turns_is_configurable_and_clamped():
    """역할극 턴 수를 사용자가 지정할 수 있어야 하고, 1~10 범위로 안전하게 clamp 되어야 한다."""
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)

    # 사용자가 6턴을 요청 → 그대로 반영
    a = c.post("/session/analyze", json={"user_id": "turns-test", "url_or_id": "cafe_order_demo",
                                         "max_turns": 6}).json()
    sid = a["session_id"]
    for i in range(6):
        r = c.post("/roleplay/turn", json={"session_id": sid, "message": f"turn {i}"}).json()
    assert r["turn"] == 6
    assert r["max_turns"] == 6

    # 범위를 벗어난 값(999)은 안전하게 clamp
    a2 = c.post("/session/analyze", json={"user_id": "turns-test2", "url_or_id": "cafe_order_demo",
                                          "max_turns": 999}).json()
    sid2 = a2["session_id"]
    r2 = c.post("/roleplay/turn", json={"session_id": sid2, "message": "hi"}).json()
    assert r2["max_turns"] == 10


# ── 역할극 AI 선공개 인사 (roleplay/start) ──────────────
def test_roleplay_agent_start_opens_without_learner_turn():
    st = dict(BASE)
    st.update(analyzer.run(st))
    st["turn_count"] = 0
    out = roleplay_partner.start(st)
    assert len(out["messages"]) == 1                # AI 메시지 1개만
    assert isinstance(out["messages"][0], AIMessage)
    assert out["turn_count"] == 0                    # 학습자 턴은 아직 안 늘어남


def test_roleplay_start_endpoint_precedes_turn():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    a = c.post("/session/analyze", json={"user_id": "open-test", "url_or_id": "cafe_order_demo"}).json()
    sid = a["session_id"]

    start_r = c.post("/roleplay/start", json={"session_id": sid, "speak": False}).json()
    assert start_r["partner_reply"]
    assert start_r["turn"] == 0                       # 아직 학습자 턴 아님

    turn_r = c.post("/roleplay/turn", json={"session_id": sid, "message": "Hi!", "speak": False}).json()
    assert turn_r["turn"] == 1                         # 이제 학습자가 응답해서 1턴


# ── 영상별 플래시카드 (사이드바 복습) ───────────────────
def test_flashcards_grouped_by_video():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    c.post("/session/analyze", json={"user_id": "fc-test", "url_or_id": "cafe_order_demo"})

    fc = c.get("/user/fc-test/flashcards").json()["flashcards_by_video"]
    assert len(fc) >= 1
    assert fc[0]["video_title"]
    assert len(fc[0]["cards"]) >= 1
    assert set(fc[0]["cards"][0].keys()) == {"expression", "meaning", "example"}


def test_flashcards_grouped_by_session_with_timestamp():
    """같은 영상을 다시 검색(분석)하면 제목+검색시간 기준으로 별도 그룹이 되어야 한다."""
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    c.post("/session/analyze", json={"user_id": "fc-ts-test", "url_or_id": "cafe_order_demo"})
    c.post("/session/analyze", json={"user_id": "fc-ts-test", "url_or_id": "cafe_order_demo"})

    fc = c.get("/user/fc-ts-test/flashcards").json()["flashcards_by_video"]
    assert len(fc) == 2                          # 같은 영상이어도 검색을 두 번 했으니 그룹 2개
    for group in fc:
        assert group["searched_at"]               # 검색 날짜시간이 채워져 있어야 함
        assert group["video_title"]
        assert group["session_id"]


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
def test_stt_available_flag():
    from backend import stt
    assert stt.available() in (True, False)


def test_stt_mock_mode_returns_rotating_demo_utterance():
    """역할극이 음성 전용이라, 키 없는 데모 모드에서도 흐름이 끊기면 안 된다."""
    from backend import stt
    if stt.available():
        return  # 실키 환경이면 이 데모 폴백 테스트는 건너뜀
    first = stt.transcribe(b"\x00\x01", language="English", turn_index=0)
    second = stt.transcribe(b"\x00\x01", language="English", turn_index=1)
    assert first and second
    assert first != second                      # 턴마다 다른 예시 발화
    assert stt.transcribe(b"", language="English") is None  # 오디오 자체가 없으면 None


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
    r = c.post("/stt", files=files, data={"language": "English", "turn_index": 0}).json()
    assert "text" in r and "available" in r
    assert r["text"]   # 음성 전용 역할극이라 데모 모드에서도 항상 텍스트를 돌려줘야 함


# ── 복습 사이드바 (전체 이력 + 완료 처리) ─────────────────
def test_reviews_all_and_mark_done():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)

    a = c.post("/session/analyze", json={"user_id": "review-test", "url_or_id": "cafe_order_demo"}).json()
    sid = a["session_id"]
    c.post("/roleplay/turn", json={"session_id": sid, "message": "hi", "speak": False})
    fb = c.post("/session/feedback", json={"session_id": sid, "mode": "roleplay"}).json()

    all_rev = c.get("/user/review-test/reviews/all").json()["reviews"]
    assert len(all_rev) >= 1
    rid = all_rev[0]["id"]

    done = c.post(f"/reviews/{rid}/done").json()
    assert done["ok"] is True

    still_due = c.get("/user/review-test/reviews").json()["reviews"]
    assert rid not in [r["id"] for r in still_due]


# ── 개발자용 폴백 문구가 학습자 화면에 노출되지 않도록 형식 고정 ──
def test_example_search_fallback_format_matches_frontend_filter():
    """tools.py의 범용 폴백 문구 형식이 바뀌면 프런트의 필터(startswith 체크)도
    깨지므로, 형식이 예상대로인지 여기서 고정해 둔다. (코드는 그대로 두고
    프런트에서만 숨기는 방식이라, 이 테스트가 그 계약을 지켜준다)"""
    from backend.tools import example_sentence_search
    result = example_sentence_search.invoke({"expression": "a totally unknown phrase xyz"})
    assert result == ["Example using 'a totally unknown phrase xyz'."]
    assert result[0].startswith("Example using '")   # 프런트 필터가 이 접두어로 걸러냄


# ── 인증: 비밀번호 해싱 ──────────────────────────────────
def test_password_hash_roundtrip():
    from backend import auth
    hashed = auth.hash_password("supersecret123")
    assert "$" in hashed
    assert auth.verify_password("supersecret123", hashed) is True
    assert auth.verify_password("wrongpassword", hashed) is False


def test_password_hash_unique_salt_per_call():
    from backend import auth
    h1 = auth.hash_password("samepassword")
    h2 = auth.hash_password("samepassword")
    assert h1 != h2                              # salt 가 매번 달라야 함
    assert auth.verify_password("samepassword", h1)
    assert auth.verify_password("samepassword", h2)


def test_password_strength_check():
    from backend import auth
    ok, _ = auth.password_is_strong_enough("longenoughpassword")
    assert ok is True
    ok2, msg = auth.password_is_strong_enough("short")
    assert ok2 is False and msg


# ── 인증: 회원가입 → 이메일 인증 → 로그인 전체 플로우 ─────
def test_signup_verify_login_full_flow():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)

    email = "newuser@example.com"
    s = c.post("/auth/signup", json={"email": email, "password": "mypassword123"}).json()
    assert s["ok"] is True
    assert "dev_code" in s                        # SMTP 미설정 → 데모 코드 노출

    # 인증 전에는 로그인 불가
    login_before = c.post("/auth/login", json={"email": email, "password": "mypassword123"}).json()
    assert login_before["ok"] is False
    assert login_before.get("needs_verification") is True

    # 틀린 코드
    bad = c.post("/auth/verify", json={"email": email, "code": "000000"}).json()
    assert bad["ok"] is False

    # 올바른 코드로 인증
    ok = c.post("/auth/verify", json={"email": email, "code": s["dev_code"]}).json()
    assert ok["ok"] is True

    # 이제 로그인 가능
    login_ok = c.post("/auth/login", json={"email": email, "password": "mypassword123"}).json()
    assert login_ok["ok"] is True
    assert login_ok["email"] == email

    # 틀린 비밀번호는 거부
    login_wrong = c.post("/auth/login", json={"email": email, "password": "wrongpass"}).json()
    assert login_wrong["ok"] is False


def test_signup_rejects_invalid_email():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    r = c.post("/auth/signup", json={"email": "not-an-email", "password": "goodpassword"}).json()
    assert r["ok"] is False


def test_signup_rejects_weak_password():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    r = c.post("/auth/signup", json={"email": "weakpass@example.com", "password": "123"}).json()
    assert r["ok"] is False


def test_signup_duplicate_verified_email_rejected():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    email = "dup@example.com"
    s = c.post("/auth/signup", json={"email": email, "password": "mypassword123"}).json()
    c.post("/auth/verify", json={"email": email, "code": s["dev_code"]})

    dup = c.post("/auth/signup", json={"email": email, "password": "anotherpassword"}).json()
    assert dup["ok"] is False


def test_resend_code_issues_new_code():
    from fastapi.testclient import TestClient
    from backend.main import app as fastapi_app
    c = TestClient(fastapi_app)
    email = "resend@example.com"
    s1 = c.post("/auth/signup", json={"email": email, "password": "mypassword123"}).json()
    r = c.post("/auth/resend", json={"email": email}).json()
    assert r["ok"] is True
    assert "dev_code" in r
    # 이전 코드는 더 이상 유효하지 않아야 함 (새 코드로 교체됨)
    old_code_result = c.post("/auth/verify", json={"email": email, "code": s1["dev_code"]}).json()
    assert old_code_result["ok"] is False
    new_code_result = c.post("/auth/verify", json={"email": email, "code": r["dev_code"]}).json()
    assert new_code_result["ok"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
