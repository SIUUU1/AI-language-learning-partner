"""
streamlit_app.py — LinguaLoop UI
"""
from __future__ import annotations

import base64
import os
import re
import requests
import time
import streamlit as st
import streamlit.components.v1 as components

API = os.getenv("LINGUALOOP_API", "http://localhost:8000")

st.set_page_config(page_title="LinguaLoop 말문", page_icon="🎓",
                   layout="centered", initial_sidebar_state="expanded")

# 개발자용 디버그 정보 표시 여부. 기본은 숨김(False).
# 켜는 방법: LINGUALOOP_DEBUG=1 환경변수, 또는 URL에 ?debug=1 붙이기.
DEBUG = os.getenv("LINGUALOOP_DEBUG", "0") == "1" or st.query_params.get("debug") == "1"


def debug_caption(text: str):
    """개발자 디버그 캡션. DEBUG=False 면 코드는 유지하되 화면엔 출력하지 않는다."""
    if DEBUG:
        st.caption(f"🛠️ [DEBUG] {text}")


# 아래 3개는 "코드는 남기되 화면엔 안 보이게" 숨겨둔 기능 플래그.
# True 로 바꾸면 언제든 다시 노출할 수 있다.
SHOW_STATS_SIDEBAR = False        # 사이드바 "📊 내 학습 통계"
SHOW_REVIEW_REMINDER = False      # 사이드바 "🔁 이전 학습 복습하기"
SHOW_TURN_SLIDER = False          # 사이드바 "역할극 턴 수" 슬라이더
FIXED_MAX_TURNS = 5               # 슬라이더를 숨긴 대신 고정하는 역할극 턴 수

PERSONA_LABEL = {"friend": "친구", "teacher": "선생님", "interviewer": "면접관",
                 "partner": "연인", "barista": "바리스타"}

# 이메일 로그인용 간단한 형식 검증 (실제 인증서버 없이, 이메일을 사용자 식별자로만 사용)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# 학습 언어 → 브라우저 음성(Web Speech API) 언어 코드
_BCP47 = {"English": "en-US", "日本語": "ja-JP", "中文": "zh-CN",
          "Español": "es-ES", "Français": "fr-FR"}

# 진행 단계 순서 (하나씩 언락되며 나타남)
STEP_ORDER = ["expressions", "quiz", "roleplay", "feedback"]


# ─────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────
def api_get(path: str):
    try:
        return requests.get(f"{API}{path}", timeout=60).json()
    except Exception as e:
        st.error(f"백엔드 연결 실패: {e}")
        return {}


def api_post(path: str, payload: dict):
    try:
        return requests.post(f"{API}{path}", json=payload, timeout=120).json()
    except Exception as e:
        st.error(f"백엔드 연결 실패: {e}")
        return {}


def browser_speak(text: str, lang: str = "en-US"):
    """OpenAI TTS 가 없을 때 브라우저 내장 음성으로 대체 재생 (키 불필요)."""
    safe = (text or "").replace("\\", " ").replace("`", " ").replace('"', " ")
    components.html(f"""
        <script>
        const u = new SpeechSynthesisUtterance("{safe}");
        u.lang = "{lang}";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        </script>""", height=0)


def audio_play_button(audio_b64: str, key: str, label: str = "🔊 다시 듣기",
                      autoplay: bool = False):
    """재생/일시정지 버튼만 보이는 미니 오디오 플레이어 (재생바 없음)."""
    html = f"""
    <div style="margin:6px 0;">
      <button id="btn_{key}" style="
          background:#6c5ce7;color:white;border:none;border-radius:999px;
          padding:8px 20px;font-size:14px;cursor:pointer;">{label}</button>
      <audio id="au_{key}" src="data:audio/mp3;base64,{audio_b64}"></audio>
    </div>
    <script>
      const audio_{key} = document.getElementById('au_{key}');
      const btn_{key} = document.getElementById('btn_{key}');
      const label_{key} = "{label}";
      btn_{key}.onclick = function() {{
        if (audio_{key}.paused) {{ audio_{key}.play(); }} else {{ audio_{key}.pause(); }}
      }};
      audio_{key}.onplay = function() {{ btn_{key}.innerText = '⏸ 일시정지'; }};
      audio_{key}.onpause = function() {{ btn_{key}.innerText = label_{key}; }};
      audio_{key}.onended = function() {{ btn_{key}.innerText = label_{key}; }};
      {"audio_" + key + ".play().catch(()=>{});" if autoplay else ""}
    </script>
    """
    components.html(html, height=54)


def goto_step(name: str):
    ss["learn_step"] = name
    st.rerun()


def step_reached(name: str) -> bool:
    """현재까지 언락된 단계인지 (하나씩 나타나게 하는 게이트)."""
    if not ss.get("learn_step"):
        return False
    return STEP_ORDER.index(name) <= STEP_ORDER.index(ss["learn_step"])


def step_is_current(name: str) -> bool:
    return ss.get("learn_step") == name


# ─────────────────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────────────────
ss = st.session_state
_DEFAULTS = {
    "session_id": None, "analysis": None, "learn_step": None,
    "chat": list, "turn": 0, "graded": None, "feedback": None,
    "pending_speak": False, "pending_audio": None, "pending_text": "",
    "search_results": list, "video_error": None, "roleplay_opened": False,
    "max_turns": 3, "llm_warnings": list, "chat_audio_idx": dict, "expr_audio": dict,
    "session_native": None, "session_target": None, "lang_change_dismissed_for": None,
    "url_edit_unlocked": False, "url_edit_confirm_pending": False,
}
for k, v in _DEFAULTS.items():
    ss.setdefault(k, v() if callable(v) else v)


def _reset_session_state():
    """학습 세션 관련 상태만 초기화한다 (로그인·사이드바 설정은 그대로 둔다)."""
    for k in _DEFAULTS:
        ss.pop(k, None)
    for k, v in _DEFAULTS.items():
        ss.setdefault(k, v() if callable(v) else v)


# ─────────────────────────────────────────────────────────
# 이메일 로그인 게이트 — 로그인 전에는 이 아래로 아무것도 렌더링하지 않는다
# ─────────────────────────────────────────────────────────
ss.setdefault("user_email", None)
ss.setdefault("searching", False)
ss.setdefault("auth_stage", "login")          # "login" | "signup" | "verify"
ss.setdefault("pending_verify_email", None)
ss.setdefault("pending_dev_code", None)   # SMTP 미설정 시 데모용 인증 코드 (실발송 시엔 None)

if not ss["user_email"]:
    st.title("🎓 LinguaLoop 말문")

    # ── 이메일 인증 코드 입력 화면 (회원가입 직후, 또는 미인증 상태로 로그인 시도 시) ──
    if ss["auth_stage"] == "verify":
        st.subheader("📧 이메일 인증")
        v_email = ss.get("pending_verify_email", "")
        st.caption(f"**{v_email}** 로 인증 코드를 보냈어요. 코드 6자리를 입력해 주세요. "
                   f"(유효 시간 10분)")
        # rerun 직전에 st.info()를 호출하면 그 delta가 전송되기 전에 실행이 버려지므로
        # (동일한 문제를 앞서 다시듣기 버튼/디버그 로그에서도 겪었다), 코드는
        # session_state 에 저장해 두고 이 "안정적인" 다음 렌더에서 표시한다.
        if ss.get("pending_dev_code"):
            st.info(f"🧪 (데모 모드: 이메일 발송 미설정) 인증 코드: **{ss['pending_dev_code']}**")
        code_input = st.text_input("인증 코드", max_chars=6, key="verify_code_input")
        vc1, vc2 = st.columns(2)
        if vc1.button("✅ 인증하기", type="primary", use_container_width=True):
            r = api_post("/auth/verify", {"email": v_email, "code": code_input.strip()})
            if r.get("ok"):
                ss["flash_message"] = "인증 완료! 이제 로그인해 주세요."
                ss["auth_stage"] = "login"
                ss["pending_verify_email"] = None
                ss["pending_dev_code"] = None
                st.rerun()
            else:
                st.error(r.get("message", "인증에 실패했어요."))
        if vc2.button("↩️ 뒤로", use_container_width=True):
            ss["auth_stage"] = "login"
            ss["pending_dev_code"] = None
            st.rerun()
        if st.button("코드 다시 받기", use_container_width=True):
            r = api_post("/auth/resend", {"email": v_email})
            if r.get("ok"):
                ss["pending_dev_code"] = r.get("dev_code")   # 없으면(실발송) None 으로 지워짐
                st.rerun()
            else:
                st.error(r.get("message", "재발송에 실패했어요."))
        st.stop()

    # ── 로그인 / 회원가입 탭 ──
    if ss.get("flash_message"):
        st.success(ss.pop("flash_message"))

    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        st.caption("이메일과 비밀번호로 로그인하세요.")
        login_email = st.text_input("이메일", placeholder="you@example.com", key="login_email_input")
        login_pw = st.text_input("비밀번호", type="password", key="login_pw_input")
        if st.button("로그인", type="primary", use_container_width=True, key="login_submit_btn"):
            candidate = login_email.strip().lower()
            if not EMAIL_RE.match(candidate):
                st.error("올바른 이메일 형식을 입력해 주세요.")
            elif not login_pw:
                st.error("비밀번호를 입력해 주세요.")
            else:
                r = api_post("/auth/login", {"email": candidate, "password": login_pw})
                if r.get("ok"):
                    ss["user_email"] = candidate
                    st.rerun()
                elif r.get("needs_verification"):
                    ss["auth_stage"] = "verify"
                    ss["pending_verify_email"] = candidate
                    st.rerun()
                else:
                    st.error(r.get("message", "로그인에 실패했어요."))

    with tab_signup:
        st.caption("이메일 인증 후 가입이 완료돼요. 비밀번호는 최소 8자 이상이어야 해요.")
        signup_email = st.text_input("이메일", placeholder="you@example.com", key="signup_email_input")
        signup_pw = st.text_input("비밀번호", type="password", key="signup_pw_input")
        signup_pw2 = st.text_input("비밀번호 확인", type="password", key="signup_pw2_input")
        if st.button("인증 코드 받기", type="primary", use_container_width=True, key="signup_submit_btn"):
            candidate = signup_email.strip().lower()
            if not EMAIL_RE.match(candidate):
                st.error("올바른 이메일 형식을 입력해 주세요.")
            elif signup_pw != signup_pw2:
                st.error("비밀번호가 서로 달라요.")
            else:
                r = api_post("/auth/signup", {"email": candidate, "password": signup_pw})
                if r.get("ok"):
                    ss["auth_stage"] = "verify"
                    ss["pending_verify_email"] = candidate
                    ss["pending_dev_code"] = r.get("dev_code")   # 실발송이면 None
                    st.rerun()
                else:
                    st.error(r.get("message", "회원가입에 실패했어요."))

    st.stop()


# ─────────────────────────────────────────────────────────
# 검색/분석 헬퍼 — 사이드바에서도 호출해야 해서(언어 변경 시 재분석) 먼저 정의한다.
# user_id/native/target/persona/max_turns_setting 은 아래 사이드바 블록에서 값이
# 채워지며, 이 함수들은 실제 호출 시점(사이드바 렌더 이후)에만 실행되므로 문제없다.
# ─────────────────────────────────────────────────────────
def _start_session(url_or_id: str):
    with st.spinner("영상 분석 중... (자막 → 안되면 Whisper 음성 인식 시도)"):
        a = api_post("/session/analyze", {
            "user_id": user_id, "url_or_id": url_or_id,
            "native_language": native, "target_language": target, "persona": persona,
            "max_turns": max_turns_setting})
    ss["searching"] = False
    ss["url_edit_unlocked"] = False   # 새 세션이 시작됐으니 다시 잠금 상태로
    if a.get("error") == "video_unavailable":
        ss["video_error"] = a.get("message", "이 영상으로는 학습 콘텐츠를 만들 수 없어요.")
        st.rerun()
    elif "session_id" in a:
        ss["video_error"] = None
        ss["session_id"] = a["session_id"]
        ss["analysis"] = a
        ss["max_turns"] = max_turns_setting
        ss["llm_warnings"] = a.get("llm_warnings", [])
        ss["session_native"] = native      # 이후 이 세션은 이 시점의 언어로 고정
        ss["session_target"] = target
        ss["lang_change_dismissed_for"] = None
        ss["learn_step"] = "expressions"   # 첫 단계만 언락
        st.rerun()


def _looks_like_youtube_url_or_id(text: str) -> bool:
    """youtube.com/youtu.be 링크거나, 11자리 video ID 형식이면 바로 분석 대상으로 본다."""
    text = text.strip()
    if re.search(r"(youtube\.com|youtu\.be)", text):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", text))


# ─────────────────────────────────────────────────────────
# 사이드바 (기본 펼침, 상단 화살표로 접을 수 있음)
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 LinguaLoop")
    #st.caption("영상 → 표현 → 퀴즈 → 역할극 → 복습")
    st.caption(f"👤 {ss['user_email']}")
    if st.button("🚪 로그아웃", use_container_width=True):
        ss.clear()
        st.rerun()

    health = api_get("/health")
    tts_available = False
    stt_available = False
    if health:
        m = health.get("mode", {})
        tts_available = health.get("tts_available", False)
        stt_available = health.get("stt_available", False)
        st.info(f"LLM: **{m.get('llm')}**  ·  YouTube: **{m.get('youtube')}**")
        personas = health.get("personas", list(PERSONA_LABEL.keys()))
    else:
        personas = list(PERSONA_LABEL.keys())

    st.subheader("학습 설정")
    debug_caption(f"last_reload_video_id={ss.get('_debug_last_reload_video_id')!r} · "
                 f"analysis_has_video_id={('video_id' in (ss.get('analysis') or {}))} · "
                 f"analysis_video_id={(ss.get('analysis') or {}).get('video_id')!r}")
    user_id = ss["user_email"]
    st.text_input("사용자 ID (이메일)", value=user_id, disabled=True)
    native = st.selectbox("모국어", ["한국어", "English", "日本語", "中文"], index=0,
                          key="native_select")
    target = st.selectbox("학습 언어", ["English", "日本語", "中文", "Español", "Français"],
                          index=0, key="target_select")
    persona = st.selectbox("역할극 파트너", personas,
                           format_func=lambda p: PERSONA_LABEL.get(p, p))
    if SHOW_TURN_SLIDER:
        max_turns_setting = st.slider("역할극 턴 수", min_value=1, max_value=10, value=3,
                                      help="기술적으로 정해진 상한은 없지만, 너무 길면 대화가 "
                                           "늘어지고 API 비용도 늘어서 1~10턴을 권장해요.")
    else:
        max_turns_setting = FIXED_MAX_TURNS

    # ── 학습 중간에 모국어/학습 언어를 바꾸면 새로 시작할지 물어본다 ──
    if ss.get("analysis"):
        current_pair = (native, target)
        session_pair = (ss.get("session_native"), ss.get("session_target"))
        if current_pair != session_pair and ss.get("lang_change_dismissed_for") != current_pair:
            st.warning("모국어/학습 언어가 바뀌었어요. 새 학습을 시작할까요?")
            lc1, lc2 = st.columns(2)
            if lc1.button("✅ 새로 시작", key="lang_change_yes", use_container_width=True):
                video_id_to_reload = ss.get("analysis", {}).get("video_id")
                ss["_debug_last_reload_video_id"] = video_id_to_reload   # rerun 이후에도 보이도록 저장
                _reset_session_state()
                if video_id_to_reload:
                    _start_session(video_id_to_reload)
                else:
                    st.rerun()
            if lc2.button("↩️ 그대로 유지", key="lang_change_no", use_container_width=True):
                ss["lang_change_dismissed_for"] = current_pair
                st.rerun()

    st.subheader("🔊 음성")
    voice_on = st.toggle("파트너 음성 재생", value=True)
    #if voice_on:
    #    st.caption("gpt-4o-mini-tts · 페르소나별 말투 · **AI 합성 음성**" if tts_available
    #               else "OpenAI 키 없음 → 브라우저 내장 음성 (**AI 합성 음성**)")
    #st.caption("🎤 말하기(STT): " +
    #           ("gpt-4o-mini-transcribe 사용" if stt_available
    #            else "OpenAI 키 없음 → 데모 예시 발화 사용 (역할극은 음성 전용)"))

    st.divider()
    if st.button("🔄 새 학습 시작", use_container_width=True):
        _reset_session_state()
        st.rerun()

    stats = api_get(f"/user/{user_id}/stats") if user_id else {}
    if SHOW_STATS_SIDEBAR and stats:
        st.subheader("📊 내 학습 통계")
        c1, c2, c3 = st.columns(3)
        c1.metric("표현", stats.get("expressions_learned", 0))
        c2.metric("퀴즈", stats.get("quizzes_taken", 0))
        c3.metric("복습", stats.get("reviews_due", 0))

    # ── 이전 학습 복습하기 (예정된 복습) ──
    if SHOW_REVIEW_REMINDER:
        st.divider()
        n_due = stats.get("reviews_due", 0) if stats else 0
        with st.expander(f"🔁 이전 학습 복습하기 ({n_due})"):
            due = api_get(f"/user/{user_id}/reviews").get("reviews", []) if user_id else []
            if not due:
                st.caption("복습할 표현이 없어요. 학습을 먼저 진행해 보세요!")
            for r in due:
                st.markdown(f"**{r['expression']}**")
                st.caption(r.get("meaning", ""))
                if st.button("✅ 복습 완료", key=f"revdone_{r['id']}", use_container_width=True):
                    api_post(f"/reviews/{r['id']}/done", {})
                    st.rerun()
                st.divider()

    # ── 영상별 플래시카드 (제목 + 검색 날짜시간 기준으로 묶어서, 기본은 접힌 상태) ──
    fc_data = api_get(f"/user/{user_id}/flashcards").get("flashcards_by_video", []) if user_id else []
    total_cards = sum(len(v["cards"]) for v in fc_data)
    with st.expander(f"🃏 플래시카드로 복습 ({total_cards})"):
        if not fc_data:
            st.caption("아직 학습한 영상이 없어요. 영상을 먼저 분석해 보세요!")
        for video in fc_data:
            group_label = f"🎬 {video['video_title']}"
            if video.get("searched_at"):
                group_label += f" · 🕒 {video['searched_at']}"
            with st.expander(f"{group_label} ({len(video['cards'])})", expanded=False):
                for card in video["cards"]:
                    with st.expander(card["expression"]):
                        st.write(f"**뜻:** {card['meaning']}")
                        if card.get("example"):
                            st.write(f"**예문:** {card['example']}")


# ─────────────────────────────────────────────────────────
# STEP 0: 검색 & 영상 선택
# (_start_session / _looks_like_youtube_url_or_id 는 사이드바보다 앞에서 이미 정의함)
# ─────────────────────────────────────────────────────────
st.header("1️⃣ 유튜브 영상 선택")

# 이미 학습 컨텐츠가 만들어졌으면, 검색창을 잠그고 "수정하기" 확인을 거치게 한다
# (URL/검색어를 바꾸면 지금까지의 학습 내용이 초기화되기 때문)
content_exists = bool(ss.get("analysis"))
locked = content_exists and not ss.get("url_edit_unlocked")

q = st.text_input("검색어 또는 YouTube URL/ID",
                  placeholder="예: ordering coffee in english  /  https://youtu.be/...",
                  disabled=locked, key="search_query_input")

if locked:
    if st.button("✏️ URL/검색어 수정하기", use_container_width=True):
        ss["url_edit_confirm_pending"] = True
        st.rerun()

if ss.get("url_edit_confirm_pending"):
    st.warning("URL/검색어를 수정하면 지금까지 학습한 내용이 초기화돼요. 계속할까요?")
    ec1, ec2 = st.columns(2)
    if ec1.button("예, 수정할게요", key="url_edit_yes", use_container_width=True):
        ss["url_edit_unlocked"] = True
        ss["url_edit_confirm_pending"] = False
        st.rerun()
    if ec2.button("아니오", key="url_edit_no", use_container_width=True):
        ss["url_edit_confirm_pending"] = False
        st.rerun()

# 진행 중인 세션이 있으면, 이후 모든 로직은 세션을 "시작한 시점"의 언어를 그대로 쓴다.
# (사이드바에서 언어를 바꿔도 '그대로 유지'를 선택하면 진행 중인 세션은 영향받지 않는다)
effective_native = ss.get("session_native") or native
effective_target = ss.get("session_target") or target

# 검색 버튼 — 로딩(분석/검색) 중에는 비활성화해서 중복 클릭을 막는다.
# 클릭 → searching=True 로 두고 즉시 rerun (버튼이 비활성화된 채로 먼저 그려짐)
# → 다음 렌더에서 searching=True 를 보고 실제 작업을 수행한다.
search_clicked = st.button("🔍 검색", type="primary", use_container_width=True,
                           disabled=ss.get("searching", False), key="search_btn")
if search_clicked and q and not locked:
    ss["searching"] = True
    st.rerun()

if ss.get("searching"):
    time.sleep(0.3)  # 비활성화된 버튼이 화면에 그려질 시간을 확보한 뒤 실제 작업 시작
    if content_exists and ss.get("url_edit_unlocked"):
        _reset_session_state()   # URL/검색어를 바꾸기로 확정했으니 기존 학습 내용 초기화
    if _looks_like_youtube_url_or_id(q):
        _start_session(q)        # 내부에서 searching=False 처리 후 rerun
    else:
        with st.spinner("검색 중..."):
            ss["search_results"] = api_post("/youtube/search", {"query": q}).get("results", [])
        ss["searching"] = False
        st.rerun()

for v in ss.get("search_results", []):
    with st.container(border=True):
        cols = st.columns([1, 3])
        if v.get("thumbnail"):
            cols[0].image(v["thumbnail"])
        cols[1].markdown(f"**{v['title']}**")
        cols[1].caption(f"{v['channel']}")
        if cols[1].button("이 영상으로 학습", key=f"pick_{v['video_id']}",
                          disabled=ss.get("searching", False)):
            if content_exists and ss.get("url_edit_unlocked"):
                _reset_session_state()
            _start_session(v["video_id"])

if ss.get("video_error"):
    st.error("😢 " + ss["video_error"])
    st.caption("💡 자막(1순위) → Whisper 음성 인식(2순위) 모두 실패한 영상이에요. "
               "위 검색창에 **다른 유튜브 URL**을 입력해서 다시 시도해 주세요.")


# ═════════════════════════════════════════════════════════
# STEP 1: 핵심 표현 (모국어 설명 포함) — 학습 완료해야 다음 단계 언락
# ═════════════════════════════════════════════════════════
if ss.get("analysis") and step_reached("expressions"):
    a = ss["analysis"]
    done = not step_is_current("expressions")
    with st.container(border=True):
        st.subheader(("✅ " if done else "2️⃣ ") + "핵심 표현")
        src_label = {"captions": "📝 공식 자막", "whisper": "🎙️ Whisper 음성 인식",
                    "sample": "🧪 샘플 데이터"}.get(a.get("transcript_source"), "")
        st.caption(f"🎬 {a.get('video_title','')}  ·  🆕 새 표현 {a.get('new_expression_count',0)}개 "
                   f"(ChromaDB 기준)  ·  {src_label}")
        if ss.get("llm_warnings"):
            for w in ss["llm_warnings"]:
                st.warning(f"⚠️ {w} 아래 표현은 이 영상 내용이 아니라 예시 데이터일 수 있어요.")
        debug_caption(f"expr_audio keys: {list(ss['expr_audio'].keys())}")
        for i, e in enumerate(a["expressions"]):
            badge = "🆕" if e.get("is_new") else "🔁"
            expander_key = f"expr_expander_{i}"
            with st.expander(f"{badge} {e['expression']} — {e['meaning']}",
                             expanded=not done, key=expander_key):
                st.write(f"**뜻:** {e['meaning']}")
                if e.get("explanation"):
                    st.info(e["explanation"])          # 모국어로 더 자세한 설명
                st.write(f"**예문:** {e.get('example','')}")
                if e.get("synonyms"):
                    st.write("**유의어:** " + ", ".join(e["synonyms"]))
                # "추가 예문"이 검색 실패 시의 범용 placeholder("Example using 'X'.")면
                # 코드(tools.py의 폴백 로직)는 그대로 두되, 학습자에게는 보여주지 않는다
                # (개발자가 폴백이 동작했는지 확인할 땐 tools.py에서 여전히 볼 수 있다)
                extra = (e.get("extra_examples") or [None])[0]
                if extra and not extra.startswith("Example using '"):
                    st.write("**추가 예문:** " + extra)

                # 🔊 발음 듣기 — st.empty() 슬롯 하나만 사용해서, 트리거 버튼과 재생 버튼이
                # 같은 자리를 "교체"하도록 만든다. 클릭한 바로 그 렌더 안에서도
                # slot.empty()로 버튼을 지운 뒤에 재생 버튼을 넣으므로, 두 개가
                # 동시에 화면에 남는 경우가 없다 (rerun 타이밍에 기대지 않는 방식).
                listen_key = f"listen_{i}"
                text_to_speak = e.get("example") or e["expression"]
                cached = ss["expr_audio"].get(listen_key)
                slot = st.empty()

                if cached:
                    with slot.container():
                        audio_play_button(cached, key=f"play_{listen_key}", label="🔊 다시 듣기")
                else:
                    clicked = slot.button("🔊 발음 듣기", key=f"btn_{listen_key}")
                    if clicked:
                        slot.empty()   # 버튼을 즉시 제거 — 같은 렌더 안에서도 사라짐
                        if tts_available:
                            with st.spinner("음성 생성 중... (gpt-4o-mini-tts)"):
                                tts_resp = api_post("/tts", {"text": text_to_speak, "persona": persona})
                            new_audio = tts_resp.get("audio_b64")
                            if new_audio:
                                ss["expr_audio"][listen_key] = new_audio
                                cached = new_audio
                        if cached:
                            with slot.container():
                                audio_play_button(cached, key=f"play_{listen_key}", autoplay=True)
                        elif voice_on:
                            browser_speak(text_to_speak, _BCP47.get(effective_target, "en-US"))

        if step_is_current("expressions"):
            if st.button("표현 학습 완료 → 퀴즈로 이동 ➡️", type="primary", use_container_width=True):
                goto_step("quiz")


# ═════════════════════════════════════════════════════════
# STEP 2: 객관식 퀴즈
# ═════════════════════════════════════════════════════════
if ss.get("analysis") and step_reached("quiz"):
    a = ss["analysis"]
    done = not step_is_current("quiz")
    with st.container(border=True):
        st.subheader(("✅ " if done else "3️⃣ ") + "퀴즈")

        if not done:
            selections = []
            for i, qz in enumerate(a["quiz"]):
                sel = st.radio(f"Q{i+1}. {qz['question']}", qz["choices"],
                               index=None, key=f"quiz_{i}")
                selections.append(sel)

            all_answered = all(s is not None for s in selections)
            if st.button("✅ 채점하고 역할극으로 이동", disabled=not all_answered,
                        type="primary", use_container_width=True):
                g = api_post("/quiz/grade", {"session_id": ss["session_id"], "answers": selections})
                ss["graded"] = g
                ss["quiz_selections"] = selections
                ss["learn_step"] = "roleplay"   # 채점과 동시에 다음 단계 언락
                st.rerun()
            if not all_answered:
                st.caption("모든 문제에 답하면 채점할 수 있어요.")
        elif ss.get("graded"):
            g = ss["graded"]
            sels = ss.get("quiz_selections", [])
            st.success(f"점수: {g['score']} / {g['total']}")
            for i, qz in enumerate(g["quiz"]):
                mine = sels[i] if i < len(sels) else "—"
                ok = str(mine).strip().lower() == str(qz["answer"]).strip().lower()
                st.write(f"{'✅' if ok else '❌'} 정답: **{qz['answer']}**  ·  내 답: {mine}")


# ═════════════════════════════════════════════════════════
# STEP 3: AI 파트너와 음성 역할극
# ═════════════════════════════════════════════════════════
if ss.get("analysis") and step_reached("roleplay"):
    done = not step_is_current("roleplay")
    session_max_turns = ss.get("max_turns", 3)
    with st.container(border=True):
        partner_name = PERSONA_LABEL.get(persona, persona)
        st.subheader(("✅ " if done else "4️⃣ ") + "AI 파트너와 역할극")
        st.caption(f"파트너: **{partner_name}** · 배운 표현을 실제로 말해보세요 "
                   f"(최대 {session_max_turns}턴)")
        debug_caption(f"roleplay_opened={ss.get('roleplay_opened')} · "
                     f"turn={ss.get('turn')}/{session_max_turns} · "
                     f"chat_len={len(ss.get('chat', []))}")

        # AI가 먼저 대화를 연다 — 이 인사가 화면에 나오기 전까지는 마이크 입력을 보여주지 않는다
        if not ss.get("roleplay_opened") and not done:
            with st.spinner(f"{partner_name}이(가) 대화를 시작하는 중..."):
                want_openai_audio = bool(voice_on and tts_available)
                r = api_post("/roleplay/start", {"session_id": ss["session_id"],
                                                 "speak": want_openai_audio})
            reply = r.get("partner_reply", "")
            if reply:
                ss["chat"].append(("ai", reply))
                ai_idx = len(ss["chat"]) - 1
                audio_b64 = r.get("audio_b64")
                if audio_b64:
                    ss["chat_audio_idx"][ai_idx] = audio_b64
                ss["pending_audio"] = audio_b64
                ss["pending_text"] = reply
                ss["pending_speak"] = bool(voice_on)
            ss["roleplay_opened"] = True

        for i, (role, text) in enumerate(ss["chat"]):
            with st.chat_message("user" if role == "me" else "assistant"):
                st.write(text)
                if role == "ai":
                    audio_b64 = ss.get("chat_audio_idx", {}).get(i)
                    if audio_b64:
                        is_latest = (i == len(ss["chat"]) - 1)
                        autoplay = is_latest and ss.get("pending_speak", False)
                        audio_play_button(audio_b64, key=f"replay_{i}", autoplay=autoplay)

        # 새 파트너 응답에 OpenAI 음성이 없을 때만(브라우저 폴백) 별도 처리.
        # 음성이 있으면 위 루프에서 같은 자리에 이미 재생 버튼이 그려졌으므로 여기서 더 안 그린다
        # → "다시 듣기" 버튼이 매번 같은 위치(말풍선 바로 아래)에 고정된다.
        if ss.get("pending_speak"):
            if not ss.get("pending_audio") and voice_on:
                browser_speak(ss.get("pending_text", ""), _BCP47.get(effective_target, "en-US"))
            ss["pending_speak"] = False

        def do_turn(user_text: str):
            if not user_text:
                return
            want_openai_audio = bool(voice_on and tts_available)
            r = api_post("/roleplay/turn", {"session_id": ss["session_id"],
                                            "message": user_text, "speak": want_openai_audio})
            reply = r.get("partner_reply", "")
            ss["chat"].append(("me", user_text))
            ss["chat"].append(("ai", reply))
            ai_idx = len(ss["chat"]) - 1
            audio_b64 = r.get("audio_b64")
            if audio_b64:
                ss.setdefault("chat_audio_idx", {})[ai_idx] = audio_b64
            ss["turn"] = r.get("turn", ss["turn"] + 1)
            ss["pending_audio"] = audio_b64
            ss["pending_text"] = reply
            ss["pending_speak"] = bool(voice_on)

        if not done and ss.get("roleplay_opened"):
            if ss["turn"] < session_max_turns:
                # 음성 전용 입력: 채팅창의 🎤 버튼으로 녹음 → 제출하면 자동으로 초기화됨
                # (st.audio_input과 달리 '트리거 위젯'이라 매 턴마다 새로 녹음 가능 —
                #  이전 방식에서 2턴 이후 더 말할 수 없던 문제가 여기서 해결된다)
                prompt = st.chat_input("🎤 마이크 버튼을 눌러 말해보세요 (텍스트 입력 불가)",
                                       accept_audio=True, audio_sample_rate=16000,
                                       submit_mode="disable",
                                       key=f"roleplay_mic_{ss['session_id']}")
                # accept_audio=True 여도 st.chat_input 의 텍스트 입력창 자체는 막히지
                # 않는다 (타이핑은 되고, 제출해야 비로소 경고가 뜨는 수준). 실제로
                # 타이핑 자체가 안 되도록 텍스트 영역을 readonly 로 만든다.
                components.html("""
                    <script>
                    const doc = window.parent.document;
                    const lock = () => {
                        doc.querySelectorAll('[data-testid="stChatInputTextArea"]')
                           .forEach(ta => { ta.readOnly = true; });
                    };
                    lock();
                    new MutationObserver(lock).observe(doc.body, {childList: true, subtree: true});
                    </script>
                """, height=0)
                if prompt:
                    if prompt.audio:
                        data = prompt.audio.getvalue()
                        with st.spinner("받아쓰는 중... (gpt-4o-mini-transcribe)"):
                            try:
                                resp = requests.post(
                                    f"{API}/stt",
                                    files={"file": ("speech.wav", data, "audio/wav")},
                                    data={"language": effective_target, "turn_index": ss["turn"]},
                                    timeout=120).json()
                            except Exception as e:
                                resp = {"text": None, "available": False}
                                st.error(f"STT 요청 실패: {e}")
                        text = resp.get("text")
                        if text:
                            if not resp.get("available"):
                                st.caption("🧪 데모 모드: 예시 발화를 사용했어요 (OPENAI_API_KEY 없음)")
                            do_turn(text)
                            st.rerun()
                        else:
                            st.warning("잘 안 들렸어요. 다시 말해 주세요.")
                    elif prompt.text:
                        st.warning("🎤 마이크 버튼을 눌러 음성으로 답해주세요. (이 역할극은 말하기 전용이에요)")
            else:
                st.info(f"역할극 {session_max_turns}턴 완료!")
                if st.button("역할극 완료 → 피드백으로 이동 ➡️", type="primary", use_container_width=True):
                    goto_step("feedback")


# ═════════════════════════════════════════════════════════
# STEP 4: 피드백 & 복습
# ═════════════════════════════════════════════════════════
if ss.get("analysis") and step_reached("feedback"):
    with st.container(border=True):
        st.subheader("5️⃣ 피드백 & 복습")
        if st.button("💬 역할극 피드백 받기", use_container_width=True):
            ss["feedback"] = api_post("/session/feedback",
                                      {"session_id": ss["session_id"], "mode": "roleplay"})

        if ss.get("feedback"):
            fb = ss["feedback"]
            if fb.get("llm_warning"):
                st.warning(f"⚠️ {fb['llm_warning']}")
            if fb.get("feedback"):
                st.text(fb["feedback"])
            if fb.get("corrections"):
                st.subheader("📝 어색했던 표현 교정")
                for c in fb["corrections"]:
                    with st.container(border=True):
                        st.write(f"~~{c.get('original','')}~~")
                        st.caption(c.get("issue", ""))
                        st.write(f"→ **{c.get('suggestion','')}**")
            if fb.get("review_list"):
                st.subheader("🔁 복습 예정")
                for r in fb["review_list"]:
                    st.write(f"- **{r['expression']}** ({r['meaning']}) → {r['review_after_days']}일 후")
                st.caption("사이드바의 '🃏 플래시카드로 복습'에서 영상별로 모아 복습할 수 있어요.")

# ── 하단: 학습 이력 ──────────────────────────────────────
with st.expander("📜 최근 학습 이력 보기"):
    hist = api_get(f"/user/{user_id}/history").get("history", [])
    for h in hist[:20]:
        st.caption(f"• {h['event']}")
