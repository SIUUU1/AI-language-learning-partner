"""
streamlit_app.py — LinguaLoop UI
"""

from __future__ import annotations

import base64
import hashlib
import os
import requests
import streamlit as st
import streamlit.components.v1 as components

API = os.getenv("LINGUALOOP_API", "http://localhost:8000")

st.set_page_config(page_title="LinguaLoop 말문", page_icon="🎓", layout="centered")

# Target Language → Browser Speech (Web Speech API) Language Code
_BCP47 = {"English": "en-US", "日本語": "ja-JP", "中文": "zh-CN",
          "Español": "es-ES", "Français": "fr-FR"}


def browser_speak(text: str, lang: str = "en-US"):
    """Falls back to the browser's built-in speech synthesis if OpenAI TTS is unavailable (no API key required)."""
    safe = (text or "").replace("\\", " ").replace("`", " ").replace('"', " ")
    components.html(f"""
        <script>
        const u = new SpeechSynthesisUtterance("{safe}");
        u.lang = "{lang}";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        </script>""", height=0)


def api_get(path: str):
    try:
        return requests.get(f"{API}{path}", timeout=60).json()
    except Exception as e:
        st.error(f"Backend connection failed: {e}")
        return {}


def api_post(path: str, payload: dict):
    try:
        return requests.post(f"{API}{path}", json=payload, timeout=120).json()
    except Exception as e:
        st.error(f"Backend connection failed: {e}")
        return {}


# ── Initialize session state ──────────────────────────────────────
ss = st.session_state
ss.setdefault("step", "search")
ss.setdefault("session_id", None)
ss.setdefault("analysis", None)
ss.setdefault("chat", [])           # [(role, text)]
ss.setdefault("turn", 0)
ss.setdefault("graded", None)
ss.setdefault("feedback", None)
ss.setdefault("pending_speak", False)
ss.setdefault("pending_audio", None)
ss.setdefault("pending_text", "")
ss.setdefault("last_audio_hash", None)

# ── sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 LinguaLoop")
    st.caption("영상 → 표현 → 퀴즈 → 역할극 → 복습")

    health = api_get("/health")
    tts_available = False
    stt_available = False
    if health:
        m = health.get("mode", {})
        tts_available = health.get("tts_available", False)
        stt_available = health.get("stt_available", False)
        st.info(f"LLM: **{m.get('llm')}**  ·  YouTube: **{m.get('youtube')}**")
        personas = health.get("personas", ["friend", "teacher", "interviewer", "partner", "barista"])
    else:
        personas = ["friend", "teacher", "interviewer", "partner", "barista"]

    st.subheader("학습 설정")
    user_id = st.text_input("사용자 ID", value="learner-001")
    native = st.selectbox("모국어", ["한국어", "English", "日本語", "中文"], index=0)
    target = st.selectbox("학습 언어", ["English", "日本語", "中文", "Español", "Français"], index=0)
    persona = st.selectbox("역할극 파트너", personas,
                           format_func=lambda p: {"friend": "친구", "teacher": "선생님",
                                                  "interviewer": "면접관", "partner": "연인",
                                                  "barista": "바리스타"}.get(p, p))

    st.subheader("🔊 음성")
    voice_on = st.toggle("파트너 음성 재생", value=True)
    if voice_on:
        if tts_available:
            st.caption("gpt-4o-mini-tts · 페르소나별 말투 · **AI 합성 음성**")
        else:
            st.caption("OpenAI 키 없음 → 브라우저 내장 음성으로 재생 (**AI 합성 음성**)")
    st.caption("🎤 말하기(STT): " +
               ("gpt-4o-mini-transcribe 사용" if stt_available
                else "OpenAI 키 없음 → 텍스트 입력 사용"))

    st.divider()
    if st.button("🔄 새 학습 시작", use_container_width=True):
        for k in ["step", "session_id", "analysis", "chat", "turn", "graded",
                  "feedback", "pending_speak", "pending_audio", "pending_text",
                  "last_audio_hash"]:
            ss.pop(k, None)
        st.rerun()

    stats = api_get(f"/user/{user_id}/stats") if user_id else {}
    if stats:
        st.subheader("📊 내 학습 통계")
        c1, c2, c3 = st.columns(3)
        c1.metric("표현", stats.get("expressions_learned", 0))
        c2.metric("퀴즈", stats.get("quizzes_taken", 0))
        c3.metric("복습", stats.get("reviews_due", 0))


# ── STEP 1: Search & Select Video ─────────────────────────────
st.header("1️⃣ 유튜브 영상 선택")
q = st.text_input("검색어 또는 YouTube URL/ID",
                  placeholder="예: ordering coffee in english  /  https://youtu.be/...")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🔍 검색", use_container_width=True) and q:
        res = api_post("/youtube/search", {"query": q}).get("results", [])
        ss["search_results"] = res
with col_b:
    if st.button("▶️ 이 URL/ID로 바로 분석", use_container_width=True) and q:
        with st.spinner("영상 분석 중... (ContentAnalyzerAgent)"):
            a = api_post("/session/analyze", {
                "user_id": user_id, "url_or_id": q,
                "native_language": native, "target_language": target, "persona": persona})
        if "session_id" in a:
            ss["session_id"] = a["session_id"]
            ss["analysis"] = a
            ss["step"] = "learn"
            st.rerun()

for v in ss.get("search_results", []):
    with st.container(border=True):
        cols = st.columns([1, 3])
        if v.get("thumbnail"):
            cols[0].image(v["thumbnail"])
        cols[1].markdown(f"**{v['title']}**")
        cols[1].caption(f"{v['channel']}")
        if cols[1].button("이 영상으로 학습", key=f"pick_{v['video_id']}"):
            with st.spinner("영상 분석 중... (ContentAnalyzerAgent)"):
                a = api_post("/session/analyze", {
                    "user_id": user_id, "url_or_id": v["video_id"],
                    "native_language": native, "target_language": target, "persona": persona})
            if "session_id" in a:
                ss["session_id"] = a["session_id"]
                ss["analysis"] = a
                ss["step"] = "learn"
                st.rerun()


# ── STEP 2: Representation learning ─────────────────────────────────────
if ss.get("analysis"):
    a = ss["analysis"]
    st.header("2️⃣ 핵심 표현")
    st.caption(f"🎬 {a.get('video_title','')}  ·  🆕 새 표현 {a.get('new_expression_count',0)}개 "
               "(ChromaDB 기준)")
    for e in a["expressions"]:
        badge = "🆕" if e.get("is_new") else "🔁"
        with st.expander(f"{badge} {e['expression']} — {e['meaning']}"):
            st.write(f"**뜻:** {e['meaning']}")
            st.write(f"**예문:** {e.get('example','')}")
            if e.get("synonyms"):
                st.write("**유의어:** " + ", ".join(e["synonyms"]))
            if e.get("extra_examples"):
                st.write("**추가 예문:** " + e["extra_examples"][0])

    # ── STEP 3: Quiz ──
    st.header("3️⃣ 퀴즈")
    answers = []
    for i, qz in enumerate(a["quiz"]):
        answers.append(st.text_input(f"Q{i+1}. {qz['question']}", key=f"quiz_{i}"))
    if st.button("✅ 채점하기"):
        g = api_post("/quiz/grade", {"session_id": ss["session_id"], "answers": answers})
        ss["graded"] = g
    if ss.get("graded"):
        g = ss["graded"]
        st.success(f"점수: {g['score']} / {g['total']}")
        for i, qz in enumerate(g["quiz"]):
            mine = answers[i] if i < len(answers) else ""
            ok = str(mine).strip().lower() == str(qz["answer"]).strip().lower()
            st.write(f"{'✅' if ok else '❌'} 정답: **{qz['answer']}**  ·  내 답: {mine or '—'}")

    # ── STEP 4: Role-playing ──
    st.header("4️⃣ AI 파트너와 역할극")
    partner_name = {"friend": "친구", "teacher": "선생님", "interviewer": "면접관",
                    "partner": "연인", "barista": "바리스타"}.get(persona, persona)
    st.caption(f"파트너: **{partner_name}** · 배운 표현을 실제로 써보세요 (최대 3턴)")

    for role, text in ss["chat"]:
        with st.chat_message("user" if role == "me" else "assistant"):
            st.write(text)

    # If there is a response from the new partner, play it exactly once during this render.
    if ss.get("pending_speak"):
        if ss.get("pending_audio"):                 # OpenAI gpt-4o-mini-tts (mp3)
            st.audio(base64.b64decode(ss["pending_audio"]), format="audio/mp3", autoplay=True)
            st.caption("🔊 AI 합성 음성 (gpt-4o-mini-tts)")
        elif voice_on:                              # Browser-native speech replacement
            browser_speak(ss.get("pending_text", ""), _BCP47.get(target, "en-US"))
        ss["pending_speak"] = False                 # Prevent repeated playback upon restart

    def do_turn(user_text: str):
        """Text/Voice Common: Status update after one turn."""
        if not user_text:
            return
        want_openai_audio = bool(voice_on and tts_available)
        r = api_post("/roleplay/turn", {"session_id": ss["session_id"],
                                        "message": user_text, "speak": want_openai_audio})
        reply = r.get("partner_reply", "")
        ss["chat"].append(("me", user_text))
        ss["chat"].append(("ai", reply))
        ss["turn"] = r.get("turn", ss["turn"] + 1)
        ss["pending_audio"] = r.get("audio_b64")
        ss["pending_text"] = reply
        ss["pending_speak"] = bool(voice_on)

    if ss["turn"] < 3:
        # ① 텍스트 입력
        msg = st.chat_input("영어로 입력하거나, 아래 🎤 로 말해보세요...")
        if msg:
            do_turn(msg)
            st.rerun()

        # ② Voice Input (STT) — Automatically transcribes recordings and displays them in the chat.
        rec = st.audio_input("🎤 눌러서 말하기 (말한 뒤 잠깐 기다리면 받아써요)")
        if rec is not None:
            data = rec.getvalue()
            h = hashlib.md5(data).hexdigest()
            if data and h != ss.get("last_audio_hash"):
                ss["last_audio_hash"] = h              # 같은 녹음 재처리 방지
                with st.spinner("받아쓰는 중... (gpt-4o-mini-transcribe)"):
                    try:
                        resp = requests.post(
                            f"{API}/stt",
                            files={"file": ("speech.wav", data, "audio/wav")},
                            data={"language": target}, timeout=120).json()
                    except Exception as e:
                        resp = {"text": None, "available": False}
                        st.error(f"STT 요청 실패: {e}")
                text = resp.get("text")
                if text:
                    do_turn(text)
                    st.rerun()
                elif not resp.get("available"):
                    st.warning("음성 받아쓰기에는 OPENAI_API_KEY 가 필요해요. "
                               "위 채팅창에 직접 입력해 주세요.")
                else:
                    st.warning("잘 안 들렸어요. 다시 말해 주세요.")
    else:
        st.info("역할극 3턴 완료! 아래에서 피드백을 받아보세요.")

    # ── STEP 5: Feedback & Review ──
    st.header("5️⃣ 피드백 & 복습")
    colf1, colf2 = st.columns(2)
    if colf1.button("💬 역할극 피드백 받기", use_container_width=True):
        ss["feedback"] = api_post("/session/feedback",
                                  {"session_id": ss["session_id"], "mode": "roleplay"})
    if colf2.button("🃏 플래시카드 만들기", use_container_width=True):
        ss["feedback"] = api_post("/session/feedback",
                                  {"session_id": ss["session_id"], "mode": "flashcards"})

    if ss.get("feedback"):
        fb = ss["feedback"]
        if fb.get("feedback"):
            st.text(fb["feedback"])
        if fb.get("review_list"):
            st.subheader("🔁 복습 예정")
            for r in fb["review_list"]:
                st.write(f"- **{r['expression']}** ({r['meaning']}) → {r['review_after_days']}일 후")
        if fb.get("flashcards"):
            st.subheader("🃏 플래시카드")
            for card in fb["flashcards"]:
                with st.expander(card["front"]):
                    st.write(f"**뜻:** {card['back']}")
                    st.write(f"**예:** {card.get('example','')}")

# ── Bottom: Learning History ──────────────────────────────────────
with st.expander("📜 최근 학습 이력 보기"):
    hist = api_get(f"/user/{user_id}/history").get("history", [])
    for h in hist[:20]:
        st.caption(f"• {h['event']}")
