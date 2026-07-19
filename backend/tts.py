"""
tts.py — 역할극 음성 합성 (gpt-4o-mini-tts)

gpt-4o-mini-tts 는 `instructions` 로 "어떻게 말할지"까지 지시할 수 있어
페르소나별 말투(친구=밝게 / 면접관=차분·격식 / 연인=다정하게)를 표현하기 좋다.

- OPENAI_API_KEY 있으면 실제 합성(mp3 bytes), 없으면 None 반환 →
  UI 는 텍스트만 표시하거나 브라우저 내장 음성(Web Speech API)으로 대체.
- OpenAI 정책상 "AI 합성 음성"임을 사용자에게 고지해야 하므로 UI 에 안내를 넣는다.
"""
from __future__ import annotations

from typing import Optional

from .config import OPENAI_API_KEY, USE_REAL_LLM

TTS_MODEL = "gpt-4o-mini-tts"

# 페르소나 → (voice, 말투 지시)
PERSONA_VOICE = {
    "friend":      ("coral",   "Speak in a warm, upbeat, casual tone, like chatting with a close friend over coffee."),
    "teacher":     ("sage",    "Speak clearly and patiently, enunciating each word, like a kind language teacher."),
    "interviewer": ("onyx",    "Speak in a calm, professional, measured tone, like a polite job interviewer."),
    "partner":     ("shimmer", "Speak softly and affectionately, like a caring significant other."),
    "barista":     ("nova",    "Speak in a friendly, energetic, welcoming tone, like a cheerful cafe barista."),
}
DEFAULT_VOICE = ("coral", "Speak in a clear, friendly, natural tone.")


def available() -> bool:
    """실제 TTS 사용 가능 여부 (UI 토글 판단용)."""
    return USE_REAL_LLM and bool(OPENAI_API_KEY)


def synthesize(text: str, persona: str = "friend",
               speed: float = 1.0) -> Optional[bytes]:
    """텍스트 → mp3 bytes. 키 없거나 실패 시 None."""
    if not text or not available():
        return None
    voice, instructions = PERSONA_VOICE.get(persona, DEFAULT_VOICE)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text[:1900],           # 입력 토큰 여유 (모델 상한 2000)
            instructions=instructions,
            response_format="mp3",
            speed=speed,
        )
        return resp.content              # bytes
    except Exception as e:  # pragma: no cover
        print(f"[tts fallback → no audio] {e}")
        return None
