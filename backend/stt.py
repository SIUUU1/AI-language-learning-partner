"""
stt.py — 음성 입력 받아쓰기
"""
from __future__ import annotations

import io
from typing import Optional

from .config import OPENAI_API_KEY, USE_REAL_LLM

STT_MODEL = "gpt-4o-mini-transcribe"

# 학습 언어 → ISO-639-1 (transcribe language 힌트, 정확도 향상)
_ISO = {"English": "en", "日本語": "ja", "中文": "zh",
        "Español": "es", "Français": "fr", "한국어": "ko",
        "Japanese": "ja", "Chinese": "zh", "Spanish": "es",
        "French": "fr", "Korean": "ko", "German": "de"}

# 데모 모드(키 없음)에서 마이크 녹음을 대신할 예시 학습자 발화 (턴 순서대로 회전)
_MOCK_LEARNER_UTTERANCES = [
    "Hi, I'll have a latte, please.",
    "Could you make it iced?",
    "To go, thanks.",
]


def available() -> bool:
    return USE_REAL_LLM and bool(OPENAI_API_KEY)


def iso_code(language: str) -> Optional[str]:
    return _ISO.get(language)


def transcribe(audio_bytes: bytes, filename: str = "speech.wav",
               language: Optional[str] = None, turn_index: int = 0) -> Optional[str]:
    """오디오 bytes → 텍스트.
    키가 없으면(데모 모드) 녹음이 있다는 전제 하에 예시 발화를 순서대로 돌려준다
    (역할극이 음성 전용이라 텍스트 폴백이 없기 때문 — 데모에서도 흐름 유지)."""
    if not audio_bytes:
        return None
    if not available():
        return _MOCK_LEARNER_UTTERANCES[turn_index % len(_MOCK_LEARNER_UTTERANCES)]

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    lang = iso_code(language) if language else None

    def _call(model: str) -> str:
        f = io.BytesIO(audio_bytes)
        f.name = filename                 # SDK 가 확장자로 포맷 판별
        kwargs = {"model": model, "file": f}
        if lang:
            kwargs["language"] = lang
        return client.audio.transcriptions.create(**kwargs).text.strip()

    try:
        return _call(STT_MODEL)
    except Exception as e:  # pragma: no cover
        print(f"[stt {STT_MODEL} 실패 → whisper-1 폴백] {e}")
        try:
            return _call("whisper-1")
        except Exception as e2:  # pragma: no cover
            print(f"[stt whisper-1 도 실패 → 데모 예시 발화 사용] {e2}")
            return _MOCK_LEARNER_UTTERANCES[turn_index % len(_MOCK_LEARNER_UTTERANCES)]
