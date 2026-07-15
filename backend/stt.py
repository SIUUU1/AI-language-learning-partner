"""
stt.py — Speech-to-Text Transcription (gpt-4o-mini-transcribe)

In a role-play scenario, when the learner speaks into the microphone → this module converts the speech to text → and displays it as a script in the chat window.

- If `OPENAI_API_KEY` is provided, actual transcription is performed; otherwise, it returns `None` (→ the UI switches to text input).
- Falls back to `whisper-1` if `gpt-4o-mini-transcribe` fails.
"""
from __future__ import annotations

import io
from typing import Optional

from .config import OPENAI_API_KEY, USE_REAL_LLM

STT_MODEL = "gpt-4o-mini-transcribe"

# Target language → ISO-639-1 (transcription language hint, improves accuracy)
_ISO = {"English": "en", "日本語": "ja", "中文": "zh",
        "Español": "es", "Français": "fr", "한국어": "ko",
        "Japanese": "ja", "Chinese": "zh", "Spanish": "es",
        "French": "fr", "Korean": "ko", "German": "de"}


def available() -> bool:
    return USE_REAL_LLM and bool(OPENAI_API_KEY)


def iso_code(language: str) -> Optional[str]:
    return _ISO.get(language)


def transcribe(audio_bytes: bytes, filename: str = "speech.wav",
               language: Optional[str] = None) -> Optional[str]:
    """Audio bytes → text. Returns None if the key is missing or the operation fails."""
    if not audio_bytes or not available():
        return None
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    lang = iso_code(language) if language else None

    def _call(model: str) -> str:
        f = io.BytesIO(audio_bytes)
        f.name = filename                 # The SDK determines the format based on the file extension.
        kwargs = {"model": model, "file": f}
        if lang:
            kwargs["language"] = lang
        return client.audio.transcriptions.create(**kwargs).text.strip()

    try:
        return _call(STT_MODEL)
    except Exception as e:  # pragma: no cover
        print(f"[stt {STT_MODEL} Failure → whisper-1 fallback] {e}")
        try:
            return _call("whisper-1")
        except Exception as e2:
            print(f"[stt whisper-1 failure → Use text input] {e2}")
            return None
