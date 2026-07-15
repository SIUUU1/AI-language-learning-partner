"""
tts.py — Role-play Speech Synthesis (gpt-4o-mini-tts)

gpt-4o-mini-tts uses `instructions` to define speech styles based on the persona (e.g., friend = cheerful; interviewer = calm and formal; lover = affectionate).

- If `OPENAI_API_KEY` is provided, it performs actual synthesis (returning MP3 bytes); otherwise, it returns `None` →
  The UI then displays only text or falls back to the browser's built-in speech synthesis (Web Speech API).
- Per OpenAI policy, users must be informed that the audio is AI-generated, so a notification is included in the UI.
"""
from __future__ import annotations

from typing import Optional

from .config import OPENAI_API_KEY, USE_REAL_LLM

TTS_MODEL = "gpt-4o-mini-tts"

# Persona → (voice, tone instructions)
PERSONA_VOICE = {
    "friend":      ("coral",   "Speak in a warm, upbeat, casual tone, like chatting with a close friend over coffee."),
    "teacher":     ("sage",    "Speak clearly and patiently, enunciating each word, like a kind language teacher."),
    "interviewer": ("onyx",    "Speak in a calm, professional, measured tone, like a polite job interviewer."),
    "partner":     ("shimmer", "Speak softly and affectionately, like a caring significant other."),
    "barista":     ("nova",    "Speak in a friendly, energetic, welcoming tone, like a cheerful cafe barista."),
}
DEFAULT_VOICE = ("coral", "Speak in a clear, friendly, natural tone.")


def available() -> bool:
    """Actual TTS availability (for UI toggle determination)."""
    return USE_REAL_LLM and bool(OPENAI_API_KEY)


def synthesize(text: str, persona: str = "friend",
               speed: float = 1.0) -> Optional[bytes]:
    """Text → MP3 bytes. Returns None if the key is missing or the operation fails."""
    if not text or not available():
        return None
    voice, instructions = PERSONA_VOICE.get(persona, DEFAULT_VOICE)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text[:1900],           # Available input tokens (model limit: 2,000)
            instructions=instructions,
            response_format="mp3",
            speed=speed,
        )
        return resp.content              # bytes
    except Exception as e:  # pragma: no cover
        print(f"[tts fallback → no audio] {e}")
        return None
